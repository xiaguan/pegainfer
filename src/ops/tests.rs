use anyhow::{Result, anyhow};
use cudarc::driver::CudaSlice;
use half::bf16;

use super::*;
use crate::model::qwen35::prefill_buffers::GdrChunkwiseScratch35;
use crate::tensor::*;

fn bf16_vec(data: &[f32]) -> Vec<bf16> {
    data.iter().map(|&x| bf16::from_f32(x)).collect()
}

fn rms_norm_reference(x: &[bf16], weight: &[bf16], eps: f32, offset: bool) -> Vec<f32> {
    let sum_sq: f32 = x
        .iter()
        .map(|value| {
            let v = value.to_f32();
            v * v
        })
        .sum();
    let inv_rms = 1.0 / ((sum_sq / x.len() as f32) + eps).sqrt();

    x.iter()
        .zip(weight.iter())
        .map(|(value, weight)| {
            let normed = bf16::from_f32(value.to_f32() * inv_rms).to_f32();
            let scale = if offset {
                1.0 + weight.to_f32()
            } else {
                weight.to_f32()
            };
            bf16::from_f32(normed * scale).to_f32()
        })
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= tol,
            "index {} expected {} got {} (tol {})",
            idx,
            expected,
            actual,
            tol
        );
    }
}

#[test]
fn test_gemv() -> Result<()> {
    let ctx = DeviceContext::new()?;

    // A = [[1, 2, 3], [4, 5, 6]] (2x3) row-major
    // x = [1, 2, 3]
    // y = A @ x = [1*1+2*2+3*3, 4*1+5*2+6*3] = [14, 32]
    let a_data = bf16_vec(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let x_data = bf16_vec(&[1.0, 2.0, 3.0]);

    let a = DeviceMatrix::from_host(&ctx, &a_data, 2, 3)?;
    let x = DeviceVec::from_host(&ctx, &x_data)?;
    let y = linear(&ctx, &x, &a)?;

    let result = y.to_host(&ctx)?;
    assert!(
        (result[0] - 14.0).abs() < 0.1,
        "Expected 14, got {}",
        result[0]
    );
    assert!(
        (result[1] - 32.0).abs() < 0.1,
        "Expected 32, got {}",
        result[1]
    );

    Ok(())
}

#[test]
fn test_argmax() -> Result<()> {
    let ctx = DeviceContext::new()?;
    let x = DeviceVec::from_host(&ctx, &bf16_vec(&[1.0, 9.0, 3.0, 8.0]))?;
    let token = argmax(&ctx, &x)?;
    assert_eq!(token, 1, "Expected argmax index 1, got {}", token);
    Ok(())
}

#[test]
fn test_rms_norm() -> Result<()> {
    let ctx = DeviceContext::new()?;

    let x_host = bf16_vec(&[1.0, 2.0, 3.0, 4.0]);
    let w_host = bf16_vec(&[1.0, 1.0, 1.0, 1.0]);
    let x = DeviceVec::from_host(&ctx, &x_host)?;
    let w = DeviceVec::from_host(&ctx, &w_host)?;
    let out = rms_norm(&ctx, &x, &w, 1e-6)?;

    let result = out.to_host(&ctx)?;
    let expected = rms_norm_reference(&x_host, &w_host, 1e-6, false);
    assert_close(&result, &expected, 0.01);

    Ok(())
}

#[test]
fn test_rms_norm_batch_multi_tile() -> Result<()> {
    let ctx = DeviceContext::new()?;
    let hidden_dim = 260;
    let seq_len = 2;

    let x_host_f32: Vec<f32> = (0..hidden_dim * seq_len)
        .map(|idx| ((idx % 17) as f32 - 8.0) * 0.25)
        .collect();
    let w_host_f32: Vec<f32> = (0..hidden_dim)
        .map(|idx| 0.5 + (idx % 11) as f32 * 0.0625)
        .collect();
    let x_host = bf16_vec(&x_host_f32);
    let w_host = bf16_vec(&w_host_f32);

    let x = HiddenStates {
        data: ctx
            .stream
            .clone_htod(&x_host)
            .map_err(|e| anyhow!("H2D copy failed: {}", e))?,
        hidden_dim,
        seq_len,
    };
    let weight = DeviceVec::from_host(&ctx, &w_host)?;
    let mut out = HiddenStates::zeros(&ctx, hidden_dim, seq_len)?;
    rms_norm_batch_into(&ctx, &x, &weight, 1e-6, &mut out);

    let result = ctx
        .stream
        .clone_dtoh(&out.data)
        .map_err(|e| anyhow!("D2H copy failed: {}", e))?;
    ctx.sync()?;
    let result: Vec<f32> = result.iter().map(|value| value.to_f32()).collect();

    let mut expected = Vec::with_capacity(hidden_dim * seq_len);
    for row in 0..seq_len {
        let start = row * hidden_dim;
        expected.extend(rms_norm_reference(
            &x_host[start..start + hidden_dim],
            &w_host,
            1e-6,
            false,
        ));
    }
    assert_close(&result, &expected, 0.02);

    Ok(())
}

#[test]
fn test_rms_norm_offset() -> Result<()> {
    let ctx = DeviceContext::new()?;

    let x_host = bf16_vec(&[-2.0, -0.5, 0.25, 1.5, 3.0, 0.75, -1.25]);
    let w_host = bf16_vec(&[0.0, 0.5, -0.25, 0.125, 1.0, -0.5, 0.25]);
    let x = DeviceVec::from_host(&ctx, &x_host)?;
    let w = DeviceVec::from_host(&ctx, &w_host)?;
    let mut out = DeviceVec::zeros(&ctx, x_host.len())?;
    rms_norm_offset_into(&ctx, &x, &w, 1e-6, &mut out)?;

    let result = out.to_host(&ctx)?;
    let expected = rms_norm_reference(&x_host, &w_host, 1e-6, true);
    assert_close(&result, &expected, 0.02);

    Ok(())
}

#[test]
fn test_embedding_variants() -> Result<()> {
    let ctx = DeviceContext::new()?;
    let embed = DeviceMatrix::from_host(
        &ctx,
        &bf16_vec(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ]),
        3,
        4,
    )?;

    let decode_meta = ctx.stream.clone_htod(&[1_i32])?;
    let mut decode_out = DeviceVec::zeros(&ctx, 4)?;
    embedding_decode_into(&ctx, &embed, &decode_meta, &mut decode_out)?;
    let decode_host = decode_out.to_host(&ctx)?;
    assert!(
        (decode_host[0] - 5.0).abs() < 0.01,
        "Expected 5.0, got {}",
        decode_host[0]
    );
    assert!(
        (decode_host[3] - 8.0).abs() < 0.01,
        "Expected 8.0, got {}",
        decode_host[3]
    );

    let token_ids = ctx.stream.clone_htod(&[2_i32, 0_i32])?;
    let mut batch_out = HiddenStates::zeros(&ctx, 4, 2)?;
    embedding_batch(&ctx, &embed, &token_ids, &mut batch_out)?;
    let batch_host = ctx.stream.clone_dtoh(&batch_out.data)?;
    ctx.sync()?;
    assert!(
        (batch_host[0].to_f32() - 9.0).abs() < 0.01,
        "Expected 9.0, got {}",
        batch_host[0]
    );
    assert!(
        (batch_host[3].to_f32() - 12.0).abs() < 0.01,
        "Expected 12.0, got {}",
        batch_host[3]
    );
    assert!(
        (batch_host[4].to_f32() - 1.0).abs() < 0.01,
        "Expected 1.0, got {}",
        batch_host[4]
    );
    assert!(
        (batch_host[7].to_f32() - 4.0).abs() < 0.01,
        "Expected 4.0, got {}",
        batch_host[7]
    );

    Ok(())
}

#[test]
fn test_gpu_sample() -> Result<()> {
    let ctx = DeviceContext::new()?;

    // Create logits with a clear winner at index 2 (highest logit)
    // but with temperature sampling, other tokens have a chance
    let logits_data = bf16_vec(&[1.0, 2.0, 10.0, 1.5, 0.5]);
    let logits = DeviceVec::from_host(&ctx, &logits_data)?;
    let mut probs: CudaSlice<f32> = ctx
        .stream
        .alloc_zeros(5)
        .map_err(|e| anyhow!("Alloc failed: {}", e))?;

    // Test 1: With very low temperature (near-greedy), should pick token 2
    let params = crate::sampler::SamplingParams {
        temperature: 0.01,
        top_k: -1,
        top_p: 1.0,
        ..Default::default()
    };
    let token = gpu_sample(&ctx, &logits, &mut probs, &params, 0.5)?;
    assert_eq!(token, 2, "near-greedy should pick index 2 (highest logit)");

    // Test 2: With high temperature, random_val=0.0 should pick first nonzero token
    let params = crate::sampler::SamplingParams {
        temperature: 1.0,
        top_k: -1,
        top_p: 1.0,
        ..Default::default()
    };
    let token = gpu_sample(&ctx, &logits, &mut probs, &params, 0.0)?;
    // random_val=0.0 should pick the first token (index 0)
    assert_eq!(token, 0, "random_val=0.0 should pick first token");

    // Test 3: top_k=1 should always pick the highest
    let params = crate::sampler::SamplingParams {
        temperature: 1.0,
        top_k: 1,
        top_p: 1.0,
        ..Default::default()
    };
    let token = gpu_sample(&ctx, &logits, &mut probs, &params, 0.5)?;
    assert_eq!(token, 2, "top_k=1 should pick highest probability token");

    Ok(())
}

#[test]
fn test_conv1d_prefill_handoff_matches_single_prefill() -> Result<()> {
    let ctx = DeviceContext::new()?;
    let num_channels = 1024usize;
    let kernel_size = 4usize;
    let total_seq = 18usize;
    let prefix_seq = 5usize;

    let x_host = bf16_vec(
        &(0..num_channels * total_seq)
            .map(|i| ((i % 71) as f32 - 35.0) * 0.03125)
            .collect::<Vec<_>>(),
    );
    let w_host = bf16_vec(
        &(0..num_channels * kernel_size)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.0625)
            .collect::<Vec<_>>(),
    );

    let x_all = HiddenStates {
        data: ctx.stream.clone_htod(&x_host)?,
        hidden_dim: num_channels,
        seq_len: total_seq,
    };
    let conv_weight = DeviceVec::from_host(&ctx, &w_host)?;

    let state_len = num_channels * (kernel_size - 1);
    let zero_state = vec![bf16::ZERO; state_len];

    let mut state_all = DeviceVec::from_host(&ctx, &zero_state)?;
    let mut out_all = HiddenStates::zeros(&ctx, num_channels, total_seq)?;
    conv1d_prefill_batch_into(
        &ctx,
        &x_all,
        &conv_weight,
        &mut state_all,
        &mut out_all,
        kernel_size,
    );

    let x_prefix = HiddenStates {
        data: ctx
            .stream
            .clone_htod(&x_host[..num_channels * prefix_seq])?,
        hidden_dim: num_channels,
        seq_len: prefix_seq,
    };
    let mut state_split = DeviceVec::from_host(&ctx, &zero_state)?;
    let mut out_prefix = HiddenStates::zeros(&ctx, num_channels, prefix_seq)?;
    conv1d_prefill_batch_into(
        &ctx,
        &x_prefix,
        &conv_weight,
        &mut state_split,
        &mut out_prefix,
        kernel_size,
    );

    for step in prefix_seq..total_seq {
        let x_step = HiddenStates {
            data: ctx
                .stream
                .clone_htod(&x_host[num_channels * step..num_channels * (step + 1)])?,
            hidden_dim: num_channels,
            seq_len: 1,
        };
        let mut out_step = HiddenStates::zeros(&ctx, num_channels, 1)?;
        conv1d_prefill_batch_into(
            &ctx,
            &x_step,
            &conv_weight,
            &mut state_split,
            &mut out_step,
            kernel_size,
        );
    }

    let out_all_host = ctx.stream.clone_dtoh(&out_all.data)?;
    let state_all_host = state_all.to_host(&ctx)?;
    let state_split_host = state_split.to_host(&ctx)?;
    ctx.sync()?;

    let out_all_host: Vec<f32> = out_all_host.iter().map(|x| x.to_f32()).collect();
    let expected_last = &out_all_host[num_channels * (total_seq - 1)..num_channels * total_seq];

    let x_last = HiddenStates {
        data: ctx
            .stream
            .clone_htod(&x_host[num_channels * (total_seq - 1)..num_channels * total_seq])?,
        hidden_dim: num_channels,
        seq_len: 1,
    };
    let mut state_last = DeviceVec::from_host(&ctx, &zero_state)?;
    let x_before_last = HiddenStates {
        data: ctx
            .stream
            .clone_htod(&x_host[..num_channels * (total_seq - 1)])?,
        hidden_dim: num_channels,
        seq_len: total_seq - 1,
    };
    let mut scratch_before_last = HiddenStates::zeros(&ctx, num_channels, total_seq - 1)?;
    conv1d_prefill_batch_into(
        &ctx,
        &x_before_last,
        &conv_weight,
        &mut state_last,
        &mut scratch_before_last,
        kernel_size,
    );
    let mut out_last = HiddenStates::zeros(&ctx, num_channels, 1)?;
    conv1d_prefill_batch_into(
        &ctx,
        &x_last,
        &conv_weight,
        &mut state_last,
        &mut out_last,
        kernel_size,
    );
    let out_last_host = ctx.stream.clone_dtoh(&out_last.data)?;
    ctx.sync()?;
    let out_last_host: Vec<f32> = out_last_host.iter().map(|x| x.to_f32()).collect();

    let max_out_diff = expected_last
        .iter()
        .zip(out_last_host.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let max_state_diff = state_all_host
        .iter()
        .zip(state_split_host.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    assert!(max_out_diff < 0.02, "output diff {max_out_diff}");
    assert!(max_state_diff < 0.02, "state diff {max_state_diff}");
    Ok(())
}

fn run_gdr_chunkwise_vs_recurrent(seq_len: usize) -> Result<(f32, f32)> {
    let ctx = DeviceContext::new()?;

    let num_key_heads = 16usize;
    let num_value_heads = 32usize;
    let key_dim = 128usize;
    let value_dim = 128usize;
    let qkv_dim = num_key_heads * key_dim * 2 + num_value_heads * value_dim;
    let z_dim = num_value_heads * value_dim;
    let state_len = num_value_heads * key_dim * value_dim;

    let qkv_host = bf16_vec(
        &(0..seq_len * qkv_dim)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.03125)
            .collect::<Vec<_>>(),
    );
    let b_host = bf16_vec(
        &(0..seq_len * num_value_heads)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.0625)
            .collect::<Vec<_>>(),
    );
    let a_host = bf16_vec(
        &(0..seq_len * num_value_heads)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.05)
            .collect::<Vec<_>>(),
    );
    let dt_bias_host = bf16_vec(
        &(0..num_value_heads)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.125)
            .collect::<Vec<_>>(),
    );
    let a_log_host: Vec<f32> = (0..num_value_heads)
        .map(|i| -1.0 + (i % 7) as f32 * 0.125)
        .collect();

    let qkv = HiddenStates {
        data: ctx.stream.clone_htod(&qkv_host)?,
        hidden_dim: qkv_dim,
        seq_len,
    };
    let b_proj = HiddenStates {
        data: ctx.stream.clone_htod(&b_host)?,
        hidden_dim: num_value_heads,
        seq_len,
    };
    let a_proj = HiddenStates {
        data: ctx.stream.clone_htod(&a_host)?,
        hidden_dim: num_value_heads,
        seq_len,
    };
    let dt_bias = DeviceVec::from_host(&ctx, &dt_bias_host)?;
    let a_log: CudaSlice<f32> = ctx.stream.clone_htod(&a_log_host)?;

    let zero_state = vec![0.0_f32; state_len];
    let mut state_chunkwise: CudaSlice<f32> = ctx.stream.clone_htod(&zero_state)?;
    let mut state_recurrent: CudaSlice<f32> = ctx.stream.clone_htod(&zero_state)?;
    let mut scratch =
        GdrChunkwiseScratch35::from_dims(&ctx, num_value_heads, key_dim, value_dim, seq_len)?;
    let mut out_chunkwise = HiddenStates::zeros(&ctx, z_dim, seq_len)?;

    gated_delta_rule_prefill_chunkwise_into(
        &ctx,
        &qkv,
        &b_proj,
        &a_proj,
        &dt_bias,
        &a_log,
        &mut state_chunkwise,
        &mut scratch,
        &mut out_chunkwise,
        num_key_heads,
        num_value_heads,
        key_dim,
        value_dim,
    )?;

    let mut out_recurrent_host = Vec::with_capacity(seq_len * z_dim);
    for token_idx in 0..seq_len {
        let qkv_t = DeviceVec::from_host(
            &ctx,
            &qkv_host[token_idx * qkv_dim..(token_idx + 1) * qkv_dim],
        )?;
        let b_t = DeviceVec::from_host(
            &ctx,
            &b_host[token_idx * num_value_heads..(token_idx + 1) * num_value_heads],
        )?;
        let a_t = DeviceVec::from_host(
            &ctx,
            &a_host[token_idx * num_value_heads..(token_idx + 1) * num_value_heads],
        )?;
        let mut out_t = DeviceVec::zeros(&ctx, z_dim)?;
        gated_delta_rule_decode_vec_into(
            &ctx,
            &qkv_t,
            &b_t,
            &a_t,
            &dt_bias,
            &a_log,
            &mut state_recurrent,
            &mut out_t,
            num_key_heads,
            num_value_heads,
            key_dim,
            value_dim,
        )?;
        out_recurrent_host.extend(out_t.to_host(&ctx)?);
    }

    let out_chunkwise_host: Vec<f32> = ctx
        .stream
        .clone_dtoh(&out_chunkwise.data)?
        .into_iter()
        .map(|x| x.to_f32())
        .collect();
    let state_chunkwise_host = ctx.stream.clone_dtoh(&state_chunkwise)?;
    let state_recurrent_host = ctx.stream.clone_dtoh(&state_recurrent)?;
    ctx.sync()?;

    let max_out_diff = out_chunkwise_host
        .iter()
        .zip(out_recurrent_host.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let max_state_diff = state_chunkwise_host
        .iter()
        .zip(state_recurrent_host.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    Ok((max_out_diff, max_state_diff))
}

#[test]
fn test_gdr_chunkwise_prefill_matches_recurrent_single_chunk() -> Result<()> {
    let (max_out_diff, max_state_diff) = run_gdr_chunkwise_vs_recurrent(64)?;
    assert!(max_out_diff < 0.05, "single-chunk output diff {max_out_diff}");
    assert!(max_state_diff < 0.05, "single-chunk state diff {max_state_diff}");
    Ok(())
}

#[test]
fn test_gdr_chunkwise_prefill_matches_recurrent_cross_chunk() -> Result<()> {
    let (max_out_diff, max_state_diff) = run_gdr_chunkwise_vs_recurrent(65)?;
    assert!(max_out_diff < 0.05, "cross-chunk output diff {max_out_diff}");
    assert!(max_state_diff < 0.05, "cross-chunk state diff {max_state_diff}");
    Ok(())
}

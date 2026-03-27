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
fn test_argmax_tie_prefers_smallest_index() -> Result<()> {
    let ctx = DeviceContext::new()?;
    let mut host = vec![bf16::from_f32(-1.0); 300];
    host[2] = bf16::from_f32(10.0);
    host[257] = bf16::from_f32(10.0);
    let x = DeviceVec::from_host(&ctx, &host)?;
    let token = argmax(&ctx, &x)?;
    assert_eq!(token, 2, "Expected smallest-index winner 2, got {}", token);
    Ok(())
}

#[test]
fn test_argmax_tie_prefers_smallest_index_across_thread_strides() -> Result<()> {
    let ctx = DeviceContext::new()?;
    let mut host = vec![bf16::from_f32(-1.0); 2048];
    host[727] = bf16::from_f32(20.0);
    host[1206] = bf16::from_f32(20.0);
    let x = DeviceVec::from_host(&ctx, &host)?;
    let token = argmax(&ctx, &x)?;
    assert_eq!(
        token, 727,
        "Expected smallest-index winner 727, got {}",
        token
    );
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
#[ignore = "slow reference comparison"]
fn test_flash_attention_prefill_hd256_matches_cpu_reference() -> Result<()> {
    let ctx = DeviceContext::new()?;
    let num_qheads = 4;
    let num_kvheads = 1;
    let head_dim = 256;
    let q_dim = num_qheads * head_dim;
    let gqa_ratio = num_qheads / num_kvheads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    for (start_pos, seq_len) in [(0_usize, 1_usize), (0, 6), (3, 6)] {
        let total_seq = start_pos + seq_len;
        let q_host_bf16 = bf16_vec(
            &(0..q_dim * seq_len)
                .map(|i| ((i % 41) as f32 - 20.0) * 0.0625)
                .collect::<Vec<_>>(),
        );
        let q_host: Vec<f32> = q_host_bf16.iter().map(|x| x.to_f32()).collect();

        let cache_len = num_kvheads * 4096 * head_dim;
        let mut k_cache_host_bf16 = vec![bf16::ZERO; cache_len];
        let mut v_cache_host_bf16 = vec![bf16::ZERO; cache_len];
        for kv_head in 0..num_kvheads {
            for pos in 0..total_seq {
                let base = (kv_head * 4096 + pos) * head_dim;
                for dim in 0..head_dim {
                    let k_val = (((kv_head * 31 + pos * 7 + dim) % 67) as f32 - 33.0) * 0.03125;
                    let v_val = (((kv_head * 19 + pos * 5 + dim) % 59) as f32 - 29.0) * 0.03125;
                    k_cache_host_bf16[base + dim] = bf16::from_f32(k_val);
                    v_cache_host_bf16[base + dim] = bf16::from_f32(v_val);
                }
            }
        }
        let k_cache_host: Vec<f32> = k_cache_host_bf16.iter().map(|x| x.to_f32()).collect();
        let v_cache_host: Vec<f32> = v_cache_host_bf16.iter().map(|x| x.to_f32()).collect();

        let q_batch = HiddenStates {
            data: ctx.stream.clone_htod(&q_host_bf16)?,
            hidden_dim: q_dim,
            seq_len,
        };
        let k_cache = DeviceVec::from_host(&ctx, &k_cache_host_bf16)?;
        let v_cache = DeviceVec::from_host(&ctx, &v_cache_host_bf16)?;
        let mut out = HiddenStates::zeros(&ctx, q_dim, seq_len)?;

        flash_attention_prefill_hd256_into(
            &ctx,
            &q_batch,
            &k_cache,
            &v_cache,
            &mut out,
            num_qheads,
            num_kvheads,
            start_pos,
        )?;

        let out_host_bf16 = ctx.stream.clone_dtoh(&out.data)?;
        ctx.sync()?;
        let out_host: Vec<f32> = out_host_bf16.iter().map(|x| x.to_f32()).collect();

        let mut ref_out = vec![0.0_f32; q_dim * seq_len];
        for token in 0..seq_len {
            let causal_end = start_pos + token;
            for q_head in 0..num_qheads {
                let kv_head = q_head / gqa_ratio;
                let q_base = token * q_dim + q_head * head_dim;
                let q_slice = &q_host[q_base..q_base + head_dim];

                let mut scores = vec![0.0_f32; causal_end + 1];
                for (pos, score) in scores.iter_mut().enumerate() {
                    let k_base = (kv_head * 4096 + pos) * head_dim;
                    *score = (0..head_dim)
                        .map(|dim| q_slice[dim] * k_cache_host[k_base + dim])
                        .sum::<f32>()
                        * scale;
                }

                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exp_scores: Vec<f32> = scores.iter().map(|x| (x - max_score).exp()).collect();
                let sum_exp = exp_scores.iter().sum::<f32>();
                let probs: Vec<f32> = exp_scores.iter().map(|x| x / sum_exp).collect();

                for dim in 0..head_dim {
                    let mut acc = 0.0_f32;
                    for (pos, prob) in probs.iter().enumerate() {
                        let v_base = (kv_head * 4096 + pos) * head_dim;
                        acc += prob * v_cache_host[v_base + dim];
                    }
                    ref_out[q_base + dim] = acc;
                }
            }
        }

        let max_out_diff = out_host
            .iter()
            .zip(ref_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_out_diff < 0.1,
            "start_pos={start_pos} seq_len={seq_len} output diff {max_out_diff}"
        );
    }

    Ok(())
}

#[test]
#[ignore = "slow reference comparison"]
fn test_prefill_attention_hd256_batch_matches_cpu_reference() -> Result<()> {
    let ctx = DeviceContext::new()?;
    let num_qheads = 4;
    let num_kvheads = 1;
    let head_dim = 256;
    let rotary_dim = 64;
    let q_dim = num_qheads * head_dim;
    let q_full_dim = q_dim * 2;
    let kv_dim = num_kvheads * head_dim;
    let gqa_ratio = num_qheads / num_kvheads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let eps = 1e-6_f32;

    let q_weight_host_bf16: Vec<bf16> = (0..head_dim)
        .map(|idx| bf16::from_f32(0.5 + (idx % 23) as f32 * 0.03125))
        .collect();
    let k_weight_host_bf16: Vec<bf16> = (0..head_dim)
        .map(|idx| bf16::from_f32(0.5 + (idx % 19) as f32 * 0.03125))
        .collect();
    let q_weight_host: Vec<f32> = q_weight_host_bf16.iter().map(|x| x.to_f32()).collect();
    let k_weight_host: Vec<f32> = k_weight_host_bf16.iter().map(|x| x.to_f32()).collect();

    let half_rotary = rotary_dim / 2;
    let theta = 10_000_000.0_f32;
    let inv_freq: Vec<f32> = (0..half_rotary)
        .map(|i| 1.0 / theta.powf(i as f32 * 2.0 / rotary_dim as f32))
        .collect();
    let mut cos_host = vec![bf16::ZERO; 4096 * rotary_dim];
    let mut sin_host = vec![bf16::ZERO; 4096 * rotary_dim];
    for pos in 0..4096 {
        for i in 0..half_rotary {
            let freq = pos as f32 * inv_freq[i];
            let cos = bf16::from_f32(freq.cos());
            let sin = bf16::from_f32(freq.sin());
            cos_host[pos * rotary_dim + i] = cos;
            cos_host[pos * rotary_dim + i + half_rotary] = cos;
            sin_host[pos * rotary_dim + i] = sin;
            sin_host[pos * rotary_dim + i + half_rotary] = sin;
        }
    }

    let rms_norm_offset = |x: &[f32], weight: &[f32]| -> Vec<f32> {
        let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        x.iter()
            .zip(weight.iter())
            .map(|(v, w)| v * inv * (1.0 + w))
            .collect()
    };

    let apply_partial_rope = |x: &[f32], pos: usize| -> Vec<f32> {
        let mut out = x.to_vec();
        for i in 0..half_rotary {
            let cos = cos_host[pos * rotary_dim + i].to_f32();
            let sin = sin_host[pos * rotary_dim + i].to_f32();
            let lo = x[i];
            let hi = x[i + half_rotary];
            out[i] = lo * cos - hi * sin;
            out[i + half_rotary] = lo * sin + hi * cos;
        }
        out
    };

    for (start_pos, seq_len) in [(0_usize, 1_usize), (0, 6), (3, 6)] {
        let q_full_host_bf16 = bf16_vec(
            &(0..q_full_dim * seq_len)
                .map(|i| ((i % 73) as f32 - 36.0) * 0.03125)
                .collect::<Vec<_>>(),
        );
        let q_full_host: Vec<f32> = q_full_host_bf16.iter().map(|x| x.to_f32()).collect();
        let k_batch_host_bf16 = bf16_vec(
            &(0..kv_dim * seq_len)
                .map(|i| ((i % 61) as f32 - 30.0) * 0.03125)
                .collect::<Vec<_>>(),
        );
        let v_batch_host_bf16 = bf16_vec(
            &(0..kv_dim * seq_len)
                .map(|i| ((i % 67) as f32 - 33.0) * 0.03125)
                .collect::<Vec<_>>(),
        );
        let k_batch_host: Vec<f32> = k_batch_host_bf16.iter().map(|x| x.to_f32()).collect();
        let v_batch_host: Vec<f32> = v_batch_host_bf16.iter().map(|x| x.to_f32()).collect();

        let cache_len = num_kvheads * 4096 * head_dim;
        let mut k_cache_init_bf16 = vec![bf16::ZERO; cache_len];
        let mut v_cache_init_bf16 = vec![bf16::ZERO; cache_len];
        for pos in 0..start_pos {
            let base = pos * head_dim;
            for dim in 0..head_dim {
                k_cache_init_bf16[base + dim] =
                    bf16::from_f32(((pos * 11 + dim) % 43) as f32 * 0.05 - 1.0);
                v_cache_init_bf16[base + dim] =
                    bf16::from_f32(((pos * 7 + dim) % 47) as f32 * 0.04 - 0.8);
            }
        }
        let mut ref_k_cache: Vec<f32> = k_cache_init_bf16.iter().map(|x| x.to_f32()).collect();
        let mut ref_v_cache: Vec<f32> = v_cache_init_bf16.iter().map(|x| x.to_f32()).collect();

        let q_full_batch = HiddenStates {
            data: ctx.stream.clone_htod(&q_full_host_bf16)?,
            hidden_dim: q_full_dim,
            seq_len,
        };
        let k_batch = HiddenStates {
            data: ctx.stream.clone_htod(&k_batch_host_bf16)?,
            hidden_dim: kv_dim,
            seq_len,
        };
        let v_batch = HiddenStates {
            data: ctx.stream.clone_htod(&v_batch_host_bf16)?,
            hidden_dim: kv_dim,
            seq_len,
        };
        let q_weight = DeviceVec::from_host(&ctx, &q_weight_host_bf16)?;
        let k_weight = DeviceVec::from_host(&ctx, &k_weight_host_bf16)?;
        let cos_cache = DeviceVec::from_host(&ctx, &cos_host)?;
        let sin_cache = DeviceVec::from_host(&ctx, &sin_host)?;
        let mut k_cache = DeviceVec::from_host(&ctx, &k_cache_init_bf16)?;
        let mut v_cache = DeviceVec::from_host(&ctx, &v_cache_init_bf16)?;
        let mut out = HiddenStates::zeros(&ctx, q_dim, seq_len)?;

        prefill_attention_hd256_batch(
            &ctx,
            &q_full_batch,
            &k_batch,
            &v_batch,
            &q_weight,
            &k_weight,
            &cos_cache,
            &sin_cache,
            &mut k_cache,
            &mut v_cache,
            &mut out,
            num_qheads,
            num_kvheads,
            start_pos,
            rotary_dim,
            eps,
        )?;

        let out_host_bf16 = ctx.stream.clone_dtoh(&out.data)?;
        let got_k_cache = k_cache.to_host(&ctx)?;
        let got_v_cache = v_cache.to_host(&ctx)?;
        let out_host: Vec<f32> = out_host_bf16.iter().map(|x| x.to_f32()).collect();

        let mut ref_out = vec![0.0_f32; q_dim * seq_len];
        for token in 0..seq_len {
            let pos = start_pos + token;

            for kv_head in 0..num_kvheads {
                let k_base = token * kv_dim + kv_head * head_dim;
                let k_head = &k_batch_host[k_base..k_base + head_dim];
                let k_normed = rms_norm_offset(k_head, &k_weight_host);
                let k_rot = apply_partial_rope(&k_normed, pos);
                let cache_base = (kv_head * 4096 + pos) * head_dim;
                ref_k_cache[cache_base..cache_base + head_dim].copy_from_slice(&k_rot);
                ref_v_cache[cache_base..cache_base + head_dim]
                    .copy_from_slice(&v_batch_host[k_base..k_base + head_dim]);
            }

            for q_head in 0..num_qheads {
                let q_base = token * q_full_dim + q_head * 2 * head_dim;
                let q_head_slice = &q_full_host[q_base..q_base + head_dim];
                let gate_slice = &q_full_host[q_base + head_dim..q_base + 2 * head_dim];
                let q_normed = rms_norm_offset(q_head_slice, &q_weight_host);
                let q_rot = apply_partial_rope(&q_normed, pos);
                let kv_head = q_head / gqa_ratio;

                let mut scores = vec![0.0_f32; pos + 1];
                for (cache_pos, score) in scores.iter_mut().enumerate() {
                    let k_base = (kv_head * 4096 + cache_pos) * head_dim;
                    *score = (0..head_dim)
                        .map(|dim| q_rot[dim] * ref_k_cache[k_base + dim])
                        .sum::<f32>()
                        * scale;
                }
                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exp_scores: Vec<f32> = scores.iter().map(|x| (x - max_score).exp()).collect();
                let sum_exp = exp_scores.iter().sum::<f32>();
                let probs: Vec<f32> = exp_scores.iter().map(|x| x / sum_exp).collect();

                let out_base = token * q_dim + q_head * head_dim;
                for dim in 0..head_dim {
                    let mut acc = 0.0_f32;
                    for (cache_pos, prob) in probs.iter().enumerate() {
                        let v_base = (kv_head * 4096 + cache_pos) * head_dim;
                        acc += prob * ref_v_cache[v_base + dim];
                    }
                    let sig_gate = 1.0 / (1.0 + (-gate_slice[dim]).exp());
                    ref_out[out_base + dim] = acc * sig_gate;
                }
            }
        }

        let max_k_diff = (0..num_kvheads * (start_pos + seq_len) * head_dim)
            .map(|idx| (got_k_cache[idx] - ref_k_cache[idx]).abs())
            .fold(0.0_f32, f32::max);
        let max_v_diff = (0..num_kvheads * (start_pos + seq_len) * head_dim)
            .map(|idx| (got_v_cache[idx] - ref_v_cache[idx]).abs())
            .fold(0.0_f32, f32::max);
        let max_out_diff = out_host
            .iter()
            .zip(ref_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        assert!(
            max_k_diff < 0.05,
            "start_pos={start_pos} seq_len={seq_len} k_cache diff {max_k_diff}"
        );
        assert!(
            max_v_diff < 0.02,
            "start_pos={start_pos} seq_len={seq_len} v_cache diff {max_v_diff}"
        );
        assert!(
            max_out_diff < 0.12,
            "start_pos={start_pos} seq_len={seq_len} output diff {max_out_diff}"
        );
    }

    Ok(())
}

#[test]
#[ignore = "slow handoff comparison"]
fn test_prefill_attention_hd256_handoff_matches_single_prefill() -> Result<()> {
    let ctx = DeviceContext::new()?;
    let num_qheads = 4;
    let num_kvheads = 1;
    let head_dim = 256;
    let rotary_dim = 64;
    let q_dim = num_qheads * head_dim;
    let q_full_dim = q_dim * 2;
    let kv_dim = num_kvheads * head_dim;
    let eps = 1e-6_f32;

    let q_weight_host_bf16: Vec<bf16> = (0..head_dim)
        .map(|idx| bf16::from_f32(0.5 + (idx % 23) as f32 * 0.03125))
        .collect();
    let k_weight_host_bf16: Vec<bf16> = (0..head_dim)
        .map(|idx| bf16::from_f32(0.5 + (idx % 19) as f32 * 0.03125))
        .collect();

    let half_rotary = rotary_dim / 2;
    let theta = 10_000_000.0_f32;
    let inv_freq: Vec<f32> = (0..half_rotary)
        .map(|i| 1.0 / theta.powf(i as f32 * 2.0 / rotary_dim as f32))
        .collect();
    let mut cos_host = vec![bf16::ZERO; 4096 * rotary_dim];
    let mut sin_host = vec![bf16::ZERO; 4096 * rotary_dim];
    for pos in 0..4096 {
        for i in 0..half_rotary {
            let freq = pos as f32 * inv_freq[i];
            let cos = bf16::from_f32(freq.cos());
            let sin = bf16::from_f32(freq.sin());
            cos_host[pos * rotary_dim + i] = cos;
            cos_host[pos * rotary_dim + i + half_rotary] = cos;
            sin_host[pos * rotary_dim + i] = sin;
            sin_host[pos * rotary_dim + i + half_rotary] = sin;
        }
    }

    let total_seq = 66usize;
    let prefix_seq = total_seq - 1;
    let q_full_host_bf16 = bf16_vec(
        &(0..q_full_dim * total_seq)
            .map(|i| ((i % 73) as f32 - 36.0) * 0.03125)
            .collect::<Vec<_>>(),
    );
    let k_batch_host_bf16 = bf16_vec(
        &(0..kv_dim * total_seq)
            .map(|i| ((i % 61) as f32 - 30.0) * 0.03125)
            .collect::<Vec<_>>(),
    );
    let v_batch_host_bf16 = bf16_vec(
        &(0..kv_dim * total_seq)
            .map(|i| ((i % 67) as f32 - 33.0) * 0.03125)
            .collect::<Vec<_>>(),
    );

    let q_weight = DeviceVec::from_host(&ctx, &q_weight_host_bf16)?;
    let k_weight = DeviceVec::from_host(&ctx, &k_weight_host_bf16)?;
    let cos_cache = DeviceVec::from_host(&ctx, &cos_host)?;
    let sin_cache = DeviceVec::from_host(&ctx, &sin_host)?;
    let cache_len = num_kvheads * 4096 * head_dim;
    let zero_cache = vec![bf16::ZERO; cache_len];

    let q_full_all = HiddenStates {
        data: ctx.stream.clone_htod(&q_full_host_bf16)?,
        hidden_dim: q_full_dim,
        seq_len: total_seq,
    };
    let k_all = HiddenStates {
        data: ctx.stream.clone_htod(&k_batch_host_bf16)?,
        hidden_dim: kv_dim,
        seq_len: total_seq,
    };
    let v_all = HiddenStates {
        data: ctx.stream.clone_htod(&v_batch_host_bf16)?,
        hidden_dim: kv_dim,
        seq_len: total_seq,
    };
    let mut k_cache_all = DeviceVec::from_host(&ctx, &zero_cache)?;
    let mut v_cache_all = DeviceVec::from_host(&ctx, &zero_cache)?;
    let mut out_all = HiddenStates::zeros(&ctx, q_dim, total_seq)?;
    prefill_attention_hd256_batch(
        &ctx,
        &q_full_all,
        &k_all,
        &v_all,
        &q_weight,
        &k_weight,
        &cos_cache,
        &sin_cache,
        &mut k_cache_all,
        &mut v_cache_all,
        &mut out_all,
        num_qheads,
        num_kvheads,
        0,
        rotary_dim,
        eps,
    )?;

    let q_full_prefix = HiddenStates {
        data: ctx
            .stream
            .clone_htod(&q_full_host_bf16[..q_full_dim * prefix_seq])?,
        hidden_dim: q_full_dim,
        seq_len: prefix_seq,
    };
    let k_prefix = HiddenStates {
        data: ctx
            .stream
            .clone_htod(&k_batch_host_bf16[..kv_dim * prefix_seq])?,
        hidden_dim: kv_dim,
        seq_len: prefix_seq,
    };
    let v_prefix = HiddenStates {
        data: ctx
            .stream
            .clone_htod(&v_batch_host_bf16[..kv_dim * prefix_seq])?,
        hidden_dim: kv_dim,
        seq_len: prefix_seq,
    };
    let mut k_cache_split = DeviceVec::from_host(&ctx, &zero_cache)?;
    let mut v_cache_split = DeviceVec::from_host(&ctx, &zero_cache)?;
    let mut out_prefix = HiddenStates::zeros(&ctx, q_dim, prefix_seq)?;
    prefill_attention_hd256_batch(
        &ctx,
        &q_full_prefix,
        &k_prefix,
        &v_prefix,
        &q_weight,
        &k_weight,
        &cos_cache,
        &sin_cache,
        &mut k_cache_split,
        &mut v_cache_split,
        &mut out_prefix,
        num_qheads,
        num_kvheads,
        0,
        rotary_dim,
        eps,
    )?;

    let q_full_next = HiddenStates {
        data: ctx
            .stream
            .clone_htod(&q_full_host_bf16[q_full_dim * prefix_seq..q_full_dim * total_seq])?,
        hidden_dim: q_full_dim,
        seq_len: 1,
    };
    let k_next = HiddenStates {
        data: ctx
            .stream
            .clone_htod(&k_batch_host_bf16[kv_dim * prefix_seq..kv_dim * total_seq])?,
        hidden_dim: kv_dim,
        seq_len: 1,
    };
    let v_next = HiddenStates {
        data: ctx
            .stream
            .clone_htod(&v_batch_host_bf16[kv_dim * prefix_seq..kv_dim * total_seq])?,
        hidden_dim: kv_dim,
        seq_len: 1,
    };
    let mut out_next = HiddenStates::zeros(&ctx, q_dim, 1)?;
    prefill_attention_hd256_batch(
        &ctx,
        &q_full_next,
        &k_next,
        &v_next,
        &q_weight,
        &k_weight,
        &cos_cache,
        &sin_cache,
        &mut k_cache_split,
        &mut v_cache_split,
        &mut out_next,
        num_qheads,
        num_kvheads,
        prefix_seq,
        rotary_dim,
        eps,
    )?;

    let out_all_host = ctx.stream.clone_dtoh(&out_all.data)?;
    let out_next_host = ctx.stream.clone_dtoh(&out_next.data)?;
    let k_cache_all_host = k_cache_all.to_host(&ctx)?;
    let v_cache_all_host = v_cache_all.to_host(&ctx)?;
    let k_cache_split_host = k_cache_split.to_host(&ctx)?;
    let v_cache_split_host = v_cache_split.to_host(&ctx)?;
    ctx.sync()?;

    let out_all_host: Vec<f32> = out_all_host.iter().map(|x| x.to_f32()).collect();
    let out_next_host: Vec<f32> = out_next_host.iter().map(|x| x.to_f32()).collect();
    let out_all_last = &out_all_host[q_dim * prefix_seq..q_dim * total_seq];
    let max_out_diff = out_all_last
        .iter()
        .zip(out_next_host.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let max_k_diff = (0..num_kvheads * total_seq * head_dim)
        .map(|idx| (k_cache_all_host[idx] - k_cache_split_host[idx]).abs())
        .fold(0.0_f32, f32::max);
    let max_v_diff = (0..num_kvheads * total_seq * head_dim)
        .map(|idx| (v_cache_all_host[idx] - v_cache_split_host[idx]).abs())
        .fold(0.0_f32, f32::max);

    assert!(max_k_diff < 0.02, "k_cache diff {max_k_diff}");
    assert!(max_v_diff < 0.02, "v_cache diff {max_v_diff}");
    assert!(max_out_diff < 0.1, "output diff {max_out_diff}");

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

#[test]
fn test_conv1d_prefill_seq1_matches_decode() -> Result<()> {
    let ctx = DeviceContext::new()?;
    let num_channels = 1024usize;
    let kernel_size = 4usize;

    let x_host = bf16_vec(
        &(0..num_channels)
            .map(|i| ((i % 71) as f32 - 35.0) * 0.03125)
            .collect::<Vec<_>>(),
    );
    let w_host = bf16_vec(
        &(0..num_channels * kernel_size)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.0625)
            .collect::<Vec<_>>(),
    );
    let state_host = bf16_vec(
        &(0..num_channels * (kernel_size - 1))
            .map(|i| ((i % 23) as f32 - 11.0) * 0.03125)
            .collect::<Vec<_>>(),
    );

    let x_vec = DeviceVec::from_host(&ctx, &x_host)?;
    let x_seq = HiddenStates {
        data: ctx.stream.clone_htod(&x_host)?,
        hidden_dim: num_channels,
        seq_len: 1,
    };
    let conv_weight = DeviceVec::from_host(&ctx, &w_host)?;

    let mut state_decode = DeviceVec::from_host(&ctx, &state_host)?;
    let mut out_decode = DeviceVec::zeros(&ctx, num_channels)?;
    conv1d_decode_into(
        &ctx,
        &x_vec,
        &conv_weight,
        &mut state_decode,
        &mut out_decode,
        kernel_size,
    )?;

    let mut state_prefill = DeviceVec::from_host(&ctx, &state_host)?;
    let mut out_prefill = HiddenStates::zeros(&ctx, num_channels, 1)?;
    conv1d_prefill_batch_into(
        &ctx,
        &x_seq,
        &conv_weight,
        &mut state_prefill,
        &mut out_prefill,
        kernel_size,
    );

    let out_decode_host = out_decode.to_host(&ctx)?;
    let out_prefill_host_bf16 = ctx.stream.clone_dtoh(&out_prefill.data)?;
    let state_decode_host = state_decode.to_host(&ctx)?;
    let state_prefill_host = state_prefill.to_host(&ctx)?;
    ctx.sync()?;
    let out_prefill_host: Vec<f32> = out_prefill_host_bf16.iter().map(|x| x.to_f32()).collect();

    let max_out_diff = out_decode_host
        .iter()
        .zip(out_prefill_host.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let max_state_diff = state_decode_host
        .iter()
        .zip(state_prefill_host.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    assert!(max_out_diff < 0.02, "output diff {max_out_diff}");
    assert!(max_state_diff < 0.02, "state diff {max_state_diff}");
    Ok(())
}

#[test]
#[ignore = "slow reference comparison"]
fn test_gated_delta_rule_prefill_chunkwise_matches_decode_reference() -> Result<()> {
    let ctx = DeviceContext::new()?;
    let num_key_heads = 2usize;
    let num_value_heads = 4usize;
    let key_dim = 128usize;
    let val_dim = 128usize;
    let seq_len = 5usize;
    let qkv_dim = num_key_heads * key_dim * 2 + num_value_heads * val_dim;
    let out_dim = num_value_heads * val_dim;
    let state_len = num_value_heads * key_dim * val_dim;

    let qkv_host = bf16_vec(
        &(0..qkv_dim * seq_len)
            .map(|i| ((i % 37) as f32 - 18.0) * 0.03125)
            .collect::<Vec<_>>(),
    );
    let b_host = bf16_vec(
        &(0..num_value_heads * seq_len)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.125)
            .collect::<Vec<_>>(),
    );
    let a_host = bf16_vec(
        &(0..num_value_heads * seq_len)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.09375)
            .collect::<Vec<_>>(),
    );
    let dt_bias_host = bf16_vec(
        &(0..num_value_heads)
            .map(|i| 0.2 + i as f32 * 0.05)
            .collect::<Vec<_>>(),
    );
    let a_log_host: Vec<f32> = (0..num_value_heads)
        .map(|i| -1.5 + i as f32 * 0.125)
        .collect();
    let state_init_host: Vec<f32> = (0..state_len)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.002)
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
    let a_log = ctx.stream.clone_htod(&a_log_host)?;

    let mut state_prefill = ctx.stream.clone_htod(&state_init_host)?;
    let mut out_prefill = HiddenStates::zeros(&ctx, out_dim, seq_len)?;
    let mut scratch =
        GdrChunkwiseScratch35::from_dims(&ctx, num_value_heads, key_dim, val_dim, seq_len)?;
    gated_delta_rule_prefill_chunkwise_into(
        &ctx,
        &qkv,
        &b_proj,
        &a_proj,
        &dt_bias,
        &a_log,
        &mut state_prefill,
        &mut scratch,
        &mut out_prefill,
        num_key_heads,
        num_value_heads,
        key_dim,
        val_dim,
    )?;

    let mut state_decode = ctx.stream.clone_htod(&state_init_host)?;
    let mut out_decode = HiddenStates::zeros(&ctx, out_dim, seq_len)?;
    for t in 0..seq_len {
        let qkv_t = extract_vec(&ctx, &qkv, t)?;
        let b_t = extract_vec(&ctx, &b_proj, t)?;
        let a_t = extract_vec(&ctx, &a_proj, t)?;
        let mut out_t = DeviceVec::zeros(&ctx, out_dim)?;
        gated_delta_rule_decode_into(
            &ctx,
            &qkv_t,
            &b_t,
            &a_t,
            &dt_bias,
            &a_log,
            &mut state_decode,
            &mut out_t,
            num_key_heads,
            num_value_heads,
            key_dim,
            val_dim,
        )?;
        let offset = t * out_dim;
        let mut dst_view = out_decode.data.slice_mut(offset..offset + out_dim);
        ctx.stream
            .memcpy_dtod(&out_t.data, &mut dst_view)
            .map_err(|e| anyhow!("Device copy failed: {}", e))?;
    }

    let out_prefill_host = ctx.stream.clone_dtoh(&out_prefill.data)?;
    let out_decode_host = ctx.stream.clone_dtoh(&out_decode.data)?;
    let state_prefill_host = ctx.stream.clone_dtoh(&state_prefill)?;
    let state_decode_host = ctx.stream.clone_dtoh(&state_decode)?;
    ctx.sync()?;

    let max_out_diff = out_prefill_host
        .iter()
        .zip(out_decode_host.iter())
        .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
        .fold(0.0_f32, f32::max);
    let max_state_diff = state_prefill_host
        .iter()
        .zip(state_decode_host.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    assert!(max_out_diff < 0.05, "output diff {max_out_diff}");
    assert!(max_state_diff < 0.01, "state diff {max_state_diff}");

    Ok(())
}

#[test]
#[ignore = "slow reference comparison"]
fn test_triton_decode_attention_matches_cpu_reference() -> Result<()> {
    let ctx = DeviceContext::new()?;
    let num_qheads = 8;
    let num_kvheads = 2;
    let head_dim = 128;
    let max_seq_len = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let eps = 1e-6_f32;

    let q_host: Vec<f32> = (0..num_qheads * head_dim)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.125)
        .collect();
    let k_host: Vec<f32> = (0..num_kvheads * head_dim)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.15)
        .collect();
    let v_host: Vec<f32> = (0..num_kvheads * head_dim)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.2)
        .collect();
    let q_weight_host: Vec<f32> = (0..head_dim)
        .map(|i| 0.9 + (i % 13) as f32 * 0.03)
        .collect();
    let k_weight_host: Vec<f32> = (0..head_dim)
        .map(|i| 0.8 + (i % 11) as f32 * 0.025)
        .collect();

    let half = head_dim / 2;
    let theta = 1_000_000.0_f32;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0 / theta.powf(i as f32 * 2.0 / head_dim as f32))
        .collect();
    let mut cos_host = vec![bf16::ZERO; max_seq_len * head_dim];
    let mut sin_host = vec![bf16::ZERO; max_seq_len * head_dim];
    for pos in 0..max_seq_len {
        for i in 0..half {
            let freq = pos as f32 * inv_freq[i];
            let cos = bf16::from_f32(freq.cos());
            let sin = bf16::from_f32(freq.sin());
            cos_host[pos * head_dim + i] = cos;
            cos_host[pos * head_dim + i + half] = cos;
            sin_host[pos * head_dim + i] = sin;
            sin_host[pos * head_dim + i + half] = sin;
        }
    }

    for seq_len in [1_usize, 6_usize] {
        let current_pos = seq_len - 1;
        let cache_len = num_kvheads * 4096 * head_dim;
        let mut k_cache_host = vec![bf16::ZERO; cache_len];
        let mut v_cache_host = vec![bf16::ZERO; cache_len];
        for kv_head in 0..num_kvheads {
            for pos in 0..current_pos {
                let base = (kv_head * 4096 + pos) * head_dim;
                for dim in 0..head_dim {
                    k_cache_host[base + dim] =
                        bf16::from_f32(((kv_head * 31 + pos * 7 + dim) % 41) as f32 * 0.05 - 1.0);
                    v_cache_host[base + dim] =
                        bf16::from_f32(((kv_head * 17 + pos * 5 + dim) % 37) as f32 * 0.04 - 0.7);
                }
            }
        }

        let q = DeviceVec::from_host(&ctx, &bf16_vec(&q_host))?;
        let k = DeviceVec::from_host(&ctx, &bf16_vec(&k_host))?;
        let v = DeviceVec::from_host(&ctx, &bf16_vec(&v_host))?;
        let q_weight = DeviceVec::from_host(&ctx, &bf16_vec(&q_weight_host))?;
        let k_weight = DeviceVec::from_host(&ctx, &bf16_vec(&k_weight_host))?;
        let cos_cache = DeviceVec::from_host(&ctx, &cos_host)?;
        let sin_cache = DeviceVec::from_host(&ctx, &sin_host)?;
        let decode_meta = ctx
            .stream
            .clone_htod(&[0_i32, current_pos as i32, seq_len as i32])?;
        let mut k_cache = DeviceVec::from_host(&ctx, &k_cache_host)?;
        let mut v_cache = DeviceVec::from_host(&ctx, &v_cache_host)?;
        let mut out = DeviceVec::zeros(&ctx, num_qheads * head_dim)?;
        let num_kv_splits = 4usize;
        let mut partial_out: CudaSlice<f32> = ctx
            .stream
            .alloc_zeros(num_qheads * num_kv_splits * head_dim)?;
        let mut partial_m: CudaSlice<f32> = ctx.stream.alloc_zeros(num_qheads * num_kv_splits)?;
        let mut partial_l: CudaSlice<f32> = ctx.stream.alloc_zeros(num_qheads * num_kv_splits)?;

        fused_attention_decode_into(
            &ctx,
            &q,
            &k,
            &v,
            &q_weight,
            &k_weight,
            &cos_cache,
            &sin_cache,
            &decode_meta,
            &mut k_cache,
            &mut v_cache,
            &mut out,
            &mut partial_out,
            &mut partial_m,
            &mut partial_l,
            num_qheads,
            num_kvheads,
        )?;

        let out_host = out.to_host(&ctx)?;
        let got_k_cache = k_cache.to_host(&ctx)?;
        let got_v_cache = v_cache.to_host(&ctx)?;

        let q_heads: Vec<Vec<f32>> = q_host.chunks(head_dim).map(<[f32]>::to_vec).collect();
        let k_heads: Vec<Vec<f32>> = k_host.chunks(head_dim).map(<[f32]>::to_vec).collect();
        let v_heads: Vec<Vec<f32>> = v_host.chunks(head_dim).map(<[f32]>::to_vec).collect();
        let gqa_ratio = num_qheads / num_kvheads;

        let mut ref_k_cache: Vec<f32> = k_cache_host.iter().map(|x| x.to_f32()).collect();
        let mut ref_v_cache: Vec<f32> = v_cache_host.iter().map(|x| x.to_f32()).collect();
        let mut ref_out = vec![0.0_f32; num_qheads * head_dim];

        let rms_norm = |head: &[f32], weight: &[f32]| -> Vec<f32> {
            let mean_sq = head.iter().map(|x| x * x).sum::<f32>() / head.len() as f32;
            let inv = 1.0 / (mean_sq + eps).sqrt();
            head.iter()
                .zip(weight.iter())
                .map(|(x, w)| x * inv * w)
                .collect()
        };

        let apply_rope = |head: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0_f32; head_dim];
            for i in 0..half {
                let cos = cos_host[current_pos * head_dim + i].to_f32();
                let sin = sin_host[current_pos * head_dim + i].to_f32();
                let lo = head[i];
                let hi = head[i + half];
                out[i] = lo * cos - hi * sin;
                out[i + half] = lo * sin + hi * cos;
            }
            out
        };

        for kv_head in 0..num_kvheads {
            let k_rot = apply_rope(&rms_norm(&k_heads[kv_head], &k_weight_host));
            let base = (kv_head * 4096 + current_pos) * head_dim;
            ref_k_cache[base..base + head_dim].copy_from_slice(&k_rot);
            ref_v_cache[base..base + head_dim].copy_from_slice(&v_heads[kv_head]);
        }

        for q_head in 0..num_qheads {
            let kv_head = q_head / gqa_ratio;
            let q_rot = apply_rope(&rms_norm(&q_heads[q_head], &q_weight_host));
            let mut scores = vec![0.0_f32; seq_len];
            for (pos, score) in scores.iter_mut().enumerate() {
                let base = (kv_head * 4096 + pos) * head_dim;
                *score = (0..head_dim)
                    .map(|dim| ref_k_cache[base + dim] * q_rot[dim])
                    .sum::<f32>()
                    * scale;
            }
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_scores: Vec<f32> = scores.iter().map(|x| (x - max_score).exp()).collect();
            let sum_exp = exp_scores.iter().sum::<f32>();
            let probs: Vec<f32> = exp_scores.iter().map(|x| x / sum_exp).collect();
            for dim in 0..head_dim {
                let mut acc = 0.0_f32;
                for (pos, prob) in probs.iter().enumerate() {
                    let base = (kv_head * 4096 + pos) * head_dim;
                    acc += prob * ref_v_cache[base + dim];
                }
                ref_out[q_head * head_dim + dim] = acc;
            }
        }

        let current_base = current_pos * head_dim;
        let max_k_diff = (0..num_kvheads * head_dim)
            .map(|idx| {
                let kv_head = idx / head_dim;
                let dim = idx % head_dim;
                let offset = kv_head * 4096 * head_dim + current_base + dim;
                (got_k_cache[offset] - ref_k_cache[offset]).abs()
            })
            .fold(0.0_f32, f32::max);
        let max_v_diff = (0..num_kvheads * head_dim)
            .map(|idx| {
                let kv_head = idx / head_dim;
                let dim = idx % head_dim;
                let offset = kv_head * 4096 * head_dim + current_base + dim;
                (got_v_cache[offset] - ref_v_cache[offset]).abs()
            })
            .fold(0.0_f32, f32::max);
        let max_out_diff = out_host
            .iter()
            .zip(ref_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        assert!(
            max_k_diff < 0.02,
            "seq_len={seq_len} k_cache diff {max_k_diff}"
        );
        assert!(
            max_v_diff < 0.02,
            "seq_len={seq_len} v_cache diff {max_v_diff}"
        );
        assert!(
            max_out_diff < 0.1,
            "seq_len={seq_len} output diff {max_out_diff}"
        );
    }

    Ok(())
}

#[test]
#[ignore = "slow reference comparison"]
fn test_fused_attention_hd256_decode_matches_cpu_reference() -> Result<()> {
    let ctx = DeviceContext::new()?;
    let num_qheads = 16;
    let num_kvheads = 4;
    let head_dim = 256;
    let rotary_dim = 64;
    let q_dim = num_qheads * head_dim;
    let q_full_dim = q_dim * 2;
    let kv_dim = num_kvheads * head_dim;
    let gqa_ratio = num_qheads / num_kvheads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let eps = 1e-6_f32;
    let max_seq_len = 4096;

    let q_weight_host_bf16: Vec<bf16> = (0..head_dim)
        .map(|idx| bf16::from_f32(0.5 + (idx % 23) as f32 * 0.03125))
        .collect();
    let k_weight_host_bf16: Vec<bf16> = (0..head_dim)
        .map(|idx| bf16::from_f32(0.5 + (idx % 19) as f32 * 0.03125))
        .collect();
    let q_weight_host: Vec<f32> = q_weight_host_bf16.iter().map(|x| x.to_f32()).collect();
    let k_weight_host: Vec<f32> = k_weight_host_bf16.iter().map(|x| x.to_f32()).collect();

    let half_rotary = rotary_dim / 2;
    let theta = 10_000_000.0_f32;
    let inv_freq: Vec<f32> = (0..half_rotary)
        .map(|i| 1.0 / theta.powf(i as f32 * 2.0 / rotary_dim as f32))
        .collect();
    let mut cos_host = vec![bf16::ZERO; max_seq_len * rotary_dim];
    let mut sin_host = vec![bf16::ZERO; max_seq_len * rotary_dim];
    for pos in 0..max_seq_len {
        for i in 0..half_rotary {
            let freq = pos as f32 * inv_freq[i];
            let cos = bf16::from_f32(freq.cos());
            let sin = bf16::from_f32(freq.sin());
            cos_host[pos * rotary_dim + i] = cos;
            cos_host[pos * rotary_dim + i + half_rotary] = cos;
            sin_host[pos * rotary_dim + i] = sin;
            sin_host[pos * rotary_dim + i + half_rotary] = sin;
        }
    }

    let rms_norm_offset = |x: &[f32], weight: &[f32]| -> Vec<f32> {
        let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        x.iter()
            .zip(weight.iter())
            .map(|(v, w)| v * inv * (1.0 + w))
            .collect()
    };

    let apply_partial_rope = |x: &[f32], pos: usize| -> Vec<f32> {
        let mut out = x.to_vec();
        for i in 0..half_rotary {
            let cos = cos_host[pos * rotary_dim + i].to_f32();
            let sin = sin_host[pos * rotary_dim + i].to_f32();
            let lo = x[i];
            let hi = x[i + half_rotary];
            out[i] = lo * cos - hi * sin;
            out[i + half_rotary] = lo * sin + hi * cos;
        }
        out
    };

    for seq_len in [1_usize, 6_usize, 66_usize] {
        let current_pos = seq_len - 1;
        let q_full_host_bf16 = bf16_vec(
            &(0..q_full_dim)
                .map(|i| ((i % 73) as f32 - 36.0) * 0.03125)
                .collect::<Vec<_>>(),
        );
        let q_full_host: Vec<f32> = q_full_host_bf16.iter().map(|x| x.to_f32()).collect();
        let k_host_bf16 = bf16_vec(
            &(0..kv_dim)
                .map(|i| ((i % 61) as f32 - 30.0) * 0.03125)
                .collect::<Vec<_>>(),
        );
        let v_host_bf16 = bf16_vec(
            &(0..kv_dim)
                .map(|i| ((i % 67) as f32 - 33.0) * 0.03125)
                .collect::<Vec<_>>(),
        );
        let k_host: Vec<f32> = k_host_bf16.iter().map(|x| x.to_f32()).collect();
        let v_host: Vec<f32> = v_host_bf16.iter().map(|x| x.to_f32()).collect();

        let cache_len = num_kvheads * max_seq_len * head_dim;
        let mut k_cache_init_bf16 = vec![bf16::ZERO; cache_len];
        let mut v_cache_init_bf16 = vec![bf16::ZERO; cache_len];
        for kv_head in 0..num_kvheads {
            for pos in 0..current_pos {
                let base = (kv_head * max_seq_len + pos) * head_dim;
                for dim in 0..head_dim {
                    k_cache_init_bf16[base + dim] = bf16::from_f32(
                        ((kv_head * 31 + pos * 7 + dim) % 67) as f32 * 0.03125 - 1.0,
                    );
                    v_cache_init_bf16[base + dim] = bf16::from_f32(
                        ((kv_head * 19 + pos * 5 + dim) % 59) as f32 * 0.03125 - 0.8,
                    );
                }
            }
        }
        let mut ref_k_cache: Vec<f32> = k_cache_init_bf16.iter().map(|x| x.to_f32()).collect();
        let mut ref_v_cache: Vec<f32> = v_cache_init_bf16.iter().map(|x| x.to_f32()).collect();

        let q_full = DeviceVec::from_host(&ctx, &q_full_host_bf16)?;
        let k = DeviceVec::from_host(&ctx, &k_host_bf16)?;
        let v = DeviceVec::from_host(&ctx, &v_host_bf16)?;
        let q_weight = DeviceVec::from_host(&ctx, &q_weight_host_bf16)?;
        let k_weight = DeviceVec::from_host(&ctx, &k_weight_host_bf16)?;
        let cos_cache = DeviceVec::from_host(&ctx, &cos_host)?;
        let sin_cache = DeviceVec::from_host(&ctx, &sin_host)?;
        let decode_meta = ctx
            .stream
            .clone_htod(&[0_i32, current_pos as i32, seq_len as i32])?;
        let mut k_cache = DeviceVec::from_host(&ctx, &k_cache_init_bf16)?;
        let mut v_cache = DeviceVec::from_host(&ctx, &v_cache_init_bf16)?;
        let mut out = DeviceVec::zeros(&ctx, q_dim)?;

        fused_attention_hd256_decode_into(
            &ctx,
            &q_full,
            &k,
            &v,
            &q_weight,
            &k_weight,
            &cos_cache,
            &sin_cache,
            &decode_meta,
            &mut k_cache,
            &mut v_cache,
            &mut out,
            num_qheads,
            num_kvheads,
            rotary_dim,
            scale,
            eps,
        );

        let out_host = out.to_host(&ctx)?;
        let got_k_cache = k_cache.to_host(&ctx)?;
        let got_v_cache = v_cache.to_host(&ctx)?;

        for kv_head in 0..num_kvheads {
            let k_base = kv_head * head_dim;
            let k_head = &k_host[k_base..k_base + head_dim];
            let k_normed = rms_norm_offset(k_head, &k_weight_host);
            let k_rot = apply_partial_rope(&k_normed, current_pos);
            let cache_base = (kv_head * max_seq_len + current_pos) * head_dim;
            ref_k_cache[cache_base..cache_base + head_dim].copy_from_slice(&k_rot);
            ref_v_cache[cache_base..cache_base + head_dim]
                .copy_from_slice(&v_host[k_base..k_base + head_dim]);
        }

        let mut ref_out = vec![0.0_f32; q_dim];
        for q_head in 0..num_qheads {
            let q_base = q_head * 2 * head_dim;
            let q_head_slice = &q_full_host[q_base..q_base + head_dim];
            let gate_slice = &q_full_host[q_base + head_dim..q_base + 2 * head_dim];
            let q_normed = rms_norm_offset(q_head_slice, &q_weight_host);
            let q_rot = apply_partial_rope(&q_normed, current_pos);
            let kv_head = q_head / gqa_ratio;

            let mut scores = vec![0.0_f32; seq_len];
            for (pos, score) in scores.iter_mut().enumerate() {
                let k_base = (kv_head * max_seq_len + pos) * head_dim;
                *score = (0..head_dim)
                    .map(|dim| q_rot[dim] * ref_k_cache[k_base + dim])
                    .sum::<f32>()
                    * scale;
            }
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_scores: Vec<f32> = scores.iter().map(|x| (x - max_score).exp()).collect();
            let sum_exp = exp_scores.iter().sum::<f32>();
            let probs: Vec<f32> = exp_scores.iter().map(|x| x / sum_exp).collect();

            let gate_sig: Vec<f32> = gate_slice
                .iter()
                .map(|x| 1.0 / (1.0 + (-x).exp()))
                .collect();
            for dim in 0..head_dim {
                let mut acc = 0.0_f32;
                for (pos, prob) in probs.iter().enumerate() {
                    let v_base = (kv_head * max_seq_len + pos) * head_dim;
                    acc += prob * ref_v_cache[v_base + dim];
                }
                ref_out[q_head * head_dim + dim] = acc * gate_sig[dim];
            }
        }

        let current_base = current_pos * head_dim;
        let max_k_diff = (0..num_kvheads * head_dim)
            .map(|idx| {
                let kv_head = idx / head_dim;
                let dim = idx % head_dim;
                let offset = kv_head * max_seq_len * head_dim + current_base + dim;
                (got_k_cache[offset] - ref_k_cache[offset]).abs()
            })
            .fold(0.0_f32, f32::max);
        let max_v_diff = (0..num_kvheads * head_dim)
            .map(|idx| {
                let kv_head = idx / head_dim;
                let dim = idx % head_dim;
                let offset = kv_head * max_seq_len * head_dim + current_base + dim;
                (got_v_cache[offset] - ref_v_cache[offset]).abs()
            })
            .fold(0.0_f32, f32::max);
        let max_out_diff = out_host
            .iter()
            .zip(ref_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        assert!(
            max_k_diff < 0.02,
            "seq_len={seq_len} k_cache diff {max_k_diff}"
        );
        assert!(
            max_v_diff < 0.02,
            "seq_len={seq_len} v_cache diff {max_v_diff}"
        );
        assert!(
            max_out_diff < 0.1,
            "seq_len={seq_len} output diff {max_out_diff}"
        );
    }

    Ok(())
}

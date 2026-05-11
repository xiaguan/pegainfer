use super::*;

pub fn block_prefill_rank_local_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    weights: &RankWeightView<'_>,
    layer: usize,
    input: &HcHiddenStates,
    token_ids: &CudaSlice<u32>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
) -> Result<HcHiddenStates> {
    ctx.set_current()?;
    let block = weights.block(layer)?;

    let (attn_input, attn_hc) = hc_pre_bf16_hidden(
        ctx,
        config,
        input,
        &block.hc_attn_fn,
        &block.hc_attn_scale,
        &block.hc_attn_base,
    )?;
    let attn_norm = rms_norm_bf16_hidden(ctx, &attn_input, &block.attn_norm, config.rms_norm_eps)?;
    let attn_out = attention_prefill_rank_local_bf16_hidden(
        ctx,
        config,
        layer,
        &attn_norm,
        &block.attn,
        rope,
        start_pos,
    )?;
    let after_attn = hc_post_bf16_hidden(ctx, &attn_out, input, &attn_hc)?;

    let (ffn_input, ffn_hc) = hc_pre_bf16_hidden(
        ctx,
        config,
        &after_attn,
        &block.hc_ffn_fn,
        &block.hc_ffn_scale,
        &block.hc_ffn_base,
    )?;
    let ffn_norm = rms_norm_bf16_hidden(ctx, &ffn_input, &block.ffn_norm, config.rms_norm_eps)?;
    let ffn_out = moe_rank_local_bf16_hidden(ctx, config, weights, layer, &ffn_norm, token_ids)?;
    hc_post_bf16_hidden(ctx, &ffn_out, &after_attn, &ffn_hc)
}

pub(crate) fn block_prefill_rank_lane_bf16_hidden_with_decode_cache(
    ctx: &RankGpuContext,
    weights: &RankWeightView<'_>,
    comm: &Comm,
    config: &Config,
    layer: usize,
    input: &HcHiddenStates,
    token_ids: &CudaSlice<u32>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
    cache: &mut LayerDecodeCache,
) -> Result<HcHiddenStates> {
    ensure!(
        layer < config.n_layers,
        "rank-lane block prefill cache layer {layer} out of range"
    );
    let block = weights.block(layer)?;

    let (attn_input, attn_hc) = hc_pre_bf16_hidden(
        ctx,
        config,
        input,
        &block.hc_attn_fn,
        &block.hc_attn_scale,
        &block.hc_attn_base,
    )
    .with_context(|| format!("hc_pre attention layer {layer}"))?;
    let attn_norm = rms_norm_bf16_hidden(ctx, &attn_input, &block.attn_norm, config.rms_norm_eps)
        .with_context(|| format!("attention rms_norm layer {layer}"))?;
    let mut attn_out = match config.compress_ratios[layer] {
        0 => attention_prefill_rank_local_bf16_hidden_with_cache(
            ctx,
            config,
            layer,
            &attn_norm,
            &block.attn,
            rope,
            start_pos,
            &mut cache.kv,
        )?,
        4 => attention_prefill_compressed_overlap_rank_local_collective_bf16_hidden_with_cache(
            ctx,
            config,
            &attn_norm,
            &block.attn,
            rope,
            layer,
            start_pos,
            cache,
            comm,
        )?,
        _ => attention_prefill_compressed_nonoverlap_rank_local_bf16_hidden_with_cache(
            ctx,
            config,
            &attn_norm,
            &block.attn,
            rope,
            layer,
            start_pos,
            cache,
        )?,
    };
    all_reduce_hidden_fp32_in_place(ctx, &mut attn_out, comm)
        .with_context(|| format!("attention all_reduce layer {layer}"))?;
    let after_attn = hc_post_bf16_hidden(ctx, &attn_out, input, &attn_hc)
        .with_context(|| format!("hc_post attention layer {layer}"))?;

    let (ffn_input, ffn_hc) = hc_pre_bf16_hidden(
        ctx,
        config,
        &after_attn,
        &block.hc_ffn_fn,
        &block.hc_ffn_scale,
        &block.hc_ffn_base,
    )
    .with_context(|| format!("hc_pre ffn layer {layer}"))?;
    let ffn_norm = rms_norm_bf16_hidden(ctx, &ffn_input, &block.ffn_norm, config.rms_norm_eps)
        .with_context(|| format!("ffn rms_norm layer {layer}"))?;
    let ffn_out =
        moe_rank_lane_bf16_hidden(ctx, config, weights, comm, layer, &ffn_norm, token_ids)
            .with_context(|| format!("prefill MoE rank-lane layer {layer}"))?;
    hc_post_bf16_hidden(ctx, &ffn_out, &after_attn, &ffn_hc)
        .with_context(|| format!("hc_post ffn layer {layer}"))
}

pub fn block_decode_rank_lane_bf16_hidden(
    ctx: &RankGpuContext,
    weights: &RankWeightView<'_>,
    comm: &Comm,
    config: &Config,
    layer: usize,
    input: &HcHiddenStates,
    token_ids: &CudaSlice<u32>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
    cache: &mut LayerDecodeCache,
) -> Result<HcHiddenStates> {
    ensure!(
        input.seq_len == 1,
        "rank lane block decode expects HC seq_len=1, got {}",
        input.seq_len
    );
    ensure!(
        layer < config.n_layers,
        "rank lane block decode layer {layer} out of range"
    );
    let block = weights.block(layer)?;

    let (attn_norm, attn_hc) = hc_pre_norm_bf16_hidden(
        ctx,
        config,
        input,
        &block.hc_attn_fn,
        &block.hc_attn_scale,
        &block.hc_attn_base,
        &block.attn_norm,
    )
    .with_context(|| format!("hc_pre_norm attention layer {layer}"))?;
    let mut attn_out = attention_decode_rank_local_collective_bf16_hidden(
        ctx,
        config,
        layer,
        &attn_norm,
        &block.attn,
        rope,
        start_pos,
        cache,
        comm,
    )
    .with_context(|| format!("attention decode layer {layer}"))?;
    all_reduce_hidden_fp32_in_place(ctx, &mut attn_out, comm)
        .with_context(|| format!("attention all_reduce layer {layer}"))?;
    let after_attn = hc_post_bf16_hidden(ctx, &attn_out, input, &attn_hc)
        .with_context(|| format!("hc_post attention layer {layer}"))?;

    let (ffn_norm, ffn_hc) = hc_pre_norm_bf16_hidden(
        ctx,
        config,
        &after_attn,
        &block.hc_ffn_fn,
        &block.hc_ffn_scale,
        &block.hc_ffn_base,
        &block.ffn_norm,
    )
    .with_context(|| format!("hc_pre_norm ffn layer {layer}"))?;

    let ffn_out =
        decode_moe_ag_rs_bf16_hidden(ctx, config, weights, comm, layer, &ffn_norm, token_ids)
            .with_context(|| format!("decode MoE AG/RS layer {layer}"))?;
    hc_post_bf16_hidden(ctx, &ffn_out, &after_attn, &ffn_hc)
        .with_context(|| format!("hc_post ffn layer {layer}"))
}

fn attention_decode_rank_local_collective_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    layer: usize,
    input: &Bf16HiddenStates,
    attn: &AttentionWeights<'_>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
    cache: &mut LayerDecodeCache,
    comm: &Comm,
) -> Result<Bf16HiddenStates> {
    match config.compress_ratios[layer] {
        0 => attention_decode_rank_local_bf16_hidden(
            ctx,
            config,
            layer,
            input,
            attn,
            rope,
            start_pos,
            &mut cache.kv,
        ),
        4 => attention_decode_compressed_overlap_rank_local_collective_bf16_hidden(
            ctx, config, layer, input, attn, rope, start_pos, cache, comm,
        ),
        _ => attention_decode_compressed_nonoverlap_rank_local_bf16_hidden(
            ctx, config, layer, input, attn, rope, start_pos, cache,
        ),
    }
}

fn attention_decode_compressed_overlap_rank_local_collective_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    layer: usize,
    input: &Bf16HiddenStates,
    attn: &AttentionWeights<'_>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
    cache: &mut LayerDecodeCache,
    comm: &Comm,
) -> Result<Bf16HiddenStates> {
    ensure!(input.seq_len == 1, "ratio-4 decode expects seq_len=1");
    ensure!(
        config.compress_ratios[layer] == 4,
        "ratio-4 decode called for layer {layer} with ratio {}",
        config.compress_ratios[layer]
    );

    let compressor = attn
        .compressor
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("layer {layer} missing overlap compressor weights"))?;
    let indexer = attn
        .indexer
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("layer {layer} missing ratio-4 indexer weights"))?;
    let compressor_state = cache
        .compressor
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("layer {layer} missing overlap compressor state"))?;

    let mut projections = attention_project_bf16_hidden(ctx, config, input, attn)?;
    apply_rope_attention_projections(ctx, &mut projections, rope, start_pos)?;
    copy_bf16_rows_to_cache(
        ctx,
        &projections.kv,
        &mut cache.kv,
        0,
        start_pos % config.sliding_window,
        1,
    )?;
    if let Some(compressed_kv) = compressor_overlap_decode_bf16_hidden(
        ctx,
        config,
        input,
        compressor,
        rope,
        start_pos,
        compressor_state,
    )? {
        copy_bf16_rows_to_cache(
            ctx,
            &compressed_kv,
            &mut cache.kv,
            0,
            config.sliding_window + start_pos / 4,
            1,
        )?;
    }

    let indexer_kv = cache
        .indexer_kv
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("layer {layer} missing indexer kv cache"))?;
    let indexer_state = cache
        .indexer_compressor
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("layer {layer} missing indexer compressor state"))?;
    let mut scores = indexer_scores_decode_bf16_hidden(
        ctx,
        config,
        input,
        &projections.qr,
        indexer,
        rope,
        start_pos,
        indexer_kv,
        indexer_state,
    )?;

    if let Some(scores) = scores.as_mut() {
        comm.all_reduce_in_place(scores, &ReduceOp::Sum)
            .map_err(|err| {
                anyhow::anyhow!("NCCL decode indexer score all-reduce failed: {err:?}")
            })?;
    }

    let (window_idxs, window_topk) =
        window_topk_indices_decode(ctx, start_pos, config.sliding_window)?;
    let compressed_len = (start_pos + 1) / 4;
    let (topk_idxs, topk) = if compressed_len > 0 {
        let scores = scores
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("missing indexer decode scores"))?;
        let (compress_idxs, compress_topk) = indexer_topk_indices_decode(
            ctx,
            config,
            scores,
            compressed_len,
            config.sliding_window,
        )?;
        (
            concat_topk_indices(
                ctx,
                &window_idxs,
                window_topk,
                &compress_idxs,
                compress_topk,
                1,
            )?,
            window_topk + compress_topk,
        )
    } else {
        (window_idxs, window_topk)
    };
    let mut attn_out = indexed_attention_cache_bf16_hidden(
        ctx,
        config,
        &projections,
        &cache.kv,
        attn,
        &topk_idxs,
        topk,
    )?;
    attention_output_project_bf16_hidden(
        ctx,
        &mut attn_out,
        attn,
        rope,
        projections.local_heads,
        projections.head_dim,
        start_pos,
    )
}

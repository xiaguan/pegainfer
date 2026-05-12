use super::*;

pub fn hash_route_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    ffn: &FfnWeights<'_>,
    token_ids: &CudaSlice<u32>,
) -> Result<RoutedExperts> {
    ctx.set_current()?;
    ensure!(
        ffn.gate_weight.tensor.dtype == safetensors::Dtype::BF16,
        "gate weight {} must be BF16, got {:?}",
        ffn.gate_weight.name,
        ffn.gate_weight.tensor.dtype
    );
    ensure!(
        ffn.gate_weight.tensor.shape == [config.n_routed_experts, input.hidden_dim],
        "gate weight {} shape mismatch: expected {:?}, got {:?}",
        ffn.gate_weight.name,
        [config.n_routed_experts, input.hidden_dim],
        ffn.gate_weight.tensor.shape
    );
    let tid2eid = ffn
        .gate_tid2eid
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("hash routing requires gate.tid2eid"))?;
    ensure!(
        tid2eid.tensor.dtype == safetensors::Dtype::I64,
        "gate tid2eid {} must be I64, got {:?}",
        tid2eid.name,
        tid2eid.tensor.dtype
    );
    ensure!(
        tid2eid.tensor.shape == [config.vocab_size, config.n_activated_experts],
        "gate tid2eid {} shape mismatch: expected {:?}, got {:?}",
        tid2eid.name,
        [config.vocab_size, config.n_activated_experts],
        tid2eid.tensor.shape
    );

    let mut weights = ctx
        .stream
        .alloc_zeros(input.seq_len * config.n_activated_experts)?;
    let mut indices = ctx
        .stream
        .alloc_zeros(input.seq_len * config.n_activated_experts)?;
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (gate_ptr, _gate_guard) = ffn.gate_weight.tensor.data.device_ptr(&ctx.stream);
        let (tid2eid_ptr, _tid2eid_guard) = tid2eid.tensor.data.device_ptr(&ctx.stream);
        let (token_ptr, _token_guard) = token_ids.device_ptr(&ctx.stream);
        let (weights_ptr, _weights_guard) = weights.device_ptr_mut(&ctx.stream);
        let (indices_ptr, _indices_guard) = indices.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_hash_gate_cuda(
                x_ptr as *const ffi::Half,
                gate_ptr as *const ffi::Half,
                tid2eid_ptr as *const i64,
                token_ptr as *const u32,
                weights_ptr as *mut f32,
                indices_ptr as *mut i32,
                input.seq_len as i32,
                input.hidden_dim as i32,
                config.n_routed_experts as i32,
                config.n_activated_experts as i32,
                config.routed_scaling_factor,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    Ok(RoutedExperts {
        weights,
        indices,
        topk: config.n_activated_experts,
        seq_len: input.seq_len,
    })
}

pub fn score_route_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    ffn: &FfnWeights<'_>,
) -> Result<RoutedExperts> {
    ctx.set_current()?;
    ensure!(
        ffn.gate_weight.tensor.dtype == safetensors::Dtype::BF16,
        "gate weight {} must be BF16, got {:?}",
        ffn.gate_weight.name,
        ffn.gate_weight.tensor.dtype
    );
    ensure!(
        ffn.gate_weight.tensor.shape == [config.n_routed_experts, input.hidden_dim],
        "gate weight {} shape mismatch: expected {:?}, got {:?}",
        ffn.gate_weight.name,
        [config.n_routed_experts, input.hidden_dim],
        ffn.gate_weight.tensor.shape
    );
    let bias = ffn
        .gate_bias
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("score routing requires gate.bias"))?;
    ensure!(
        bias.tensor.dtype == safetensors::Dtype::F32,
        "gate bias {} must be F32, got {:?}",
        bias.name,
        bias.tensor.dtype
    );
    ensure!(
        bias.tensor.shape == [config.n_routed_experts],
        "gate bias {} shape mismatch: expected {:?}, got {:?}",
        bias.name,
        [config.n_routed_experts],
        bias.tensor.shape
    );

    let mut weights = ctx
        .stream
        .alloc_zeros(input.seq_len * config.n_activated_experts)?;
    let mut indices = ctx
        .stream
        .alloc_zeros(input.seq_len * config.n_activated_experts)?;
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (gate_ptr, _gate_guard) = ffn.gate_weight.tensor.data.device_ptr(&ctx.stream);
        let (bias_ptr, _bias_guard) = bias.tensor.data.device_ptr(&ctx.stream);
        let (weights_ptr, _weights_guard) = weights.device_ptr_mut(&ctx.stream);
        let (indices_ptr, _indices_guard) = indices.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_score_gate_cuda(
                x_ptr as *const ffi::Half,
                gate_ptr as *const ffi::Half,
                bias_ptr as *const f32,
                weights_ptr as *mut f32,
                indices_ptr as *mut i32,
                input.seq_len as i32,
                input.hidden_dim as i32,
                config.n_routed_experts as i32,
                config.n_activated_experts as i32,
                config.routed_scaling_factor,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    Ok(RoutedExperts {
        weights,
        indices,
        topk: config.n_activated_experts,
        seq_len: input.seq_len,
    })
}

pub fn build_moe_fused_route_plan(
    ctx: &RankGpuContext,
    config: &Config,
    weights: &RankWeightView<'_>,
    routed: RoutedExperts,
    input_hidden_dim: usize,
) -> Result<MoeFusedRoutePlan> {
    ctx.set_current()?;
    ensure!(
        input_hidden_dim == config.dim,
        "MoE fused route input dim mismatch: expected {}, got {}",
        config.dim,
        input_hidden_dim
    );
    ensure!(
        routed.topk == config.n_activated_experts,
        "MoE fused route topk mismatch: expected {}, got {}",
        config.n_activated_experts,
        routed.topk
    );
    ensure!(
        config.n_routed_experts.is_multiple_of(weights.world_size()),
        "n_routed_experts={} must be divisible by world_size={}",
        config.n_routed_experts,
        weights.world_size()
    );

    let local_experts = config.n_routed_experts / weights.world_size();
    let global_start = weights.rank() * local_experts;
    let num_expanded = routed.seq_len * routed.topk;
    let mut pos_to_token = ctx.stream.alloc_zeros(num_expanded)?;
    let mut pos_to_token_topk = ctx.stream.alloc_zeros(num_expanded)?;
    let mut token_topk_to_pos = ctx.stream.alloc_zeros(routed.seq_len * routed.topk)?;
    let mut expert_indptr = ctx.stream.alloc_zeros(local_experts + 1)?;
    let mut expert_cursor = ctx.stream.alloc_zeros(local_experts)?;
    let mut local_count = ctx.stream.alloc_zeros(1)?;
    {
        let (indices_ptr, _indices_guard) = routed.indices.device_ptr(&ctx.stream);
        let (pos_to_token_ptr, _pos_to_token_guard) = pos_to_token.device_ptr_mut(&ctx.stream);
        let (pos_to_token_topk_ptr, _pos_to_token_topk_guard) =
            pos_to_token_topk.device_ptr_mut(&ctx.stream);
        let (token_topk_to_pos_ptr, _token_topk_to_pos_guard) =
            token_topk_to_pos.device_ptr_mut(&ctx.stream);
        let (expert_indptr_ptr, _expert_indptr_guard) = expert_indptr.device_ptr_mut(&ctx.stream);
        let (expert_cursor_ptr, _expert_cursor_guard) = expert_cursor.device_ptr_mut(&ctx.stream);
        let (local_count_ptr, _local_count_guard) = local_count.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_moe_local_mapping_cuda(
                indices_ptr as *const i32,
                pos_to_token_ptr as *mut i32,
                pos_to_token_topk_ptr as *mut i32,
                token_topk_to_pos_ptr as *mut i32,
                expert_indptr_ptr as *mut i32,
                expert_cursor_ptr as *mut i32,
                local_count_ptr as *mut i32,
                routed.seq_len as i32,
                routed.topk as i32,
                global_start as i32,
                local_experts as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(MoeFusedRoutePlan {
        routed,
        pos_to_token,
        pos_to_token_topk,
        token_topk_to_pos,
        expert_indptr,
        expert_cursor,
        local_count,
        local_experts,
        global_start,
        num_expanded,
    })
}

pub fn add_f32_bf16_to_bf16_hidden(
    ctx: &RankGpuContext,
    a: &F32HiddenStates,
    b: &Bf16HiddenStates,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        a.hidden_dim == b.hidden_dim,
        "add f32/bf16 hidden dim mismatch: a={}, b={}",
        a.hidden_dim,
        b.hidden_dim
    );
    ensure!(
        a.seq_len == b.seq_len,
        "add f32/bf16 seq len mismatch: a={}, b={}",
        a.seq_len,
        b.seq_len
    );
    let mut out = Bf16HiddenStates::zeros(ctx, a.hidden_dim, a.seq_len)?;
    {
        let (a_ptr, _a_guard) = a.data.device_ptr(&ctx.stream);
        let (b_ptr, _b_guard) = b.data.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_add_f32_bf16_to_bf16_cuda(
                a_ptr as *const f32,
                b_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                (a.hidden_dim * a.seq_len) as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(out)
}

pub fn expand_moe_fused_input(
    ctx: &RankGpuContext,
    input: &Bf16HiddenStates,
    plan: &MoeFusedRoutePlan,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        plan.routed.seq_len == input.seq_len,
        "MoE expand route seq len mismatch: route={}, input={}",
        plan.routed.seq_len,
        input.seq_len
    );
    let mut out = Bf16HiddenStates::zeros(ctx, input.hidden_dim, plan.num_expanded)?;
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (pos_ptr, _pos_guard) = plan.pos_to_token.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_moe_expand_to_fused_cuda(
                x_ptr as *const ffi::Half,
                pos_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                input.hidden_dim as i32,
                plan.num_expanded as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(out)
}

pub fn reduce_moe_fused_output_f32(
    ctx: &RankGpuContext,
    expanded: &Bf16HiddenStates,
    plan: &MoeFusedRoutePlan,
    output_hidden_dim: usize,
) -> Result<F32HiddenStates> {
    ctx.set_current()?;
    ensure!(
        expanded.hidden_dim == output_hidden_dim,
        "MoE reduce hidden dim mismatch: expanded={}, output={}",
        expanded.hidden_dim,
        output_hidden_dim
    );
    let mut out = F32HiddenStates {
        data: ctx
            .stream
            .alloc_zeros(output_hidden_dim * plan.routed.seq_len)?,
        hidden_dim: output_hidden_dim,
        seq_len: plan.routed.seq_len,
    };
    {
        let (expanded_ptr, _expanded_guard) = expanded.data.device_ptr(&ctx.stream);
        let (weights_ptr, _weights_guard) = plan.routed.weights.device_ptr(&ctx.stream);
        let (map_ptr, _map_guard) = plan.token_topk_to_pos.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_moe_reduce_fused_f32_cuda(
                expanded_ptr as *const ffi::Half,
                weights_ptr as *const f32,
                map_ptr as *const i32,
                out_ptr as *mut f32,
                plan.routed.seq_len as i32,
                output_hidden_dim as i32,
                plan.routed.topk as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(out)
}

pub fn build_moe_expert_ptr_cache(
    ctx: &RankGpuContext,
    config: &Config,
    weights: &RankWeightView<'_>,
) -> Result<MoeGroupedPtrCache> {
    ctx.set_current()?;
    let local_experts = weights.local_experts();
    ensure!(
        local_experts > 0,
        "MoE grouped pointer cache needs local experts"
    );
    let mut layers = Vec::with_capacity(config.n_layers);
    for layer in 0..config.n_layers {
        layers.push(MoeLayerGroupedPtrs {
            w1: build_moe_grouped_linear_ptrs(
                ctx,
                local_experts,
                config.dim,
                config.moe_inter_dim,
                |local_expert| {
                    weights
                        .local_expert(layer, local_expert)
                        .map(|expert| expert.w1)
                },
            )
            .with_context(|| format!("build grouped ptr cache layer {layer} w1"))?,
            w2: build_moe_grouped_linear_ptrs(
                ctx,
                local_experts,
                config.moe_inter_dim,
                config.dim,
                |local_expert| {
                    weights
                        .local_expert(layer, local_expert)
                        .map(|expert| expert.w2)
                },
            )
            .with_context(|| format!("build grouped ptr cache layer {layer} w2"))?,
            w3: build_moe_grouped_linear_ptrs(
                ctx,
                local_experts,
                config.dim,
                config.moe_inter_dim,
                |local_expert| {
                    weights
                        .local_expert(layer, local_expert)
                        .map(|expert| expert.w3)
                },
            )
            .with_context(|| format!("build grouped ptr cache layer {layer} w3"))?,
        });
    }
    Ok(MoeGroupedPtrCache {
        layers,
        local_experts,
    })
}

fn build_moe_grouped_linear_ptrs<'a, F>(
    ctx: &RankGpuContext,
    local_experts: usize,
    in_dim: usize,
    out_dim: usize,
    mut linear_for_expert: F,
) -> Result<MoeGroupedLinearPtrs>
where
    F: FnMut(usize) -> Result<QuantLinearRef<'a>>,
{
    let mut weight_ptrs = Vec::with_capacity(local_experts);
    let mut scale_ptrs = Vec::with_capacity(local_experts);
    for local_expert in 0..local_experts {
        let linear = linear_for_expert(local_expert)?;
        validate_grouped_fp4_linear(&linear, in_dim, out_dim)
            .with_context(|| format!("validate grouped FP4 local expert {local_expert}"))?;
        let (weight_ptr, _weight_guard) = linear.weight.tensor.data.device_ptr(&ctx.stream);
        let (scale_ptr, _scale_guard) = linear.scale.tensor.data.device_ptr(&ctx.stream);
        weight_ptrs.push(weight_ptr as *const u8 as u64);
        scale_ptrs.push(scale_ptr as *const u8 as u64);
    }
    Ok(MoeGroupedLinearPtrs {
        weight_ptrs: ctx.stream.clone_htod(&weight_ptrs)?,
        scale_ptrs: ctx.stream.clone_htod(&scale_ptrs)?,
        in_dim,
        out_dim,
    })
}

fn validate_grouped_fp4_linear(
    linear: &QuantLinearRef<'_>,
    in_dim: usize,
    out_dim: usize,
) -> Result<()> {
    ensure!(
        linear.weight.tensor.dtype == safetensors::Dtype::F4,
        "grouped FP4 weight {} must be F4, got {:?}",
        linear.weight.name,
        linear.weight.tensor.dtype
    );
    ensure!(
        linear.scale.tensor.dtype == safetensors::Dtype::F8_E8M0,
        "grouped FP4 scale {} must be F8_E8M0, got {:?}",
        linear.scale.name,
        linear.scale.tensor.dtype
    );
    ensure!(
        linear.weight.tensor.shape == [out_dim, in_dim],
        "grouped FP4 weight {} shape mismatch: expected {:?}, got {:?}",
        linear.weight.name,
        [out_dim, in_dim],
        linear.weight.tensor.shape
    );
    ensure!(
        in_dim.is_multiple_of(32),
        "grouped FP4 input dim must be divisible by 32, got {}",
        in_dim
    );
    ensure!(
        linear.scale.tensor.shape == [out_dim, in_dim / 32],
        "grouped FP4 scale {} shape mismatch: expected {:?}, got {:?}",
        linear.scale.name,
        [out_dim, in_dim / 32],
        linear.scale.tensor.shape
    );
    Ok(())
}

pub fn local_experts_forward_packed_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    ptr_cache: &MoeGroupedPtrCache,
    layer: usize,
    expanded_input: &Bf16HiddenStates,
    plan: &MoeFusedRoutePlan,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        layer < ptr_cache.layers.len(),
        "MoE grouped pointer cache layer {layer} out of range {}",
        ptr_cache.layers.len()
    );
    ensure!(
        ptr_cache.local_experts == plan.local_experts,
        "MoE grouped pointer cache local_experts mismatch: cache={}, plan={}",
        ptr_cache.local_experts,
        plan.local_experts
    );
    let ptrs = &ptr_cache.layers[layer];
    let gate = fp4_grouped_linear_bf16_hidden(ctx, expanded_input, &plan.expert_indptr, &ptrs.w1)?;
    let up = fp4_grouped_linear_bf16_hidden(ctx, expanded_input, &plan.expert_indptr, &ptrs.w3)?;
    let activated = swiglu_clamp_bf16_hidden(ctx, &gate, &up, config.swiglu_limit)?;
    fp4_grouped_linear_bf16_hidden(ctx, &activated, &plan.expert_indptr, &ptrs.w2)
}

fn fp4_grouped_linear_bf16_hidden(
    ctx: &RankGpuContext,
    input: &Bf16HiddenStates,
    expert_indptr: &CudaSlice<i32>,
    ptrs: &MoeGroupedLinearPtrs,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    let local_experts = expert_indptr.len().saturating_sub(1);
    ensure!(local_experts > 0, "grouped FP4 needs local experts");
    ensure!(
        ptrs.weight_ptrs.len() == local_experts,
        "grouped FP4 weight pointer count mismatch: ptrs={}, local_experts={}",
        ptrs.weight_ptrs.len(),
        local_experts
    );
    ensure!(
        ptrs.scale_ptrs.len() == local_experts,
        "grouped FP4 scale pointer count mismatch: ptrs={}, local_experts={}",
        ptrs.scale_ptrs.len(),
        local_experts
    );
    ensure!(
        input.hidden_dim == ptrs.in_dim,
        "grouped FP4 input dim mismatch: expected {}, got {}",
        ptrs.in_dim,
        input.hidden_dim
    );
    let mut out = Bf16HiddenStates::zeros(ctx, ptrs.out_dim, input.seq_len)?;
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (weights_ptr, _weights_guard) = ptrs.weight_ptrs.device_ptr(&ctx.stream);
        let (scales_ptr, _scales_guard) = ptrs.scale_ptrs.device_ptr(&ctx.stream);
        let (expert_ptr, _expert_guard) = expert_indptr.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_moe_fp4_grouped_linear_cuda(
                x_ptr as *const ffi::Half,
                weights_ptr as *const *const u8,
                scales_ptr as *const *const u8,
                expert_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                input.seq_len as i32,
                input.hidden_dim as i32,
                ptrs.out_dim as i32,
                local_experts as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(out)
}

pub fn hash_routed_moe_rank_local_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    weights: &RankWeightView<'_>,
    ptr_cache: &MoeGroupedPtrCache,
    layer: usize,
    input: &Bf16HiddenStates,
    token_ids: &CudaSlice<u32>,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    let ffn = weights.ffn(layer)?;
    let routed = hash_route_bf16_hidden(ctx, config, input, &ffn, token_ids)?;
    let plan = build_moe_fused_route_plan(ctx, config, weights, routed, input.hidden_dim)?;
    let expanded_input = expand_moe_fused_input(ctx, input, &plan)?;
    let expanded_out = local_experts_forward_packed_bf16_hidden(
        ctx,
        config,
        ptr_cache,
        layer,
        &expanded_input,
        &plan,
    )?;
    let routed_out = reduce_moe_fused_output_f32(ctx, &expanded_out, &plan, input.hidden_dim)?;
    let shared = shared_expert_forward_bf16_hidden(ctx, input, &ffn, config.swiglu_limit)?;
    add_f32_bf16_to_bf16_hidden(ctx, &routed_out, &shared)
}

pub fn moe_rank_local_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    weights: &RankWeightView<'_>,
    ptr_cache: &MoeGroupedPtrCache,
    layer: usize,
    input: &Bf16HiddenStates,
    token_ids: &CudaSlice<u32>,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    let ffn = weights.ffn(layer)?;
    let routed = if layer < config.n_hash_layers {
        hash_route_bf16_hidden(ctx, config, input, &ffn, token_ids)?
    } else {
        score_route_bf16_hidden(ctx, config, input, &ffn)?
    };
    let plan = build_moe_fused_route_plan(ctx, config, weights, routed, input.hidden_dim)?;
    let expanded_input = expand_moe_fused_input(ctx, input, &plan)?;
    let expanded_out = local_experts_forward_packed_bf16_hidden(
        ctx,
        config,
        ptr_cache,
        layer,
        &expanded_input,
        &plan,
    )?;
    let routed_out = reduce_moe_fused_output_f32(ctx, &expanded_out, &plan, input.hidden_dim)?;
    let shared = shared_expert_forward_bf16_hidden(ctx, input, &ffn, config.swiglu_limit)?;
    add_f32_bf16_to_bf16_hidden(ctx, &routed_out, &shared)
}

pub(crate) fn moe_rank_lane_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    weights: &RankWeightView<'_>,
    ptr_cache: &MoeGroupedPtrCache,
    comm: &Comm,
    layer: usize,
    input: &Bf16HiddenStates,
    token_ids: &CudaSlice<u32>,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    let ffn = weights.ffn(layer)?;
    let routed = if layer < config.n_hash_layers {
        hash_route_bf16_hidden(ctx, config, input, &ffn, token_ids)?
    } else {
        score_route_bf16_hidden(ctx, config, input, &ffn)?
    };
    let plan = build_moe_fused_route_plan(ctx, config, weights, routed, input.hidden_dim)?;
    let expanded_input = expand_moe_fused_input(ctx, input, &plan)?;
    let expanded_out = local_experts_forward_packed_bf16_hidden(
        ctx,
        config,
        ptr_cache,
        layer,
        &expanded_input,
        &plan,
    )?;
    let mut routed_out = reduce_moe_fused_output_f32(ctx, &expanded_out, &plan, input.hidden_dim)?;
    all_reduce_f32_hidden_in_place(&mut routed_out, comm)
        .with_context(|| format!("MoE routed all-reduce layer {layer}"))?;
    let shared = shared_expert_forward_bf16_hidden(ctx, input, &ffn, config.swiglu_limit)?;
    add_f32_bf16_to_bf16_hidden(ctx, &routed_out, &shared)
}

pub(crate) fn decode_moe_ag_rs_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    weights: &RankWeightView<'_>,
    ptr_cache: &MoeGroupedPtrCache,
    comm: &Comm,
    layer: usize,
    input: &Bf16HiddenStates,
    token_ids: &CudaSlice<u32>,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        token_ids.len() == input.seq_len,
        "decode MoE AG/RS token count mismatch: tokens={}, hidden seq_len={}",
        token_ids.len(),
        input.seq_len
    );
    let world_size = weights.world_size();
    let global_hidden = all_gather_bf16_hidden(ctx, comm, input, world_size)?;
    let global_token_ids = if layer < config.n_hash_layers {
        Some(all_gather_u32(ctx, comm, token_ids, world_size)?)
    } else {
        None
    };

    let ffn = weights.ffn(layer)?;
    let routed = if layer < config.n_hash_layers {
        hash_route_bf16_hidden(
            ctx,
            config,
            &global_hidden,
            &ffn,
            global_token_ids
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing gathered token ids"))?,
        )?
    } else {
        score_route_bf16_hidden(ctx, config, &global_hidden, &ffn)?
    };
    let plan = build_moe_fused_route_plan(ctx, config, weights, routed, global_hidden.hidden_dim)?;
    let expanded_input = expand_moe_fused_input(ctx, &global_hidden, &plan)?;
    let expanded_out = local_experts_forward_packed_bf16_hidden(
        ctx,
        config,
        ptr_cache,
        layer,
        &expanded_input,
        &plan,
    )?;
    let partial_routed =
        reduce_moe_fused_output_f32(ctx, &expanded_out, &plan, global_hidden.hidden_dim)?;
    let local_routed = reduce_scatter_f32_hidden(ctx, comm, &partial_routed, world_size)?;
    let shared = shared_expert_forward_bf16_hidden(ctx, input, &ffn, config.swiglu_limit)?;
    add_f32_bf16_to_bf16_hidden(ctx, &local_routed, &shared)
}

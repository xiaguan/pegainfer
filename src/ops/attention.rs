use anyhow::Result;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;

use crate::ffi;
use crate::kv_pool::KvDesc;
use crate::tensor::{DeviceContext, DeviceVec, HiddenStates};

/// Batched prefill attention with FlashAttention-2.
///
/// Pipeline:
///   1. QK norm + RoPE (CUDA kernel, in-place on q_batch/k_batch)
///   2. KV cache write (CUDA kernel)
///   3. FlashAttention-2 (Triton kernel — fused QK + causal softmax + V)
///
/// No O(n²) scratch buffers needed — FlashAttention uses online softmax.
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_batch(
    ctx: &DeviceContext,
    q_batch: &mut HiddenStates,
    k_batch: &mut HiddenStates,
    v_batch: &HiddenStates,
    q_norm: &DeviceVec,
    k_norm: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    k_cache: &mut DeviceVec,
    v_cache: &mut DeviceVec,
    output: &mut HiddenStates,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    start_pos: usize,
    rms_eps: f32,
) -> Result<()> {
    let seq_len = q_batch.seq_len;
    let q_dim = num_q_heads * head_dim;
    assert!(num_kv_heads > 0, "num_kv_heads must be > 0");
    let gqa_ratio = num_q_heads / num_kv_heads;

    {
        let (q_ptr, _gq) = q_batch.data.device_ptr_mut(&ctx.stream);
        let (k_ptr, _gk) = k_batch.data.device_ptr_mut(&ctx.stream);
        let (v_ptr, _gv) = v_batch.data.device_ptr(&ctx.stream);
        let (qn_ptr, _gqn) = q_norm.data.device_ptr(&ctx.stream);
        let (kn_ptr, _gkn) = k_norm.data.device_ptr(&ctx.stream);
        let (cos_ptr, _gc) = cos_cache.data.device_ptr(&ctx.stream);
        let (sin_ptr, _gs) = sin_cache.data.device_ptr(&ctx.stream);
        let (kc_ptr, _gkc) = k_cache.data.device_ptr_mut(&ctx.stream);
        let (vc_ptr, _gvc) = v_cache.data.device_ptr_mut(&ctx.stream);
        let (o_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);

        unsafe {
            // Steps 1-2: QK norm + RoPE, KV cache write
            ffi::prefill_attention_prep_cuda(
                q_ptr as *mut ffi::Half,
                k_ptr as *mut ffi::Half,
                v_ptr as *const ffi::Half,
                qn_ptr as *const ffi::Half,
                kn_ptr as *const ffi::Half,
                cos_ptr as *const ffi::Half,
                sin_ptr as *const ffi::Half,
                kc_ptr as *mut ffi::Half,
                vc_ptr as *mut ffi::Half,
                num_q_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                seq_len as i32,
                start_pos as i32,
                rms_eps,
                ctx.stream.cu_stream(),
            );

            // Step 3: FlashAttention-2 (Triton) — reads normed Q and KV cache
            ffi::flash_attention_prefill_cuda(
                q_ptr as *const ffi::Half,
                kc_ptr as *const ffi::Half,
                vc_ptr as *const ffi::Half,
                o_ptr as *mut ffi::Half,
                num_q_heads as i32,
                num_kv_heads as i32,
                gqa_ratio as i32,
                seq_len as i32,
                start_pos as i32,
                q_dim as i32,
                ctx.stream.cu_stream(),
            );
        }
    }

    Ok(())
}

/// FlashAttention-2 prefill for HEAD_DIM=256 with precomputed Q and KV cache.
/// Q / output layout: HiddenStates [q_dim, seq_len] in column-major token-major storage.
#[allow(clippy::too_many_arguments)]
pub(crate) fn flash_attention_prefill_hd256_into(
    ctx: &DeviceContext,
    q_batch: &HiddenStates,
    k_cache: &DeviceVec,
    v_cache: &DeviceVec,
    output: &mut HiddenStates,
    num_q_heads: usize,
    num_kv_heads: usize,
    start_pos_buf: &CudaSlice<i32>,
) -> Result<()> {
    let seq_len = q_batch.seq_len;
    let q_dim = q_batch.hidden_dim;
    let head_dim = q_dim / num_q_heads;
    assert_eq!(head_dim, 256, "HD256 kernel requires head_dim=256");
    assert_eq!(q_dim, output.hidden_dim, "output hidden_dim mismatch");
    assert_eq!(seq_len, output.seq_len, "output seq_len mismatch");
    assert!(num_kv_heads > 0, "num_kv_heads must be > 0");
    let gqa_ratio = num_q_heads / num_kv_heads;

    let (q_ptr, _gq) = q_batch.data.device_ptr(&ctx.stream);
    let (kc_ptr, _gkc) = k_cache.data.device_ptr(&ctx.stream);
    let (vc_ptr, _gvc) = v_cache.data.device_ptr(&ctx.stream);
    let (o_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
    let (sp_ptr, _gsp) = start_pos_buf.device_ptr(&ctx.stream);

    let result = unsafe {
        ffi::flash_attention_prefill_hd256_cuda(
            q_ptr as *const ffi::Half,
            kc_ptr as *const ffi::Half,
            vc_ptr as *const ffi::Half,
            o_ptr as *mut ffi::Half,
            num_q_heads as i32,
            num_kv_heads as i32,
            gqa_ratio as i32,
            seq_len as i32,
            sp_ptr as *const i32,
            q_dim as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;

    Ok(())
}

/// Qwen3.5 full-attention prefill: prep Q/K/cache, run HD256 FlashAttention-2, then apply gate.
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_hd256_batch(
    ctx: &DeviceContext,
    q_full_batch: &HiddenStates,
    k_batch: &HiddenStates,
    v_batch: &HiddenStates,
    q_norm: &DeviceVec,
    k_norm: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    k_cache: &mut DeviceVec,
    v_cache: &mut DeviceVec,
    output: &mut HiddenStates,
    num_q_heads: usize,
    num_kv_heads: usize,
    start_pos: usize,
    rotary_dim: usize,
    rms_eps: f32,
) -> Result<()> {
    let q_dim = num_q_heads * 256;
    let mut q_prepped = HiddenStates::zeros(ctx, q_dim, q_full_batch.seq_len)?;
    // Allocate temporary GPU scalar for start_pos
    let start_pos_buf: CudaSlice<i32> = ctx
        .stream
        .clone_htod(&[start_pos as i32])
        .map_err(|e| anyhow::anyhow!("start_pos H2D failed: {e}"))?;
    prefill_attention_hd256_batch_with_scratch(
        ctx,
        q_full_batch,
        k_batch,
        v_batch,
        q_norm,
        k_norm,
        cos_cache,
        sin_cache,
        k_cache,
        v_cache,
        output,
        &mut q_prepped,
        num_q_heads,
        num_kv_heads,
        &start_pos_buf,
        rotary_dim,
        rms_eps,
    )
}

/// Same as `prefill_attention_hd256_batch` but uses pre-allocated scratch buffers.
/// `start_pos_buf` is a GPU-resident `i32` for CUDA Graph safety.
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_hd256_batch_with_scratch(
    ctx: &DeviceContext,
    q_full_batch: &HiddenStates,
    k_batch: &HiddenStates,
    v_batch: &HiddenStates,
    q_norm: &DeviceVec,
    k_norm: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    k_cache: &mut DeviceVec,
    v_cache: &mut DeviceVec,
    output: &mut HiddenStates,
    q_prepped: &mut HiddenStates,
    num_q_heads: usize,
    num_kv_heads: usize,
    start_pos_buf: &CudaSlice<i32>,
    rotary_dim: usize,
    rms_eps: f32,
) -> Result<()> {
    let seq_len = q_full_batch.seq_len;
    let q_dim = num_q_heads * 256;
    let kv_dim = num_kv_heads * 256;

    assert_eq!(q_full_batch.hidden_dim, q_dim * 2);
    assert_eq!(k_batch.hidden_dim, kv_dim);
    assert_eq!(v_batch.hidden_dim, kv_dim);
    assert_eq!(k_batch.seq_len, seq_len);
    assert_eq!(v_batch.seq_len, seq_len);
    assert_eq!(output.hidden_dim, q_dim);
    assert_eq!(output.seq_len, seq_len);
    assert_eq!(q_prepped.hidden_dim, q_dim);

    unsafe {
        let (qf_ptr, _gqf) = q_full_batch.data.device_ptr(&ctx.stream);
        let (k_ptr, _gk) = k_batch.data.device_ptr(&ctx.stream);
        let (v_ptr, _gv) = v_batch.data.device_ptr(&ctx.stream);
        let (qn_ptr, _gqn) = q_norm.data.device_ptr(&ctx.stream);
        let (kn_ptr, _gkn) = k_norm.data.device_ptr(&ctx.stream);
        let (cos_ptr, _gcos) = cos_cache.data.device_ptr(&ctx.stream);
        let (sin_ptr, _gsin) = sin_cache.data.device_ptr(&ctx.stream);
        let (qp_ptr, _gqp) = q_prepped.data.device_ptr_mut(&ctx.stream);
        let (kc_ptr, _gkc) = k_cache.data.device_ptr_mut(&ctx.stream);
        let (vc_ptr, _gvc) = v_cache.data.device_ptr_mut(&ctx.stream);
        let (sp_ptr, _gsp) = start_pos_buf.device_ptr(&ctx.stream);

        ffi::prefill_attention_hd256_prep_cuda(
            qf_ptr as *const ffi::Half,
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            qn_ptr as *const ffi::Half,
            kn_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            qp_ptr as *mut ffi::Half,
            kc_ptr as *mut ffi::Half,
            vc_ptr as *mut ffi::Half,
            num_q_heads as i32,
            num_kv_heads as i32,
            seq_len as i32,
            sp_ptr as *const i32,
            rotary_dim as i32,
            rms_eps,
            ctx.stream.cu_stream(),
        );
    }

    flash_attention_prefill_hd256_into(
        ctx,
        q_prepped,
        k_cache,
        v_cache,
        output,
        num_q_heads,
        num_kv_heads,
        start_pos_buf,
    )?;

    unsafe {
        let (qf_ptr, _gqf) = q_full_batch.data.device_ptr(&ctx.stream);
        let (o_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
        ffi::attention_gate_batch_hd256_cuda(
            qf_ptr as *const ffi::Half,
            o_ptr as *mut ffi::Half,
            num_q_heads as i32,
            seq_len as i32,
            ctx.stream.cu_stream(),
        );
    }

    Ok(())
}

// ============================================================================
// Paged attention decode (FlashInfer)
// ============================================================================

/// QK RMSNorm + RoPE for a single decode token (CUDA Graph safe).
///
/// Reads position from `decode_meta[1]` on device. Modifies q and k in-place.
#[allow(clippy::too_many_arguments)]
pub(crate) fn qk_norm_rope_into(
    ctx: &DeviceContext,
    q: &mut DeviceVec,
    k: &mut DeviceVec,
    q_norm_weight: &DeviceVec,
    k_norm_weight: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    decode_meta: &CudaSlice<i32>,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rms_eps: f32,
) {
    let (q_ptr, _gq) = q.data.device_ptr_mut(&ctx.stream);
    let (k_ptr, _gk) = k.data.device_ptr_mut(&ctx.stream);
    let (qn_ptr, _gqn) = q_norm_weight.data.device_ptr(&ctx.stream);
    let (kn_ptr, _gkn) = k_norm_weight.data.device_ptr(&ctx.stream);
    let (cos_ptr, _gc) = cos_cache.data.device_ptr(&ctx.stream);
    let (sin_ptr, _gs) = sin_cache.data.device_ptr(&ctx.stream);
    let (meta_ptr, _gm) = decode_meta.device_ptr(&ctx.stream);

    unsafe {
        ffi::qk_norm_rope_cuda(
            q_ptr as *mut ffi::Half,
            k_ptr as *mut ffi::Half,
            qn_ptr as *const ffi::Half,
            kn_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            num_q_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            meta_ptr as *const i32,
            rms_eps,
            ctx.stream.cu_stream(),
        );
    }
}

/// Append one K/V token to paged cache, then run FlashInfer paged attention decode.
///
/// All GPU metadata (`page_indices_d` through `kv_chunk_size_d`) must be
/// pre-allocated and updated via `memcpy_htod` before this call.
/// This makes the function CUDA Graph safe — no GPU allocations inside.
#[allow(clippy::too_many_arguments)]
pub(crate) fn paged_attention_decode_into(
    ctx: &DeviceContext,
    q: &DeviceVec,
    k: &DeviceVec,
    v: &DeviceVec,
    kv_buffer: &CudaSlice<bf16>,
    layout: &crate::kv_pool::KvLayout,
    layer: usize,
    page_indices_d: &CudaSlice<i32>,
    page_indptr_d: &CudaSlice<i32>,
    last_page_len_d: &CudaSlice<i32>,
    request_indices_d: &CudaSlice<i32>,
    kv_tile_indices_d: &CudaSlice<i32>,
    kv_chunk_size_d: &CudaSlice<i32>,
    output: &mut DeviceVec,
    num_qo_heads: usize,
) -> Result<()> {
    let num_kv_heads = layout.num_kv_heads;
    let head_dim = layout.head_dim;
    let page_size = layout.page_size;

    // K/V offsets for this layer within the page-first buffer
    let k_offset = (layer * layout.layer_stride) as i64;
    let v_offset = (layer * layout.layer_stride + layout.kv_block_len) as i64;
    let stride_page = layout.page_stride as i64;

    let (buf_ptr, _gbuf) = kv_buffer.device_ptr(&ctx.stream);
    let (q_ptr, _gq) = q.data.device_ptr(&ctx.stream);
    let (k_ptr, _gk) = k.data.device_ptr(&ctx.stream);
    let (v_ptr, _gv) = v.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
    let (pi_ptr, _gpi) = page_indices_d.device_ptr(&ctx.stream);
    let (pip_ptr, _gpip) = page_indptr_d.device_ptr(&ctx.stream);
    let (lpl_ptr, _glpl) = last_page_len_d.device_ptr(&ctx.stream);
    let (ri_ptr, _gri) = request_indices_d.device_ptr(&ctx.stream);
    let (kti_ptr, _gkti) = kv_tile_indices_d.device_ptr(&ctx.stream);
    let (kcs_ptr, _gkcs) = kv_chunk_size_d.device_ptr(&ctx.stream);

    let stream = ctx.stream.cu_stream();

    // Step 1: Append K/V to paged cache
    let result = unsafe {
        ffi::paged_kv_append_cuda(
            buf_ptr as *const ffi::Half,
            k_offset,
            v_offset,
            pi_ptr as *const i32,
            pip_ptr as *const i32,
            lpl_ptr as *const i32,
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            num_kv_heads as i32,
            head_dim as i32,
            page_size as i32,
            /*batch_size=*/ 1,
            stride_page,
            stream,
        )
    };
    if result != 0 {
        anyhow::bail!("paged_kv_append_cuda failed with error {result}");
    }

    // Step 2: Paged attention decode
    let sm_scale = 1.0f32 / (head_dim as f32).sqrt();
    let result = unsafe {
        ffi::paged_attention_decode_cuda(
            q_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            buf_ptr as *const ffi::Half,
            k_offset,
            v_offset,
            pi_ptr as *const i32,
            pip_ptr as *const i32,
            lpl_ptr as *const i32,
            ri_ptr as *const i32,
            kti_ptr as *const i32,
            kcs_ptr as *const i32,
            num_qo_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            page_size as i32,
            /*batch_size=*/ 1,
            stride_page,
            sm_scale,
            stream,
        )
    };
    if result != 0 {
        anyhow::bail!("paged_attention_decode_cuda failed with error {result}");
    }

    Ok(())
}

/// Scatter contiguous KV cache into paged layout after prefill.
///
/// Iterates over all layers, copying each layer's K/V from contiguous HND
/// buffers into the page-first paged layout.
pub(crate) fn scatter_kv_to_paged(
    ctx: &DeviceContext,
    kv_cache: &crate::model::kv_cache::KVCache,
    desc: &KvDesc<'_>,
) -> Result<()> {
    let layout = desc.layout();
    let seq_len = desc.seq_len();
    if seq_len == 0 {
        return Ok(());
    }

    let head_dim = layout.head_dim;
    let num_kv_heads = layout.num_kv_heads;
    let page_size = layout.page_size;
    let max_seq_len = kv_cache.max_seq_len();

    // Source strides (HND per-layer): k[head, pos, dim]
    let src_stride_n = head_dim as i64;
    let src_stride_h = (max_seq_len * head_dim) as i64;

    // GPU metadata (same for all layers)
    let page_indices_d: CudaSlice<i32> = ctx.stream.clone_htod(
        &desc
            .page_indices()
            .iter()
            .map(|p| p.index() as i32)
            .collect::<Vec<_>>(),
    )?;
    let num_pages = desc.num_pages() as i32;
    let page_indptr_d: CudaSlice<i32> = ctx.stream.clone_htod(&[0i32, num_pages])?;
    let last_page_len_d: CudaSlice<i32> = ctx.stream.clone_htod(&[desc.last_page_len() as i32])?;

    // batch_indices = [0, 0, ..., 0], positions = [0, 1, 2, ..., seq_len-1]
    let batch_indices: Vec<i32> = vec![0i32; seq_len];
    let positions: Vec<i32> = (0..seq_len as i32).collect();
    let batch_indices_d: CudaSlice<i32> = ctx.stream.clone_htod(&batch_indices)?;
    let positions_d: CudaSlice<i32> = ctx.stream.clone_htod(&positions)?;

    let (buf_ptr, _gbuf) = desc.buffer().device_ptr(&ctx.stream);
    let (pi_ptr, _) = page_indices_d.device_ptr(&ctx.stream);
    let (pip_ptr, _) = page_indptr_d.device_ptr(&ctx.stream);
    let (lpl_ptr, _) = last_page_len_d.device_ptr(&ctx.stream);
    let (bi_ptr, _) = batch_indices_d.device_ptr(&ctx.stream);
    let (pos_ptr, _) = positions_d.device_ptr(&ctx.stream);
    let stream = ctx.stream.cu_stream();

    for layer in 0..layout.num_layers {
        let k_offset = (layer * layout.layer_stride) as i64;
        let v_offset = (layer * layout.layer_stride + layout.kv_block_len) as i64;
        let stride_page = layout.page_stride as i64;

        let (k_cache, v_cache) = kv_cache.get_cache(layer);
        let (sk_ptr, _gsk) = k_cache.data.device_ptr(&ctx.stream);
        let (sv_ptr, _gsv) = v_cache.data.device_ptr(&ctx.stream);

        let result = unsafe {
            ffi::paged_kv_scatter_cuda(
                buf_ptr as *const ffi::Half,
                k_offset,
                v_offset,
                pi_ptr as *const i32,
                pip_ptr as *const i32,
                lpl_ptr as *const i32,
                sk_ptr as *const ffi::Half,
                sv_ptr as *const ffi::Half,
                bi_ptr as *const i32,
                pos_ptr as *const i32,
                seq_len as i32,
                num_kv_heads as i32,
                head_dim as i32,
                page_size as i32,
                stride_page,
                src_stride_n,
                src_stride_h,
                stream,
            )
        };
        if result != 0 {
            anyhow::bail!("paged_kv_scatter_cuda failed for layer {layer} with error {result}");
        }
    }

    Ok(())
}

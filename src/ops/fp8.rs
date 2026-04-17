//! FP8 operations: block-scale quantization and DeepGEMM GEMM.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::ffi;
use crate::model::dsv32::Fp8Matrix;
use crate::tensor::{DeviceContext, HiddenStates};

/// Scratch buffers for FP8 linear operations.
///
/// Pre-allocated to avoid per-call allocation. Sized for the maximum
/// activation dimensions encountered in a model.
pub(crate) struct Fp8Scratch {
    /// FP8 e4m3 activation buffer: [max_m * max_k] bytes
    pub(crate) fp8_act: CudaSlice<u8>,
    /// Dequant scales: [ceil(max_k/128) * padded(max_m, 4)] f32
    pub(crate) scale_a: CudaSlice<f32>,
    max_m: usize,
    max_k: usize,
    last_quantized_m: usize,
    last_quantized_k: usize,
    has_quantized_shape: bool,
}

impl Fp8Scratch {
    /// Allocate scratch buffers for FP8 quantize + GEMM.
    ///
    /// `max_m` = max batch size (seq_len), `max_k` = max input dimension.
    pub(crate) fn new(ctx: &DeviceContext, max_m: usize, max_k: usize) -> Self {
        let fp8_act = ctx
            .stream
            .alloc_zeros(max_m * max_k)
            .expect("Fp8Scratch: alloc fp8_act failed");
        let scale_k_chunks = max_k.div_ceil(128);
        let padded_m = max_m.next_multiple_of(4);
        let scale_a = ctx
            .stream
            .alloc_zeros(scale_k_chunks * padded_m)
            .expect("Fp8Scratch: alloc scale_a failed");
        Self {
            fp8_act,
            scale_a,
            max_m,
            max_k,
            last_quantized_m: 0,
            last_quantized_k: 0,
            has_quantized_shape: false,
        }
    }
}

/// FP8 quantize + GEMM: output = weight @ input (with online FP8 quantization of input).
///
/// `input`: HiddenStates [K, bs] (K = hidden_dim, bs = seq_len)
/// `weight`: Fp8Matrix [N, K] (output_dim × input_dim)
/// `output`: HiddenStates [N, bs]
/// `scratch`: pre-allocated FP8 scratch buffers
pub(crate) fn fp8_linear_into(
    ctx: &DeviceContext,
    input: &HiddenStates,
    weight: &Fp8Matrix,
    scratch: &mut Fp8Scratch,
    output: &mut HiddenStates,
) {
    let m = input.seq_len; // batch size / number of tokens
    let k = input.hidden_dim; // input dimension
    let n = weight.rows; // output dimension

    assert_eq!(
        weight.cols, k,
        "FP8 weight cols {} != input dim {}",
        weight.cols, k
    );
    assert_eq!(
        output.hidden_dim, n,
        "output dim {} != weight rows {}",
        output.hidden_dim, n
    );
    assert_eq!(
        output.seq_len, m,
        "output seq_len {} != input seq_len {}",
        output.seq_len, m
    );
    assert!(
        m <= scratch.max_m,
        "m={} exceeds scratch max_m={}",
        m,
        scratch.max_m
    );
    assert!(
        k <= scratch.max_k,
        "k={} exceeds scratch max_k={}",
        k,
        scratch.max_k
    );

    // HiddenStates memory layout: token i at offset i * hidden_dim.
    // This is [hidden_dim, seq_len] column-major for cuBLAS, but
    // FP8 GEMM expects [M, K] row-major where M=seq_len, K=hidden_dim.
    // Fortunately, [hidden_dim * seq_len] contiguous with stride hidden_dim
    // between tokens IS [M=seq_len, K=hidden_dim] row-major. ✓

    let (input_ptr, _gi) = input.data.device_ptr(&ctx.stream);
    let (fp8_ptr, _gf) = scratch.fp8_act.device_ptr_mut(&ctx.stream);
    let (scale_a_ptr, _gs) = scratch.scale_a.device_ptr_mut(&ctx.stream);
    let (weight_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (scale_b_ptr, _gsb) = weight.scale_inv.device_ptr(&ctx.stream);
    let (out_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);

    unsafe {
        // Step 1: FP8 quantize activation
        ffi::fp8_quantize_1x128_cuda(
            input_ptr as *const ffi::Half,
            fp8_ptr as *mut u8,
            scale_a_ptr as *mut f32,
            m as i32,
            k as i32,
            ctx.stream.cu_stream(),
        );

        scratch.last_quantized_m = m;
        scratch.last_quantized_k = k;
        scratch.has_quantized_shape = true;

        // Step 2: FP8 GEMM — D[M,N] = A[M,K] @ B[N,K]^T
        ffi::fp8_gemm_cuda(
            fp8_ptr as *const u8,
            scale_a_ptr as *const f32,
            weight_ptr as *const u8,
            scale_b_ptr as *const f32,
            out_ptr as *mut ffi::Half,
            m as i32,
            n as i32,
            k as i32,
            ctx.stream.cu_stream(),
        );
    }
}

/// FP8 quantize activation only (for shared quantization when multiple projections
/// use the same input, e.g. q_a_proj and kv_a_proj both read from normed hidden).
///
/// After this call, `scratch.fp8_act` and `scratch.scale_a` are populated for `m × k`.
pub(crate) fn fp8_quantize_into(
    ctx: &DeviceContext,
    input: &HiddenStates,
    scratch: &mut Fp8Scratch,
) {
    let m = input.seq_len;
    let k = input.hidden_dim;
    assert!(m <= scratch.max_m && k <= scratch.max_k);

    let (input_ptr, _gi) = input.data.device_ptr(&ctx.stream);
    let (fp8_ptr, _gf) = scratch.fp8_act.device_ptr_mut(&ctx.stream);
    let (scale_a_ptr, _gs) = scratch.scale_a.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::fp8_quantize_1x128_cuda(
            input_ptr as *const ffi::Half,
            fp8_ptr as *mut u8,
            scale_a_ptr as *mut f32,
            m as i32,
            k as i32,
            ctx.stream.cu_stream(),
        );
    }

    scratch.last_quantized_m = m;
    scratch.last_quantized_k = k;
    scratch.has_quantized_shape = true;
}

/// FP8 GEMM only (uses pre-quantized activation from scratch).
///
/// Assumes `scratch.fp8_act` and `scratch.scale_a` were populated by a prior
/// `fp8_quantize_into` call with matching `m` and `k`.
pub(crate) fn fp8_gemm_into(
    ctx: &DeviceContext,
    m: usize,
    k: usize,
    weight: &Fp8Matrix,
    scratch: &Fp8Scratch,
    output: &mut HiddenStates,
) {
    let n = weight.rows;
    assert_eq!(weight.cols, k);
    assert_eq!(output.hidden_dim, n);
    assert_eq!(output.seq_len, m);
    assert!(
        scratch.has_quantized_shape,
        "fp8_gemm_into called before fp8_quantize_into populated scratch"
    );
    assert_eq!(
        scratch.last_quantized_m, m,
        "fp8_gemm_into shape mismatch: scratch m={} but requested m={}",
        scratch.last_quantized_m, m
    );
    assert_eq!(
        scratch.last_quantized_k, k,
        "fp8_gemm_into shape mismatch: scratch k={} but requested k={}",
        scratch.last_quantized_k, k
    );

    let (fp8_ptr, _gf) = scratch.fp8_act.device_ptr(&ctx.stream);
    let (scale_a_ptr, _gs) = scratch.scale_a.device_ptr(&ctx.stream);
    let (weight_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (scale_b_ptr, _gsb) = weight.scale_inv.device_ptr(&ctx.stream);
    let (out_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::fp8_gemm_cuda(
            fp8_ptr as *const u8,
            scale_a_ptr as *const f32,
            weight_ptr as *const u8,
            scale_b_ptr as *const f32,
            out_ptr as *mut ffi::Half,
            m as i32,
            n as i32,
            k as i32,
            ctx.stream.cu_stream(),
        );
    }
}

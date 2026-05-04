use std::ffi::c_void;
use std::mem::size_of;
use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use cudarc::driver::{CudaEvent, CudaSlice, DevicePtr, DevicePtrMut, sys};
use half::bf16;
use pegainfer_kernels::ffi;
use pegainfer_kernels::paged_kv::PagedKvLayout;
use pegainfer_kernels::tensor::{DeviceContext, HiddenStates};
use serde::{Deserialize, Serialize};

pub const NUM_LAYERS: usize = 1;
pub const NUM_QO_HEADS: usize = 32;
pub const NUM_KV_HEADS: usize = 8;
pub const HEAD_DIM: usize = 128;
pub const PAGE_SIZE: usize = 16;
pub const CONTEXT_LENGTHS: &[usize] = &[128, 512, 1024, 2048, 4096, 8192, 10_000];
pub const BATCH_SIZES: &[usize] = &[1, 2, 4, 8, 16, 32];
pub const BATCH_SWEEP_CONTEXT_LEN: usize = 1024;
pub const INNER_LAUNCHES: u64 = 16;
pub const REPORT_ITERS: u64 = 128;
pub const DEFAULT_SPLIT_KV_CHUNK_TOKENS: usize = 256;
pub const DEFAULT_SPLIT_KV_MAX_CHUNKS_PER_REQUEST: usize = 64;
pub const MEMORY_TRANSFERS_PER_CLOCK: f64 = 2.0;
pub const CACHE_CLEAR_L2_MULTIPLIER: usize = 2;
pub const CACHE_CLEAR_MIN_BYTES: usize = 128 * 1024 * 1024;
pub const SPLITK_TUNING_CONTEXT_LENGTHS: &[usize] = &[1024, 2048, 4096, 8192, 10_000];
pub const SPLITK_TUNING_BATCH_SIZES: &[usize] = &[1, 2];
pub const SPLITK_TUNING_CONFIGS: &[SplitKvConfig] = &[
    SplitKvConfig::new(256, 128),
    SplitKvConfig::new(256, 64),
    SplitKvConfig::new(384, 96),
    SplitKvConfig::new(512, 128),
    SplitKvConfig::new(512, 64),
    SplitKvConfig::new(512, 32),
    SplitKvConfig::new(768, 64),
    SplitKvConfig::new(1024, 64),
    SplitKvConfig::new(1024, 32),
];

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SplitKvConfig {
    pub chunk_tokens: usize,
    pub max_chunks_per_request: usize,
}

impl SplitKvConfig {
    pub const fn new(chunk_tokens: usize, max_chunks_per_request: usize) -> Self {
        Self {
            chunk_tokens,
            max_chunks_per_request,
        }
    }

    pub fn actual_chunk_size(self, kv_len: usize) -> usize {
        self.chunk_tokens
            .max(kv_len.div_ceil(self.max_chunks_per_request))
    }

    pub fn active_chunks(self, kv_len: usize) -> usize {
        kv_len.div_ceil(self.actual_chunk_size(kv_len)).max(1)
    }

    pub fn label(self) -> String {
        format!(
            "split_kv_{}x{}",
            self.chunk_tokens, self.max_chunks_per_request
        )
    }
}

pub const DEFAULT_SPLIT_KV_CONFIG: SplitKvConfig = SplitKvConfig::new(
    DEFAULT_SPLIT_KV_CHUNK_TOKENS,
    DEFAULT_SPLIT_KV_MAX_CHUNKS_PER_REQUEST,
);

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum AttentionKernelVariant {
    NonPartition,
    SplitKv(SplitKvConfig),
}

impl AttentionKernelVariant {
    pub fn label(self) -> String {
        match self {
            Self::NonPartition => "non_partition".to_string(),
            Self::SplitKv(config) => config.label(),
        }
    }

    pub fn decode_path(self) -> DecodePath {
        match self {
            Self::NonPartition => DecodePath::NonPartition,
            Self::SplitKv(_) => DecodePath::SplitK,
        }
    }

    pub fn split_config(self) -> SplitKvConfig {
        match self {
            Self::NonPartition => DEFAULT_SPLIT_KV_CONFIG,
            Self::SplitKv(config) => config,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum DecodePath {
    NonPartition,
    SplitK,
}

impl DecodePath {
    pub fn name(self, split_config: SplitKvConfig) -> String {
        match self {
            Self::NonPartition => "non_partition".to_string(),
            Self::SplitK => split_config.label(),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct AttentionKernelShape {
    pub batch_size: usize,
    pub kv_len: usize,
}

impl AttentionKernelShape {
    pub const fn new(batch_size: usize, kv_len: usize) -> Self {
        Self { batch_size, kv_len }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AttentionKernelSpec {
    pub shape: AttentionKernelShape,
    pub variant: AttentionKernelVariant,
}

pub fn default_attention_kernel_specs() -> Vec<AttentionKernelSpec> {
    let variants = [
        AttentionKernelVariant::NonPartition,
        AttentionKernelVariant::SplitKv(DEFAULT_SPLIT_KV_CONFIG),
        AttentionKernelVariant::SplitKv(SplitKvConfig::new(512, 64)),
    ];
    BATCH_SIZES
        .iter()
        .copied()
        .flat_map(|batch_size| {
            CONTEXT_LENGTHS.iter().copied().flat_map(move |kv_len| {
                variants
                    .into_iter()
                    .map(move |variant| AttentionKernelSpec {
                        shape: AttentionKernelShape::new(batch_size, kv_len),
                        variant,
                    })
            })
        })
        .collect()
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct DevicePeakBandwidth {
    pub memory_clock_khz: i32,
    pub memory_bus_width_bits: i32,
    pub peak_bytes_per_sec: f64,
}

impl DevicePeakBandwidth {
    pub fn query(ctx: &DeviceContext) -> Result<Self> {
        let memory_clock_khz = ctx
            .ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)
            .map_err(|e| anyhow!("failed to query memory clock: {e}"))?;
        let memory_bus_width_bits = ctx
            .ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)
            .map_err(|e| anyhow!("failed to query memory bus width: {e}"))?;
        let peak_bytes_per_sec = f64::from(memory_clock_khz)
            * 1_000.0
            * (f64::from(memory_bus_width_bits) / 8.0)
            * MEMORY_TRANSFERS_PER_CLOCK;

        Ok(Self {
            memory_clock_khz,
            memory_bus_width_bits,
            peak_bytes_per_sec,
        })
    }

    pub fn peak_gb_per_sec(&self) -> f64 {
        self.peak_bytes_per_sec / 1.0e9
    }
}

pub struct L2CacheClear {
    a: CudaSlice<bf16>,
    b: CudaSlice<bf16>,
    out: CudaSlice<bf16>,
    len: usize,
}

impl L2CacheClear {
    pub fn new(ctx: &DeviceContext) -> Result<Self> {
        let l2_bytes =
            ctx.ctx
                .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
                .map_err(|e| anyhow!("failed to query L2 cache size: {e}"))? as usize;
        let clear_bytes = cache_clear_bytes(l2_bytes);
        let len = clear_bytes.div_ceil(size_of::<bf16>());

        Ok(Self {
            a: ctx.stream.alloc_zeros(len)?,
            b: ctx.stream.alloc_zeros(len)?,
            out: ctx.stream.alloc_zeros(len)?,
            len,
        })
    }

    pub fn clear(&mut self, ctx: &DeviceContext) -> Result<()> {
        let (a_ptr, _a_guard) = self.a.device_ptr(&ctx.stream);
        let (b_ptr, _b_guard) = self.b.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = self.out.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::add_cuda(
                a_ptr as *const ffi::Half,
                b_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                self.len as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
        Ok(())
    }
}

pub fn cache_clear_bytes(l2_bytes: usize) -> usize {
    (l2_bytes * CACHE_CLEAR_L2_MULTIPLIER).max(CACHE_CLEAR_MIN_BYTES)
}

pub struct AttentionDecodeCase {
    pub ctx: DeviceContext,
    layout: PagedKvLayout,
    q: HiddenStates,
    output: HiddenStates,
    kv_buffer: CudaSlice<bf16>,
    page_indices_d: CudaSlice<i32>,
    page_indptr_d: CudaSlice<i32>,
    last_page_len_d: CudaSlice<i32>,
    request_indices_d: CudaSlice<i32>,
    kv_tile_indices_d: CudaSlice<i32>,
    kv_chunk_size_d: CudaSlice<i32>,
    split_request_indices_d: CudaSlice<i32>,
    split_kv_tile_indices_d: CudaSlice<i32>,
    split_kv_chunk_size_d: CudaSlice<i32>,
    split_o_indptr_d: CudaSlice<i32>,
    split_block_valid_mask_d: CudaSlice<u8>,
    split_tmp_v: CudaSlice<bf16>,
    split_tmp_s: CudaSlice<f32>,
    split_padded_slots: usize,
    split_config: SplitKvConfig,
    start: CudaEvent,
    end: CudaEvent,
    batch_size: usize,
    kv_len: usize,
}

impl AttentionDecodeCase {
    pub fn new(batch_size: usize, kv_len: usize) -> Result<Self> {
        Self::new_with_split_config(batch_size, kv_len, DEFAULT_SPLIT_KV_CONFIG)
    }

    pub fn for_spec(spec: AttentionKernelSpec) -> Result<Self> {
        Self::new_with_split_config(
            spec.shape.batch_size,
            spec.shape.kv_len,
            spec.variant.split_config(),
        )
    }

    pub fn new_with_split_config(
        batch_size: usize,
        kv_len: usize,
        split_config: SplitKvConfig,
    ) -> Result<Self> {
        let ctx = DeviceContext::new()?;
        let layout = PagedKvLayout::new(NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE);
        let q_dim = NUM_QO_HEADS * HEAD_DIM;
        let pages_per_request = kv_len.div_ceil(PAGE_SIZE);
        let total_pages = pages_per_request * batch_size;

        let q_host = patterned_bf16(q_dim * batch_size, 0.01);
        let kv_host = patterned_bf16(total_pages * layout.page_stride, 0.001);

        let q = HiddenStates {
            data: ctx.stream.clone_htod(&q_host)?,
            hidden_dim: q_dim,
            seq_len: batch_size,
        };
        let output = HiddenStates::zeros(&ctx, q_dim, batch_size)?;
        let kv_buffer = ctx.stream.clone_htod(&kv_host)?;

        let mut page_indices = Vec::with_capacity(total_pages);
        let mut page_indptr = Vec::with_capacity(batch_size + 1);
        page_indptr.push(0);
        for request_idx in 0..batch_size {
            for page_offset in 0..pages_per_request {
                page_indices.push((request_idx * pages_per_request + page_offset) as i32);
            }
            page_indptr.push(page_indices.len() as i32);
        }

        let last_page_len = match kv_len % PAGE_SIZE {
            0 => PAGE_SIZE,
            rem => rem,
        };
        let last_page_lens = vec![last_page_len as i32; batch_size];
        let request_indices: Vec<i32> = (0..batch_size as i32).collect();
        let kv_tile_indices = vec![0i32; batch_size];
        let kv_chunk_sizes = vec![kv_len as i32; batch_size];
        let split_chunk_size = split_config.actual_chunk_size(kv_len);
        let split_chunks_per_request = split_config.active_chunks(kv_len);
        let split_padded_slots = batch_size * split_config.max_chunks_per_request;
        anyhow::ensure!(
            split_chunks_per_request <= split_config.max_chunks_per_request,
            "split-K chunks/request exceeded padded slot budget"
        );

        let mut split_request_indices = Vec::with_capacity(split_padded_slots);
        let mut split_kv_tile_indices = Vec::with_capacity(split_padded_slots);
        let mut split_o_indptr = Vec::with_capacity(batch_size + 1);
        let mut split_block_valid_mask = Vec::with_capacity(split_padded_slots);
        split_o_indptr.push(0);
        for request_idx in 0..batch_size {
            for chunk_idx in 0..split_chunks_per_request {
                split_request_indices.push(request_idx as i32);
                split_kv_tile_indices.push(chunk_idx as i32);
                split_block_valid_mask.push(1);
            }
            split_o_indptr.push(split_request_indices.len() as i32);
        }
        while split_request_indices.len() < split_padded_slots {
            split_request_indices.push(0);
            split_kv_tile_indices.push(0);
            split_block_valid_mask.push(0);
        }
        let split_kv_chunk_sizes = [split_chunk_size as i32];

        let start = ctx
            .ctx
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let end = ctx
            .ctx
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;

        let page_indices_d = ctx.stream.clone_htod(&page_indices)?;
        let page_indptr_d = ctx.stream.clone_htod(&page_indptr)?;
        let last_page_len_d = ctx.stream.clone_htod(&last_page_lens)?;
        let request_indices_d = ctx.stream.clone_htod(&request_indices)?;
        let kv_tile_indices_d = ctx.stream.clone_htod(&kv_tile_indices)?;
        let kv_chunk_size_d = ctx.stream.clone_htod(&kv_chunk_sizes)?;
        let split_request_indices_d = ctx.stream.clone_htod(&split_request_indices)?;
        let split_kv_tile_indices_d = ctx.stream.clone_htod(&split_kv_tile_indices)?;
        let split_kv_chunk_size_d = ctx.stream.clone_htod(&split_kv_chunk_sizes)?;
        let split_o_indptr_d = ctx.stream.clone_htod(&split_o_indptr)?;
        let split_block_valid_mask_d = ctx.stream.clone_htod(&split_block_valid_mask)?;
        let split_tmp_v = ctx.stream.alloc_zeros(split_padded_slots * q_dim)?;
        let split_tmp_s = ctx.stream.alloc_zeros(split_padded_slots * NUM_QO_HEADS)?;

        let case = Self {
            ctx,
            layout,
            q,
            output,
            kv_buffer,
            page_indices_d,
            page_indptr_d,
            last_page_len_d,
            request_indices_d,
            kv_tile_indices_d,
            kv_chunk_size_d,
            split_request_indices_d,
            split_kv_tile_indices_d,
            split_kv_chunk_size_d,
            split_o_indptr_d,
            split_block_valid_mask_d,
            split_tmp_v,
            split_tmp_s,
            split_padded_slots,
            split_config,
            start,
            end,
            batch_size,
            kv_len,
        };
        case.ctx.sync()?;
        Ok(case)
    }

    pub fn shape(&self) -> AttentionKernelShape {
        AttentionKernelShape::new(self.batch_size, self.kv_len)
    }

    pub fn split_config(&self) -> SplitKvConfig {
        self.split_config
    }

    pub fn kv_read_bytes(&self) -> u64 {
        (self.batch_size * self.kv_len * NUM_KV_HEADS * HEAD_DIM * 2 * size_of::<bf16>()) as u64
    }

    pub fn split_chunk_size(&self) -> usize {
        self.split_config.actual_chunk_size(self.kv_len)
    }

    pub fn split_active_chunks_per_request(&self) -> usize {
        self.split_config.active_chunks(self.kv_len)
    }

    pub fn split_padded_slots(&self) -> usize {
        self.split_padded_slots
    }

    pub fn cu_context_ptr(&self) -> *mut c_void {
        self.ctx.ctx.cu_ctx().cast::<c_void>()
    }

    pub fn launch_once(&mut self, path: DecodePath) -> Result<()> {
        self.launch_inner(path)?;
        Ok(())
    }

    fn launch_inner(&mut self, path: DecodePath) -> Result<i32> {
        let (q_ptr, _q_guard) = self.q.data.device_ptr(&self.ctx.stream);
        let (out_ptr, _out_guard) = self.output.data.device_ptr_mut(&self.ctx.stream);
        let (kv_ptr, _kv_guard) = self.kv_buffer.device_ptr(&self.ctx.stream);
        let (page_indices_ptr, _page_indices_guard) =
            self.page_indices_d.device_ptr(&self.ctx.stream);
        let (page_indptr_ptr, _page_indptr_guard) = self.page_indptr_d.device_ptr(&self.ctx.stream);
        let (last_page_len_ptr, _last_page_len_guard) =
            self.last_page_len_d.device_ptr(&self.ctx.stream);
        let (request_indices_ptr, _request_indices_guard) =
            self.request_indices_d.device_ptr(&self.ctx.stream);
        let (kv_tile_indices_ptr, _kv_tile_indices_guard) =
            self.kv_tile_indices_d.device_ptr(&self.ctx.stream);
        let (kv_chunk_size_ptr, _kv_chunk_size_guard) =
            self.kv_chunk_size_d.device_ptr(&self.ctx.stream);
        let (split_request_indices_ptr, _split_request_indices_guard) =
            self.split_request_indices_d.device_ptr(&self.ctx.stream);
        let (split_kv_tile_indices_ptr, _split_kv_tile_indices_guard) =
            self.split_kv_tile_indices_d.device_ptr(&self.ctx.stream);
        let (split_kv_chunk_size_ptr, _split_kv_chunk_size_guard) =
            self.split_kv_chunk_size_d.device_ptr(&self.ctx.stream);
        let (split_o_indptr_ptr, _split_o_indptr_guard) =
            self.split_o_indptr_d.device_ptr(&self.ctx.stream);
        let (split_block_valid_mask_ptr, _split_block_valid_mask_guard) =
            self.split_block_valid_mask_d.device_ptr(&self.ctx.stream);
        let (split_tmp_v_ptr, _split_tmp_v_guard) =
            self.split_tmp_v.device_ptr_mut(&self.ctx.stream);
        let (split_tmp_s_ptr, _split_tmp_s_guard) =
            self.split_tmp_s.device_ptr_mut(&self.ctx.stream);

        let k_offset_elems = 0i64;
        let v_offset_elems = self.layout.kv_block_len as i64;
        let stride_page = self.layout.page_stride as i64;
        let sm_scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
        let stream = self.ctx.stream.cu_stream();
        let result = match path {
            DecodePath::NonPartition => unsafe {
                ffi::paged_attention_decode_cuda(
                    q_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    kv_ptr as *const ffi::Half,
                    k_offset_elems,
                    v_offset_elems,
                    page_indices_ptr as *const i32,
                    page_indptr_ptr as *const i32,
                    last_page_len_ptr as *const i32,
                    request_indices_ptr as *const i32,
                    kv_tile_indices_ptr as *const i32,
                    kv_chunk_size_ptr as *const i32,
                    NUM_QO_HEADS as i32,
                    NUM_KV_HEADS as i32,
                    HEAD_DIM as i32,
                    PAGE_SIZE as i32,
                    self.batch_size as i32,
                    stride_page,
                    sm_scale,
                    stream,
                )
            },
            DecodePath::SplitK => unsafe {
                ffi::paged_attention_decode_split_kv_cuda(
                    q_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    kv_ptr as *const ffi::Half,
                    k_offset_elems,
                    v_offset_elems,
                    page_indices_ptr as *const i32,
                    page_indptr_ptr as *const i32,
                    last_page_len_ptr as *const i32,
                    split_request_indices_ptr as *const i32,
                    split_kv_tile_indices_ptr as *const i32,
                    split_kv_chunk_size_ptr as *const i32,
                    split_o_indptr_ptr as *const i32,
                    split_block_valid_mask_ptr as *const u8,
                    split_tmp_v_ptr as *mut ffi::Half,
                    split_tmp_s_ptr as *mut f32,
                    NUM_QO_HEADS as i32,
                    NUM_KV_HEADS as i32,
                    HEAD_DIM as i32,
                    PAGE_SIZE as i32,
                    self.batch_size as i32,
                    self.split_padded_slots as i32,
                    stride_page,
                    sm_scale,
                    stream,
                )
            },
        };
        if result != 0 {
            bail!(
                "{} paged attention failed with error {result}",
                path.name(self.split_config)
            );
        }
        Ok(result)
    }

    pub fn measure_decode_only(&mut self, criterion_iters: u64, path: DecodePath) -> Duration {
        let mut elapsed_ms = 0.0f64;

        for _ in 0..criterion_iters {
            self.start
                .record(&self.ctx.stream)
                .expect("record start event failed");
            for _ in 0..INNER_LAUNCHES {
                self.launch_once(path).expect("paged attention failed");
            }
            self.end
                .record(&self.ctx.stream)
                .expect("record end event failed");
            elapsed_ms += f64::from(
                self.start
                    .elapsed_ms(&self.end)
                    .expect("cuda event elapsed failed"),
            ) / INNER_LAUNCHES as f64;
        }

        Duration::from_secs_f64(elapsed_ms / 1_000.0)
    }

    pub fn measure_decode_only_cold_l2(
        &mut self,
        criterion_iters: u64,
        path: DecodePath,
        cache_clear: &mut L2CacheClear,
    ) -> Result<Duration> {
        let mut elapsed_ms = 0.0f64;

        for _ in 0..criterion_iters {
            cache_clear.clear(&self.ctx)?;
            self.start.record(&self.ctx.stream)?;
            self.launch_once(path)?;
            self.end.record(&self.ctx.stream)?;
            elapsed_ms += f64::from(self.start.elapsed_ms(&self.end)?);
        }

        Ok(Duration::from_secs_f64(elapsed_ms / 1_000.0))
    }
}

fn patterned_bf16(len: usize, scale: f32) -> Vec<bf16> {
    (0..len)
        .map(|i| bf16::from_f32((((i % 251) as f32) - 125.0) * scale))
        .collect()
}

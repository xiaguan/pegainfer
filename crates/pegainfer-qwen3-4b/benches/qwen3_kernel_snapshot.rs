use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use cudarc::driver::sys;
use pegainfer_cupti::profile_range_with_prepare;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_qwen3_4b::kernel_bench::{
    AttentionDecodeCase, AttentionKernelShape, AttentionKernelSpec, AttentionKernelVariant,
    BATCH_SIZES, CONTEXT_LENGTHS, DecodePath, DevicePeakBandwidth, HEAD_DIM, INNER_LAUNCHES,
    L2CacheClear, NUM_KV_HEADS, NUM_QO_HEADS, PAGE_SIZE, REPORT_ITERS, SplitKvConfig,
    cache_clear_bytes,
};
use serde::{Deserialize, Serialize};

const CUPTI_METRICS: &[&str] = &[
    "gpu__time_duration.sum",
    "dram__bytes.sum",
    "dram__bytes_op_read.sum",
    "dram__bytes_op_write.sum",
    "lts__t_bytes.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__warps_active.avg.pct_of_peak_sustained_active",
];
const CUPTI_GPU_TIME_NS_IDX: usize = 0;
const CUPTI_DRAM_BYTES_IDX: usize = 1;
const CUPTI_DRAM_READ_BYTES_IDX: usize = 2;
const CUPTI_DRAM_WRITE_BYTES_IDX: usize = 3;
const CUPTI_L2_BYTES_IDX: usize = 4;
const CUPTI_SM_THROUGHPUT_PCT_IDX: usize = 5;
const CUPTI_SMSP_WARPS_ACTIVE_PCT_IDX: usize = 6;
const LATENCY_WARN_PCT: f64 = 5.0;
const LATENCY_FAIL_PCT: f64 = 10.0;
const DRAM_READ_AMPLIFICATION_FAIL: f64 = 1.2;

enum ParsedArgs {
    Run(RunArgs),
    Compare(CompareArgs),
    Help,
}

struct RunArgs {
    out: Option<PathBuf>,
    cupti: bool,
    iters: u64,
    contexts: Vec<usize>,
    batch_sizes: Vec<usize>,
    variants: Vec<AttentionKernelVariant>,
}

struct ExternalProvenance {
    git: GitProvenance,
    build: BuildProvenance,
    driver_version: Option<String>,
    cuda_toolkit: Option<String>,
}

struct CompareArgs {
    base: PathBuf,
    new: PathBuf,
}

#[derive(Deserialize, Serialize)]
struct KernelSnapshot {
    schema: u32,
    model: String,
    op: String,
    created_at_unix_secs: u64,
    git: GitProvenance,
    hardware: HardwareProvenance,
    build: BuildProvenance,
    measurement: MeasurementConfig,
    cases: Vec<CaseResult>,
}

#[derive(Deserialize, Serialize)]
struct GitProvenance {
    commit: Option<String>,
    dirty: Option<bool>,
}

#[derive(Deserialize, Serialize)]
struct HardwareProvenance {
    gpu_name: String,
    device_ordinal: usize,
    compute_capability: String,
    driver_version: Option<String>,
    cuda_toolkit: Option<String>,
    memory_clock_khz: i32,
    memory_bus_width_bits: i32,
    peak_gb_s: f64,
    l2_bytes: usize,
    cache_clear_bytes: usize,
}

#[derive(Deserialize, Serialize)]
struct BuildProvenance {
    target_sm_env: Option<String>,
    flashinfer_commit: Option<String>,
    kernel_archive: Option<String>,
    kernel_archive_fnv1a64: Option<String>,
}

#[derive(Deserialize, Serialize)]
struct MeasurementConfig {
    warm_iters: u64,
    cold_l2_iters: u64,
    inner_launches: u64,
    cupti_enabled: bool,
    cupti_metrics: Vec<String>,
}

#[derive(Deserialize, Serialize)]
struct CaseResult {
    op: String,
    variant: String,
    shape: CaseShape,
    params: CaseParams,
    theoretical_kv_read_bytes: u64,
    warm_latency_us: Option<f64>,
    cold_l2_latency_us: Option<f64>,
    cupti: Option<CuptiMeasurement>,
    error: Option<String>,
}

#[derive(Deserialize, Serialize)]
struct CaseShape {
    batch_size: usize,
    kv_len: usize,
    num_qo_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    page_size: usize,
    dtype: String,
}

#[derive(Deserialize, Serialize)]
struct CaseParams {
    chunk_tokens: Option<usize>,
    max_chunks_per_request: Option<usize>,
    actual_chunk_size: Option<usize>,
    active_chunks_per_request: Option<usize>,
    padded_slots: Option<usize>,
}

#[derive(Deserialize, Serialize)]
struct CuptiMeasurement {
    gpu_time_us: f64,
    dram_read_bytes: f64,
    dram_write_bytes: f64,
    dram_total_bytes: f64,
    l2_total_bytes: f64,
    sm_throughput_pct: f64,
    smsp_warps_active_pct: f64,
    dram_gb_s: f64,
    dram_peak_pct: f64,
    kv_read_over_dram_read_pct: f64,
}

fn usage() -> &'static str {
    "usage:
  cargo bench -p pegainfer-qwen3-4b --bench qwen3_kernel_snapshot -- run [--out PATH] [--no-cupti] [--iters N] [--contexts 1024,4096] [--batch-sizes 1,2] [--variants non_partition,split_kv_256x64]
  cargo bench -p pegainfer-qwen3-4b --bench qwen3_kernel_snapshot -- compare --base PATH --new PATH"
}

fn parse_args() -> Result<ParsedArgs> {
    let mut raw = std::env::args().skip(1);
    let Some(command) = raw.next() else {
        return Ok(ParsedArgs::Run(default_run_args()));
    };
    match command.as_str() {
        "-h" | "--help" => Ok(ParsedArgs::Help),
        "run" => parse_run_args(raw.collect()),
        "compare" => parse_compare_args(raw.collect()),
        _ => bail!("unknown command `{command}`\n{}", usage()),
    }
}

fn default_run_args() -> RunArgs {
    RunArgs {
        out: None,
        cupti: true,
        iters: REPORT_ITERS,
        contexts: CONTEXT_LENGTHS.to_vec(),
        batch_sizes: BATCH_SIZES.to_vec(),
        variants: default_variants(),
    }
}

fn default_variants() -> Vec<AttentionKernelVariant> {
    vec![
        AttentionKernelVariant::NonPartition,
        AttentionKernelVariant::SplitKv(SplitKvConfig::new(256, 64)),
        AttentionKernelVariant::SplitKv(SplitKvConfig::new(512, 64)),
    ]
}

fn parse_run_args(args: Vec<String>) -> Result<ParsedArgs> {
    let mut parsed = default_run_args();
    let mut raw = args.into_iter();
    while let Some(arg) = raw.next() {
        match arg.as_str() {
            "--bench" => {}
            "-h" | "--help" => return Ok(ParsedArgs::Help),
            "--out" => {
                parsed.out = Some(PathBuf::from(
                    raw.next().ok_or_else(|| anyhow!("--out needs a value"))?,
                ));
            }
            "--cupti" => parsed.cupti = true,
            "--no-cupti" => parsed.cupti = false,
            "--iters" => {
                let value = raw.next().ok_or_else(|| anyhow!("--iters needs a value"))?;
                parsed.iters = parse_positive_u64("--iters", &value)?;
            }
            "--contexts" => {
                let value = raw
                    .next()
                    .ok_or_else(|| anyhow!("--contexts needs a value"))?;
                parsed.contexts = parse_usize_list("--contexts", &value)?;
            }
            "--batch-sizes" => {
                let value = raw
                    .next()
                    .ok_or_else(|| anyhow!("--batch-sizes needs a value"))?;
                parsed.batch_sizes = parse_usize_list("--batch-sizes", &value)?;
            }
            "--variants" => {
                let value = raw
                    .next()
                    .ok_or_else(|| anyhow!("--variants needs a value"))?;
                parsed.variants = parse_variants(&value)?;
            }
            _ => bail!("unknown run argument `{arg}`\n{}", usage()),
        }
    }
    Ok(ParsedArgs::Run(parsed))
}

fn parse_compare_args(args: Vec<String>) -> Result<ParsedArgs> {
    let mut base = None;
    let mut new = None;
    let mut raw = args.into_iter();
    while let Some(arg) = raw.next() {
        match arg.as_str() {
            "--bench" => {}
            "-h" | "--help" => return Ok(ParsedArgs::Help),
            "--base" => {
                base = Some(PathBuf::from(
                    raw.next().ok_or_else(|| anyhow!("--base needs a value"))?,
                ));
            }
            "--new" => {
                new = Some(PathBuf::from(
                    raw.next().ok_or_else(|| anyhow!("--new needs a value"))?,
                ));
            }
            _ => bail!("unknown compare argument `{arg}`\n{}", usage()),
        }
    }
    Ok(ParsedArgs::Compare(CompareArgs {
        base: base.ok_or_else(|| anyhow!("compare requires --base"))?,
        new: new.ok_or_else(|| anyhow!("compare requires --new"))?,
    }))
}

fn parse_positive_u64(name: &str, raw: &str) -> Result<u64> {
    let value = raw
        .parse::<u64>()
        .with_context(|| format!("invalid {name} `{raw}`"))?;
    anyhow::ensure!(value > 0, "{name} must be greater than zero");
    Ok(value)
}

fn parse_usize_list(name: &str, raw: &str) -> Result<Vec<usize>> {
    let values: Vec<_> = raw
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(|token| {
            let value = token
                .parse::<usize>()
                .with_context(|| format!("invalid value `{token}` in {name}"))?;
            anyhow::ensure!(value > 0, "{name} values must be greater than zero");
            Ok(value)
        })
        .collect::<Result<_>>()?;
    anyhow::ensure!(!values.is_empty(), "{name} must not be empty");
    Ok(values)
}

fn parse_variants(raw: &str) -> Result<Vec<AttentionKernelVariant>> {
    let variants: Vec<_> = raw
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(parse_variant)
        .collect::<Result<_>>()?;
    anyhow::ensure!(!variants.is_empty(), "--variants must not be empty");
    Ok(variants)
}

fn parse_variant(raw: &str) -> Result<AttentionKernelVariant> {
    if raw == "non_partition" {
        return Ok(AttentionKernelVariant::NonPartition);
    }
    let Some(rest) = raw.strip_prefix("split_kv_") else {
        bail!("unknown variant `{raw}`");
    };
    let Some((chunk, max_chunks)) = rest.split_once('x') else {
        bail!("split-K variant must look like split_kv_256x64");
    };
    Ok(AttentionKernelVariant::SplitKv(SplitKvConfig::new(
        chunk
            .parse::<usize>()
            .with_context(|| format!("invalid chunk size in `{raw}`"))?,
        max_chunks
            .parse::<usize>()
            .with_context(|| format!("invalid max chunks in `{raw}`"))?,
    )))
}

fn selected_specs(args: &RunArgs) -> Vec<AttentionKernelSpec> {
    args.batch_sizes
        .iter()
        .copied()
        .flat_map(|batch_size| {
            args.contexts.iter().copied().flat_map(move |kv_len| {
                args.variants
                    .iter()
                    .copied()
                    .map(move |variant| AttentionKernelSpec {
                        shape: AttentionKernelShape::new(batch_size, kv_len),
                        variant,
                    })
            })
        })
        .collect()
}

fn run_snapshot(args: &RunArgs) -> Result<KernelSnapshot> {
    let external = query_external_provenance();
    let probe = DeviceContext::new()?;
    let peak = DevicePeakBandwidth::query(&probe)?;
    drop(probe);

    let specs = selected_specs(args);
    eprintln!(
        "running {} qwen3 paged decode attention kernel cases",
        specs.len()
    );
    let mut cases = Vec::with_capacity(specs.len());
    for spec in specs {
        eprintln!(
            "case variant={} bs={} ctx={}",
            spec.variant.label(),
            spec.shape.batch_size,
            spec.shape.kv_len
        );
        cases.push(measure_case(spec, args, &peak));
    }

    let probe = DeviceContext::new()?;
    let hardware = query_hardware(
        &probe,
        &peak,
        external.driver_version.clone(),
        external.cuda_toolkit.clone(),
    )?;

    Ok(KernelSnapshot {
        schema: 1,
        model: "qwen3-4b".to_string(),
        op: "paged_decode_attention".to_string(),
        created_at_unix_secs: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        git: external.git,
        hardware,
        build: external.build,
        measurement: MeasurementConfig {
            warm_iters: args.iters,
            cold_l2_iters: args.iters,
            inner_launches: INNER_LAUNCHES,
            cupti_enabled: args.cupti,
            cupti_metrics: if args.cupti {
                CUPTI_METRICS
                    .iter()
                    .map(|metric| (*metric).to_string())
                    .collect()
            } else {
                Vec::new()
            },
        },
        cases,
    })
}

fn measure_case(
    spec: AttentionKernelSpec,
    args: &RunArgs,
    peak: &DevicePeakBandwidth,
) -> CaseResult {
    let variant = spec.variant.label();
    let path = spec.variant.decode_path();
    let mut result = empty_case_result(spec);
    let case_result = (|| -> Result<()> {
        if args.cupti {
            result.cupti = Some(measure_cupti(spec, path, peak)?);
        }

        {
            let mut case = AttentionDecodeCase::for_spec(spec)?;
            let warm = case.measure_decode_only(args.iters, path);
            result.warm_latency_us = Some(warm.as_secs_f64() * 1.0e6 / args.iters as f64);

            let mut cache_clear = L2CacheClear::new(&case.ctx)?;
            let cold = case.measure_decode_only_cold_l2(args.iters, path, &mut cache_clear)?;
            result.cold_l2_latency_us = Some(cold.as_secs_f64() * 1.0e6 / args.iters as f64);
        }
        Ok(())
    })();

    if let Err(err) = case_result {
        result.error = Some(format!("{err:#}"));
        eprintln!(
            "case failed variant={} bs={} ctx={}: {err:#}",
            variant, spec.shape.batch_size, spec.shape.kv_len
        );
    }
    result
}

fn empty_case_result(spec: AttentionKernelSpec) -> CaseResult {
    let params = match spec.variant {
        AttentionKernelVariant::NonPartition => CaseParams {
            chunk_tokens: None,
            max_chunks_per_request: None,
            actual_chunk_size: None,
            active_chunks_per_request: None,
            padded_slots: None,
        },
        AttentionKernelVariant::SplitKv(config) => CaseParams {
            chunk_tokens: Some(config.chunk_tokens),
            max_chunks_per_request: Some(config.max_chunks_per_request),
            actual_chunk_size: Some(config.actual_chunk_size(spec.shape.kv_len)),
            active_chunks_per_request: Some(config.active_chunks(spec.shape.kv_len)),
            padded_slots: Some(spec.shape.batch_size * config.max_chunks_per_request),
        },
    };
    CaseResult {
        op: "paged_decode_attention".to_string(),
        variant: spec.variant.label(),
        shape: CaseShape {
            batch_size: spec.shape.batch_size,
            kv_len: spec.shape.kv_len,
            num_qo_heads: NUM_QO_HEADS,
            num_kv_heads: NUM_KV_HEADS,
            head_dim: HEAD_DIM,
            page_size: PAGE_SIZE,
            dtype: "bf16".to_string(),
        },
        params,
        theoretical_kv_read_bytes: (spec.shape.batch_size
            * spec.shape.kv_len
            * NUM_KV_HEADS
            * HEAD_DIM
            * 2
            * std::mem::size_of::<half::bf16>()) as u64,
        warm_latency_us: None,
        cold_l2_latency_us: None,
        cupti: None,
        error: None,
    }
}

fn measure_cupti(
    spec: AttentionKernelSpec,
    path: DecodePath,
    peak: &DevicePeakBandwidth,
) -> Result<CuptiMeasurement> {
    let mut case = AttentionDecodeCase::for_spec(spec)?;
    case.launch_once(path)?;
    case.ctx.sync()?;

    let mut cache_clear = L2CacheClear::new(&case.ctx)?;
    let context = case.cu_context_ptr();
    let device_ordinal = case.ctx.device_ordinal;
    let clear_ctx = case.ctx.clone();
    let range_name = format!(
        "qk/{}/b{}/k{}",
        path.name(case.split_config()),
        case.shape().batch_size,
        case.shape().kv_len
    );

    let mut prepare = || {
        cache_clear
            .clear(&clear_ctx)
            .map_err(|err| format!("{err:#}"))
    };
    let mut launch = || case.launch_once(path).map_err(|err| format!("{err:#}"));
    let values = unsafe {
        profile_range_with_prepare(
            context,
            device_ordinal,
            &range_name,
            CUPTI_METRICS,
            Some(&mut prepare),
            &mut launch,
        )?
    };

    let gpu_time_ns = values[CUPTI_GPU_TIME_NS_IDX];
    let dram_total_bytes = values[CUPTI_DRAM_BYTES_IDX];
    let dram_read_bytes = values[CUPTI_DRAM_READ_BYTES_IDX];
    let dram_write_bytes = values[CUPTI_DRAM_WRITE_BYTES_IDX];
    let l2_bytes = values[CUPTI_L2_BYTES_IDX];
    let sm_throughput_pct = values[CUPTI_SM_THROUGHPUT_PCT_IDX];
    let smsp_warps_active_pct = values[CUPTI_SMSP_WARPS_ACTIVE_PCT_IDX];
    let dram_gb_s = dram_total_bytes / (gpu_time_ns * 1.0e-9) / 1.0e9;
    let dram_peak_pct = dram_gb_s / peak.peak_gb_per_sec() * 100.0;
    let kv_read = case.kv_read_bytes() as f64;
    let kv_read_over_dram_read_pct = if dram_read_bytes > 0.0 {
        kv_read / dram_read_bytes * 100.0
    } else {
        f64::NAN
    };

    Ok(CuptiMeasurement {
        gpu_time_us: gpu_time_ns / 1.0e3,
        dram_read_bytes,
        dram_write_bytes,
        dram_total_bytes,
        l2_total_bytes: l2_bytes,
        sm_throughput_pct,
        smsp_warps_active_pct,
        dram_gb_s,
        dram_peak_pct,
        kv_read_over_dram_read_pct,
    })
}

fn query_external_provenance() -> ExternalProvenance {
    ExternalProvenance {
        git: GitProvenance {
            commit: None,
            dirty: None,
        },
        build: BuildProvenance {
            target_sm_env: std::env::var("PEGAINFER_CUDA_SM").ok(),
            flashinfer_commit: None,
            kernel_archive: None,
            kernel_archive_fnv1a64: None,
        },
        driver_version: None,
        cuda_toolkit: None,
    }
}

fn query_hardware(
    ctx: &DeviceContext,
    peak: &DevicePeakBandwidth,
    driver_version: Option<String>,
    cuda_toolkit: Option<String>,
) -> Result<HardwareProvenance> {
    let l2_bytes = ctx
        .ctx
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)?
        as usize;
    let (cc_major, cc_minor) = ctx.ctx.compute_capability()?;
    Ok(HardwareProvenance {
        gpu_name: ctx.ctx.name()?,
        device_ordinal: ctx.device_ordinal,
        compute_capability: format!("{cc_major}.{cc_minor}"),
        driver_version,
        cuda_toolkit,
        memory_clock_khz: peak.memory_clock_khz,
        memory_bus_width_bits: peak.memory_bus_width_bits,
        peak_gb_s: peak.peak_gb_per_sec(),
        l2_bytes,
        cache_clear_bytes: cache_clear_bytes(l2_bytes),
    })
}

fn write_snapshot(snapshot: &KernelSnapshot, out: Option<&Path>) -> Result<()> {
    let json = serde_json::to_string_pretty(snapshot)?;
    match out {
        Some(path) => {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut file = File::create(path)?;
            file.write_all(json.as_bytes())?;
            file.write_all(b"\n")?;
            eprintln!("wrote {}", path.display());
        }
        None => println!("{json}"),
    }
    Ok(())
}

fn read_snapshot(path: &Path) -> Result<KernelSnapshot> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("failed to read snapshot {}", path.display()))?;
    serde_json::from_str(&content).with_context(|| format!("failed to parse {}", path.display()))
}

fn compare_snapshots(args: &CompareArgs) -> Result<()> {
    let base = read_snapshot(&args.base)?;
    let new = read_snapshot(&args.new)?;
    let base_cases: HashMap<_, _> = base
        .cases
        .iter()
        .map(|case| (case_key(case), case))
        .collect();
    let mut warnings = 0usize;
    let mut failures = 0usize;

    for case in &new.cases {
        let key = case_key(case);
        if let Some(cupti) = &case.cupti {
            let max_read = case.theoretical_kv_read_bytes as f64 * DRAM_READ_AMPLIFICATION_FAIL;
            if cupti.dram_read_bytes > max_read {
                println!(
                    "FAIL dram_read_amp {key}: dram_read={:.0} theoretical={}",
                    cupti.dram_read_bytes, case.theoretical_kv_read_bytes
                );
                failures += 1;
            }
        }
        let Some(base_case) = base_cases.get(&key) else {
            println!("WARN new case {key}");
            warnings += 1;
            continue;
        };
        compare_latency(
            &key,
            "warm_latency_us",
            base_case.warm_latency_us,
            case.warm_latency_us,
            &mut warnings,
            &mut failures,
        );
        compare_latency(
            &key,
            "cold_l2_latency_us",
            base_case.cold_l2_latency_us,
            case.cold_l2_latency_us,
            &mut warnings,
            &mut failures,
        );
    }

    println!("kernel snapshot compare complete: warnings={warnings} failures={failures}");
    if failures > 0 {
        bail!("{failures} kernel snapshot comparison failures");
    }
    Ok(())
}

fn case_key(case: &CaseResult) -> String {
    format!(
        "{}|bs{}|ctx{}",
        case.variant, case.shape.batch_size, case.shape.kv_len
    )
}

fn compare_latency(
    key: &str,
    metric: &str,
    base: Option<f64>,
    new: Option<f64>,
    warnings: &mut usize,
    failures: &mut usize,
) {
    let (Some(base), Some(new)) = (base, new) else {
        println!("WARN missing {metric} {key}");
        *warnings += 1;
        return;
    };
    if base <= 0.0 {
        return;
    }
    let pct = (new / base - 1.0) * 100.0;
    if pct > LATENCY_FAIL_PCT {
        println!("FAIL {metric} {key}: {base:.3} -> {new:.3} ({pct:+.2}%)");
        *failures += 1;
    } else if pct > LATENCY_WARN_PCT {
        println!("WARN {metric} {key}: {base:.3} -> {new:.3} ({pct:+.2}%)");
        *warnings += 1;
    }
}

fn main() -> Result<()> {
    match parse_args()? {
        ParsedArgs::Run(args) => {
            let snapshot = run_snapshot(&args)?;
            write_snapshot(&snapshot, args.out.as_deref())?;
            let failed = snapshot.cases.iter().any(|case| case.error.is_some());
            if failed {
                bail!("one or more kernel cases failed");
            }
        }
        ParsedArgs::Compare(args) => compare_snapshots(&args)?,
        ParsedArgs::Help => println!("{}", usage()),
    }
    Ok(())
}

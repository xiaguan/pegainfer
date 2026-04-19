/// DSV3.2 teacher-forced top-K logprob regression test against SGLang ground truth.
///
/// For each case we prefill the prompt, then step decode over the SGLang-generated
/// token sequence (teacher forcing, ignoring pegainfer's own argmax). At every
/// output position we compare pegainfer's top-K distribution to the manifest's
/// top-K, collecting argmax match + |Δlogprob| statistics.
///
/// On the first baseline run the threshold is unset and the test only reports.
/// Once the numeric behavior is understood, callers can set
/// `PEGAINFER_DSV32_LOGPROB_MAX_ABS` / `PEGAINFER_DSV32_ARGMAX_MISMATCHES` to
/// make it fail on regression.
use std::collections::HashSet;
use std::path::PathBuf;

use half::bf16;
use pegainfer::model::DsV32Executor;
use serde::Deserialize;

const DEFAULT_MODEL_PATH: &str = "/data/models/DeepSeek-V3.2";
const DEFAULT_MANIFEST_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/test_data/dsv32_sglang_ref/manifest.json"
);
const EXPECTED_SCHEMA: &str = "dsv32_sglang_ref.v1";

#[derive(Debug, Deserialize)]
struct SglangManifest {
    schema_version: String,
    #[serde(default)]
    meta: ManifestMeta,
    cases: Vec<SglangCase>,
}

#[derive(Debug, Default, Deserialize)]
struct ManifestMeta {
    #[serde(default)]
    top_k: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct SglangCase {
    name: String,
    prompt: String,
    prompt_token_ids: Vec<u32>,
    generated_token_ids: Vec<u32>,
    /// Per-output-position top-K entries as `[token_id, logprob]` pairs,
    /// sorted descending by logprob.
    output_top_logprobs: Vec<Vec<(u32, f32)>>,
}

fn parse_device_ordinals() -> Vec<usize> {
    let raw = std::env::var("PEGAINFER_DSV32_DEVICE_ORDINALS")
        .unwrap_or_else(|_| "0,1,2,3,4,5,6,7".to_string());
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .unwrap_or_else(|_| panic!("invalid device ordinal `{s}`"))
        })
        .collect()
}

fn parse_tp_size() -> usize {
    std::env::var("PEGAINFER_DSV32_TP_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(1)
}

fn parse_case_filter() -> Option<String> {
    std::env::var("PEGAINFER_DSV32_CASE_FILTER")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn parse_max_cases() -> Option<usize> {
    std::env::var("PEGAINFER_DSV32_MAX_CASES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|v| *v > 0)
}

fn parse_f32_env(key: &str) -> Option<f32> {
    std::env::var(key).ok().and_then(|s| s.parse::<f32>().ok())
}

fn parse_usize_env(key: &str) -> Option<usize> {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
}

/// Converts a bf16 logits vector into `(argmax_token, top_k_entries, full_logprobs)`.
///
/// `top_k_entries` is sorted descending by logprob. `full_logprobs[i]` is
/// `logit[i] - log_sum_exp` so we can probe arbitrary token ids later.
fn logits_to_logprobs_topk(logits: &[bf16], k: usize) -> (u32, Vec<(u32, f32)>, Vec<f32>) {
    assert!(!logits.is_empty(), "empty logits vector");

    let mut max_val = f32::NEG_INFINITY;
    let mut argmax: u32 = 0;
    let logits_f32: Vec<f32> = logits
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let v = x.to_f32();
            if v > max_val {
                max_val = v;
                argmax = i as u32;
            }
            v
        })
        .collect();

    let mut sum_exp = 0.0f64;
    for &v in &logits_f32 {
        sum_exp += ((v - max_val) as f64).exp();
    }
    let log_sum_exp = max_val as f64 + sum_exp.ln();

    let logprobs: Vec<f32> = logits_f32
        .iter()
        .map(|&v| (v as f64 - log_sum_exp) as f32)
        .collect();

    let k = k.min(logits.len());
    let mut indexed: Vec<(u32, f32)> = logprobs
        .iter()
        .enumerate()
        .map(|(i, &lp)| (i as u32, lp))
        .collect();
    indexed.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
    let mut top: Vec<(u32, f32)> = indexed.into_iter().take(k).collect();
    top.sort_by(|a, b| b.1.total_cmp(&a.1));

    (argmax, top, logprobs)
}

#[derive(Default)]
struct PositionStats {
    argmax_mismatches: usize,
    positions: usize,
    abs_deltas: Vec<f32>,
    overlap_counts: Vec<usize>,
    /// Positions where the manifest's argmax token was NOT in pegainfer's top-K.
    argmax_outside_topk: usize,
}

impl PositionStats {
    fn merge(&mut self, other: &PositionStats) {
        self.argmax_mismatches += other.argmax_mismatches;
        self.positions += other.positions;
        self.abs_deltas.extend_from_slice(&other.abs_deltas);
        self.overlap_counts.extend_from_slice(&other.overlap_counts);
        self.argmax_outside_topk += other.argmax_outside_topk;
    }

    fn report(&self, label: &str) {
        if self.positions == 0 {
            eprintln!("{label}: no positions recorded");
            return;
        }
        let max_abs = self.abs_deltas.iter().copied().fold(0.0f32, f32::max);
        let mean_abs: f32 = if self.abs_deltas.is_empty() {
            0.0
        } else {
            self.abs_deltas.iter().copied().sum::<f32>() / self.abs_deltas.len() as f32
        };
        let p99 = percentile(&self.abs_deltas, 0.99);
        let p50 = percentile(&self.abs_deltas, 0.50);
        let avg_overlap: f32 = if self.overlap_counts.is_empty() {
            0.0
        } else {
            self.overlap_counts.iter().copied().sum::<usize>() as f32
                / self.overlap_counts.len() as f32
        };
        eprintln!(
            "{label}: positions={} argmax_mismatches={} ({:.2}%) argmax_outside_topk={} \
             |Δlogprob| max={:.6} mean={:.6} p50={:.6} p99={:.6} \
             top-K overlap mean={:.2}",
            self.positions,
            self.argmax_mismatches,
            100.0 * self.argmax_mismatches as f32 / self.positions as f32,
            self.argmax_outside_topk,
            max_abs,
            mean_abs,
            p50,
            p99,
            avg_overlap,
        );
    }
}

fn percentile(values: &[f32], q: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let idx = ((sorted.len() as f32 - 1.0) * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[test]
#[ignore = "requires 8 GPUs and DeepSeek-V3.2 weights; run manually on H20"]
fn dsv32_teacher_forced_sglang() {
    let model_path = std::env::var("PEGAINFER_DSV32_MODEL_PATH")
        .unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_string());
    if !PathBuf::from(&model_path).exists() {
        eprintln!(
            "Skipped: model path not found. Set PEGAINFER_DSV32_MODEL_PATH (current: {model_path})"
        );
        return;
    }

    let manifest_path = std::env::var("PEGAINFER_DSV32_SGLANG_REF_MANIFEST")
        .unwrap_or_else(|_| DEFAULT_MANIFEST_PATH.to_string());
    let manifest_path_buf = PathBuf::from(&manifest_path);
    if !manifest_path_buf.exists() {
        eprintln!(
            "Skipped: sglang manifest not found. Generate it first (current: {manifest_path})"
        );
        return;
    }

    let manifest_content = std::fs::read_to_string(&manifest_path_buf)
        .unwrap_or_else(|e| panic!("read manifest {}: {e}", manifest_path_buf.display()));
    let manifest: SglangManifest =
        serde_json::from_str(&manifest_content).expect("parse sglang manifest");
    assert_eq!(
        manifest.schema_version, EXPECTED_SCHEMA,
        "unexpected manifest schema version"
    );
    assert!(
        !manifest.cases.is_empty(),
        "manifest contains no cases: {}",
        manifest_path_buf.display()
    );

    let manifest_top_k = manifest.meta.top_k.unwrap_or_else(|| {
        manifest
            .cases
            .first()
            .and_then(|c| c.output_top_logprobs.first())
            .map(|row| row.len())
            .unwrap_or(20)
    });

    let case_filter = parse_case_filter();
    let max_cases = parse_max_cases();
    let fail_max_abs = parse_f32_env("PEGAINFER_DSV32_LOGPROB_MAX_ABS");
    let fail_max_argmax_mismatches = parse_usize_env("PEGAINFER_DSV32_ARGMAX_MISMATCHES");

    let selected: Vec<&SglangCase> = manifest
        .cases
        .iter()
        .filter(|c| {
            case_filter
                .as_ref()
                .map(|f| c.name.contains(f))
                .unwrap_or(true)
        })
        .take(max_cases.unwrap_or(usize::MAX))
        .collect();
    assert!(
        !selected.is_empty(),
        "no cases matched filter {:?}",
        case_filter
    );

    let device_ordinals = parse_device_ordinals();
    assert_eq!(
        device_ordinals.len(),
        8,
        "dsv32_teacher_forced_sglang expects 8 device ordinals, got {device_ordinals:?}"
    );
    let tp_size = parse_tp_size();
    assert_eq!(
        device_ordinals.len() % tp_size,
        0,
        "invalid parallel config: world_size={} tp_size={tp_size}",
        device_ordinals.len()
    );

    let executor = DsV32Executor::load(&model_path, &device_ordinals, tp_size)
        .unwrap_or_else(|e| panic!("DsV32Executor::load failed (tp_size={tp_size}): {e}"));

    let mut global_stats = PositionStats::default();

    for case in &selected {
        assert_eq!(
            case.output_top_logprobs.len(),
            case.generated_token_ids.len(),
            "case `{}`: output_top_logprobs / generated_token_ids length mismatch",
            case.name,
        );
        assert!(
            !case.generated_token_ids.is_empty(),
            "case `{}`: empty generated_token_ids",
            case.name,
        );

        executor
            .reset_generation_state()
            .expect("reset_generation_state failed");
        let positions: Vec<i32> = (0..case.prompt_token_ids.len()).map(|i| i as i32).collect();
        let mut logits = executor
            .prefill(&case.prompt_token_ids, &positions)
            .unwrap_or_else(|e| panic!("case `{}` prefill failed: {e}", case.name));

        let mut case_stats = PositionStats::default();
        for i in 0..case.generated_token_ids.len() {
            let (pega_argmax, pega_top, pega_logprobs) =
                logits_to_logprobs_topk(&logits, manifest_top_k);

            let manifest_top = &case.output_top_logprobs[i];
            assert!(
                !manifest_top.is_empty(),
                "case `{}` position {i}: empty manifest top-K",
                case.name,
            );
            let manifest_argmax = manifest_top[0].0;

            if pega_argmax != manifest_argmax {
                case_stats.argmax_mismatches += 1;
            }

            let pega_tokens: HashSet<u32> = pega_top.iter().map(|(t, _)| *t).collect();
            let manifest_tokens: HashSet<u32> = manifest_top.iter().map(|(t, _)| *t).collect();
            let overlap = pega_tokens.intersection(&manifest_tokens).count();
            case_stats.overlap_counts.push(overlap);
            if !pega_tokens.contains(&manifest_argmax) {
                case_stats.argmax_outside_topk += 1;
            }

            for &(tok, ref_lp) in manifest_top.iter() {
                let pega_lp = pega_logprobs
                    .get(tok as usize)
                    .copied()
                    .expect("vocab mismatch: manifest token out of pegainfer logits range");
                case_stats.abs_deltas.push((pega_lp - ref_lp).abs());
            }
            case_stats.positions += 1;

            if i + 1 < case.generated_token_ids.len() {
                let next_input = case.generated_token_ids[i];
                logits = executor.decode(next_input).unwrap_or_else(|e| {
                    panic!("case `{}` decode failed at step {i}: {e}", case.name)
                });
            }
        }

        case_stats.report(&format!("  case {}", case.name));
        if case_stats.argmax_mismatches > 0 {
            eprintln!(
                "    prompt: {:?}",
                &case.prompt.chars().take(80).collect::<String>()
            );
        }
        global_stats.merge(&case_stats);
    }

    eprintln!("========================================");
    global_stats.report("OVERALL");
    eprintln!("========================================");

    if let Some(limit) = fail_max_abs {
        let observed = global_stats
            .abs_deltas
            .iter()
            .copied()
            .fold(0.0f32, f32::max);
        assert!(
            observed <= limit,
            "max |Δlogprob| {observed:.6} exceeds threshold {limit:.6}"
        );
    }
    if let Some(limit) = fail_max_argmax_mismatches {
        assert!(
            global_stats.argmax_mismatches <= limit,
            "argmax mismatches {} exceeds limit {limit}",
            global_stats.argmax_mismatches
        );
    }
}

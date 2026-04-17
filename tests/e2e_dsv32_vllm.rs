/// DSV3.2 full-forward integration test against vLLM-generated ground truth.
///
/// This test is intentionally ignored by default because it requires 8 GPUs
/// and a large model checkpoint.
use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

use half::bf16;
use pegainfer::model::DsV32Executor;
use serde::Deserialize;

const DEFAULT_MODEL_PATH: &str = "/data/models/DeepSeek-V3.2";
const DEFAULT_MANIFEST_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/test_data/dsv32_vllm_logits_ref/manifest.json"
);

#[derive(Debug, Deserialize)]
struct VllmManifest {
    schema_version: String,
    cases: Vec<VllmCase>,
}

#[derive(Debug, Deserialize)]
struct VllmCase {
    name: String,
    prompt: String,
    token_ids: Vec<u32>,
    #[serde(default)]
    positions: Vec<i32>,
    generated_token_id: usize,
    top10_ids: Vec<usize>,
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

fn topk_indices(logits: &[bf16], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(idx, v)| (idx, v.to_f32()))
        .collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
    indexed.into_iter().take(k).map(|(idx, _)| idx).collect()
}

fn rank_of_token(logits: &[bf16], token_id: usize) -> usize {
    let target = logits[token_id].to_f32();
    let higher = logits.iter().filter(|v| v.to_f32() > target).count();
    higher + 1
}

#[test]
#[ignore = "requires 8 GPUs and DeepSeek-V3.2 weights; run manually on H20"]
fn dsv32_forward_full_ep8_vllm() {
    let model_path = std::env::var("PEGAINFER_DSV32_MODEL_PATH")
        .unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_string());
    let model_path_buf = PathBuf::from(&model_path);
    if !model_path_buf.exists() {
        eprintln!(
            "Skipped: model path not found. Set PEGAINFER_DSV32_MODEL_PATH (current: {})",
            model_path
        );
        return;
    }

    let manifest_path = std::env::var("PEGAINFER_DSV32_REF_MANIFEST")
        .unwrap_or_else(|_| DEFAULT_MANIFEST_PATH.to_string());
    let manifest_path_buf = PathBuf::from(&manifest_path);
    if !manifest_path_buf.exists() {
        eprintln!(
            "Skipped: manifest not found. Generate it first (current: {})",
            manifest_path
        );
        return;
    }

    let manifest_content = std::fs::read_to_string(&manifest_path_buf).unwrap_or_else(|e| {
        panic!(
            "Failed to read manifest {}: {e}",
            manifest_path_buf.display()
        )
    });
    let manifest: VllmManifest =
        serde_json::from_str(&manifest_content).expect("Failed to parse manifest JSON");
    assert_eq!(
        manifest.schema_version, "dsv32_vllm_logits_ref.v1",
        "unexpected manifest schema version"
    );
    assert!(
        !manifest.cases.is_empty(),
        "manifest contains no cases: {}",
        manifest_path_buf.display()
    );
    let case_filter = parse_case_filter();
    let max_cases = parse_max_cases();
    let selected_cases: Vec<&VllmCase> = manifest
        .cases
        .iter()
        .filter(|case| {
            case_filter
                .as_ref()
                .map(|f| case.name.contains(f))
                .unwrap_or(true)
        })
        .take(max_cases.unwrap_or(usize::MAX))
        .collect();
    assert!(
        !selected_cases.is_empty(),
        "no cases matched filter {:?} in manifest {}",
        case_filter,
        manifest_path_buf.display()
    );
    eprintln!(
        "Selected cases: {} (filter={:?}, max_cases={:?})",
        selected_cases.len(),
        case_filter,
        max_cases
    );

    let device_ordinals = parse_device_ordinals();
    assert_eq!(
        device_ordinals.len(),
        8,
        "dsv32_forward_full_ep8_vllm expects 8 device ordinals, got {:?}",
        device_ordinals
    );

    let tp_size = parse_tp_size();
    assert_eq!(
        device_ordinals.len() % tp_size,
        0,
        "invalid parallel config: world_size={} tp_size={}",
        device_ordinals.len(),
        tp_size
    );

    let load_start = Instant::now();
    let executor = DsV32Executor::load(&model_path, &device_ordinals, tp_size)
        .unwrap_or_else(|e| panic!("Failed to load DsV32Executor (tp_size={tp_size}): {e}"));
    eprintln!(
        "Loaded DsV32Executor: world_size={}, elapsed={:.2}s",
        executor.world_size(),
        load_start.elapsed().as_secs_f64()
    );

    let mut all_passed = true;

    for case in selected_cases {
        let positions: Vec<i32> = if case.positions.is_empty() {
            (0..case.token_ids.len() as i32).collect()
        } else {
            case.positions.clone()
        };
        assert_eq!(
            positions.len(),
            case.token_ids.len(),
            "positions length mismatch for case {}",
            case.name
        );
        assert!(
            case.top10_ids.len() >= 5,
            "case {} has invalid top10_ids (len={})",
            case.name,
            case.top10_ids.len()
        );

        let start = Instant::now();
        let logits = executor
            .forward(&case.token_ids, &positions)
            .unwrap_or_else(|e| panic!("forward failed for case {}: {e}", case.name));
        let elapsed = start.elapsed();

        let our_top10 = topk_indices(&logits, 10);
        assert!(
            !our_top10.is_empty(),
            "empty top10 from logits for case {}",
            case.name
        );
        let our_top1 = our_top10[0];
        let our_top5: HashSet<usize> = our_top10.iter().take(5).copied().collect();
        let ref_top5: HashSet<usize> = case.top10_ids.iter().take(5).copied().collect();
        let our_top10_set: HashSet<usize> = our_top10.iter().copied().collect();
        let ref_top10_set: HashSet<usize> = case.top10_ids.iter().copied().collect();
        let overlap = our_top5.intersection(&ref_top5).count();
        let overlap10 = our_top10_set.intersection(&ref_top10_set).count();
        let ref_top1_rank_in_ours = rank_of_token(&logits, case.generated_token_id);
        let our_top1_in_ref_top10 = ref_top10_set.contains(&our_top1);

        eprintln!("========================================");
        eprintln!("Case: {}", case.name);
        eprintln!("Prompt: {:?}", case.prompt);
        eprintln!("Prompt tokens: {}", case.token_ids.len());
        eprintln!("Forward elapsed: {:.2}s", elapsed.as_secs_f64());
        eprintln!("Our top5: {:?}", &our_top10[..5]);
        eprintln!("Our top10: {:?}", &our_top10);
        eprintln!("Ref top5: {:?}", &case.top10_ids[..5]);
        eprintln!("Ref top10: {:?}", &case.top10_ids);
        eprintln!(
            "Top1: ours={} ref={} {}",
            our_top1,
            case.generated_token_id,
            if our_top1 == case.generated_token_id {
                "MATCH"
            } else {
                "MISMATCH"
            }
        );
        eprintln!("Top5 overlap: {overlap}/5");
        eprintln!("Top10 overlap: {overlap10}/10");
        eprintln!(
            "Ref top1 rank in our logits: {} (our_top1_in_ref_top10={})",
            ref_top1_rank_in_ours, our_top1_in_ref_top10
        );

        if our_top1 != case.generated_token_id {
            all_passed = false;
        }
        if overlap < 3 {
            all_passed = false;
        }
    }

    assert!(all_passed, "Some DSV3.2 vLLM alignment cases failed");
}

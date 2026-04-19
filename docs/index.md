# docs index

| Path | TL;DR |
| --- | --- |
| `projects/dsv32-inference.md` | DeepSeek-V3.2-0324 671B on 8xH20-3e: Phase 2d combine timeout 已修，61 层 forward 跑通。Next: logits 对齐 |
| `archives/batch-optimization.md` | Archived optimization record for the single-GPU batching/perf push: pegainfer reached within 2% of vLLM throughput while beating it on TTFT, TPOT, and latency stability, with dynamic KV cache landed |
| `archives/continuous-batching.md` | Archived implementation record for continuous batching: page allocator, paged KV direction, scheduler, and batch decode infrastructure landed and now serve as historical rollout context |
| `archives/q2-2026-plan.md` | Archived Q2 2026 planning snapshot for batching hardening, PegaFlow integration, and differentiation work; preserved as period roadmap context |
| `archives/nonstandard-attention-milestone.md` | Archived milestone-direction document for pegainfer's non-standard-attention focus, kept as historical framing for model-family readiness and service priorities |
| `archives/qwen3-tp-design.md` | Archived design record for the `Qwen3-4B` tensor-parallel milestone and the first-pass controller-plus-workers runtime direction |
| `archives/flashinfer-sampling-benchmark.md` | Archived blocked workstream for FlashInfer sampling: greedy FlashInfer token selection landed, but batched decode sampling remained unfinished |
| `archives/qwen35-4b-accuracy.md` | Archived Qwen3.5-4B HF parity record: major decode-state bugs were fixed and exact-match coverage improved to 11/13, but full parity remained incomplete |
| `archives/qwen35-4b-optimization.md` | Archived Qwen3.5-4B optimization record: near-parity with vLLM was reached and the post-refactor decode regression was recovered |
| `archives/model-forward-trait.md` | Archived architecture record for the `ModelForward` trait extraction and shared generation-loop design |
| `archives/runtime-complexity-paydown.md` | Archived runtime-paydown record covering the phase where runtime branching cleanup and boundary tightening were tracked as a dedicated effort |
| `archives/accuracy-eval-results.md` | Archived Phase 1 GSM8K evaluation snapshot with one passing Qwen3-4B comparison and one failing Qwen3.5-4B comparison |
| `archives/pure-gpu-decode-loop.md` | Concluded: CPU overhead is ~0.6% of TPOT (~77μs/token). Batch launch saves ~1ms/128tok. Not worth further investment — TPOT is GPU-compute bound |
| `archives/qwen3-4b-optimization.md` | Dense-attention Qwen3-4B optimization record; archived as reference material after pegainfer led the measured RTX 5070 Ti workloads |
| `archives/qwen35-gdr-chunkwise-plan.md` | Qwen3.5 chunk-wise GDR plan and validation history; archived after the plan landed in the real runtime and rolled into the broader Qwen3.5 optimization record |
| `areas/bench-regression.md` | Benchmark regression tracking: one snapshot per model, git-tracked history, TPOT >2% / TTFT >3% thresholds |
| `areas/dsv32.md` | DeepSeek-V3.2 runtime map + 当前验证状态：提交版精度回归收敛到 small greedy JSON E2E；SGLang teacher-forced top-K harness 保留为本地/H20 深度回归 |
| `resources/accuracy-parity-playbook.md` | Accuracy debugging playbook: truth-source rules, first-diff workflow, bf16 rounding traps, and verified Qwen3.5 parity commands |
| `resources/developer-onboarding.md` | New-developer onboarding — toolchain, unified venv, build, tests, benchmark smoke test |
| `resources/profiling-guide.md` | GPU profiling playbook: nsys pitfalls, diagnostic paths, measured kernel comparisons |
| `resources/bench-vs-vllm.md` | pegainfer vs vLLM comparative benchmarking: method, workflow, typical configs, gotchas |
| `resources/model-optimization-pipeline.md` | Per-model optimization methodology: 2 standard profiles, vLLM baseline, e2e dashboard + append-only optimization log |
| `resources/kernel-technology-reference.md` | Kernel tech reference: current stack, ecosystem survey (Triton/Gluon/CUTLASS/ThunderKittens/FlashAttention/FlashInfer), decision framework, source-level lessons, and operator policy |
| `resources/flashinfer-reference.md` | FlashInfer map: official docs structure, operator families, major features, and which source areas matter beyond the docs index |
| `areas/coding-style.md` | Testing principle: prefer integration tests, don't test what E2E catches |

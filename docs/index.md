# docs index

Organized by domain (model line / subsystem / playbook / lesson) instead of by lifecycle stage. A doc's freshness is recorded in its own header (TL;DR / Status), not by which directory it lives in.

| Where it lives | What it is |
| --- | --- |
| `roadmap/` | Strategic plans and milestones — quarterly direction, product positioning. |
| `models/<line>/` | Per-model living docs: design, accuracy, perf, refactor records, gotchas. |
| `subsystems/<area>/` | Cross-cutting components (runtime / scheduler / frontend / kernels). |
| `playbooks/` | Reusable how-to: benching, profiling, accuracy debugging, onboarding. |
| `lessons/` | Tribal knowledge from research / other projects worth keeping. |
| `benchmarks/` | Standalone benchmark snapshots and eval reports. |
| `conventions/` | Ongoing standards (bench regression policy, coding style). |
| `archives/` | Concluded work kept for reference only — no longer updated. |
| `private/` | Local-only notes (gitignored). |

## roadmap

| Path | TL;DR |
| --- | --- |
| `roadmap/q2-2026-plan.md` | Q2 plan: W1 harden batching, W2 PegaInfer+PegaFlow native, W3 differentiation. Competitive intel: Qwen3.5 is both competitors' Achilles' heel, startup time 215s vs seconds, observability as product moat. MTP deferred to Q3. |
| `roadmap/nonstandard-attention-milestone.md` | Milestone direction: pegainfer focuses on non-standard attention models, with emphasis on model-family readiness, service experience, framework debt repayment, and disciplined evaluation. |

## models / qwen3

| Path | TL;DR |
| --- | --- |
| `models/qwen3/model-crate.md` | `pegainfer-qwen3-4b` owns Qwen3 config/weights/executor/scheduler/tests/kernel plan; root sees generic `EngineHandle`; split-K retuned to `256/64`, with 4k/64 serving TPOT p50 at `6.46ms` on RTX 5090. |
| `models/qwen3/kernels-crate.md` | Phase 1 split implemented and 5090-verified: Qwen3-4B kernel surface lives in `pegainfer-kernels`; release build, test-target compile, Qwen3 e2e, and bench snapshot pass. |
| `models/qwen3/tp-design.md` | Qwen3 tensor-parallel design: `TP=2` milestone scope plus the controller/worker broadcast execution model, request identity, and coarse-grained step protocol for future TP/MoE work. |
| `models/qwen3/kv-pressure-hang.md` | Issue #85 Qwen3-4B KV pressure hang fixed by full-lifetime scheduler KV admission, waiting-queue deferral, cleanup on disconnect/error, impossible-request errors, scheduler/bridge gates, and real `vllm bench serve` QPS=2 `500/500` pass with post-pressure completion healthy. |

## models / qwen35

| Path | TL;DR |
| --- | --- |
| `models/qwen35/optimization.md` | Hybrid 24 linear + 8 full attn. At parity with vLLM: TTFT 225ms, TPOT 11.81ms (+1%). Post-accuracy-fix GDR decode kernel restore (#9). |
| `models/qwen35/accuracy.md` | Qwen3.5-4B HF parity work: major decode-state bugs are fixed, `conv1d` now matches HF's bf16 pre-`SiLU` rounding, exact HF matches improved to 11/13, and only two small-logit-drift cases remain. |
| `models/qwen35/model-crate.md` | `pegainfer-qwen35-4b` owns Qwen3.5 model/scheduler/recurrent ops/tests/benches; root loads it through `EngineHandle`. Build/check/clippy, root bench sanity check, Qwen3.5 e2e, and scheduler e2e pass. |
| `models/qwen35/e2e-gibberish.md` | Qwen3.5 e2e gibberish fixed: scheduler threads now bind CUDA context and initialize thread-local cuBLAS handles; Qwen3.5 greedy stays on FlashInfer top1, and the default e2e remains an exact golden-text regression. |

## models / deepseek-v4

| Path | TL;DR |
| --- | --- |
| `models/deepseek-v4/support.md` | Initial DeepSeek V4 support PR record: native MP8 engine, official-style TileLang build-time kernels, exact E2E, HTTP validation, nsys-guided speed fixes, prefill RoPE reuse, sync removal, scratch reuse, and GPU index generation. |
| `models/deepseek-v4/decode-performance.md` | Fixed long decode is retained sub-30 with exact E2E `20/20` and hash `6346f03343d75a65`; stable sub-25 remains open. |
| `models/deepseek-v4/serving-baseline.md` | Serving baseline gate: HTTP single-request smoke and direct TPOT/hash regression available; bs>1 serving, continuous batching, and service-level KV management remain follow-up. |
| `models/deepseek-v4/http-serving-benchmark.md` | HTTP serving benchmark gate: streaming `/v1/completions` load records QPS, TTFT, TPOT/ITL, latency percentiles, error rate, and output hashes without using direct bench as serving evidence. |
| `models/deepseek-v4/prefix-paged-kv-pd-handoff.md` | Prefix/paged KV and P-D handoff design contract: evolves slot-owned direct KV leases into page ownership, prefix cache, allocator telemetry, and transport-agnostic handoff handles. |
| `models/deepseek-v4/moe-ag-rs.md` | Decode MoE now uses GPU AG/RS, GPU route compaction, and grouped TileLang FP4 local experts; no route/expert D2H in hot path. Current 1x32 TPOT avg `105.54ms`, exact E2E `20/20`. |
| `models/deepseek-v4/moe-tilelang-review.md` | Persistent rank workers + decode-only direct top-k MoE cut 1x32 steady TPOT to `80.49ms/token`; remaining cost is rank arrival skew before `107` f32 collectives/token. |
| `models/deepseek-v4/kernel-paths.md` | DeepSeek V4 CUDA sources, TileLang generator path, and `pegainfer-kernels/KERNELS.md` routing index are organized. |

## subsystems / runtime

| Path | TL;DR |
| --- | --- |
| `subsystems/runtime/core-entry-crate.md` | `pegainfer-core` owns shared runtime/API pieces, root keeps compatibility re-exports, trace is out of the active entry, and full 5090 build/clippy/e2e/bench passes. |
| `subsystems/runtime/virtual-workspace-top-level-crates.md` | Root is a virtual workspace; product code lives in `pegainfer-server/`, internal packages live at top-level `pegainfer-*` paths. |
| `subsystems/runtime/model-forward-trait.md` | ModelForward trait extraction: weights/state separation, shared generation loop, designed for bs > 1. |
| `subsystems/runtime/runtime-complexity-paydown.md` | Effort to reduce model-specific runtime fragmentation; focus shifted to architecture-level abstraction (ModelForward trait). |

## subsystems / scheduler

| Path | TL;DR |
| --- | --- |
| `subsystems/scheduler/continuous-batching.md` | Phase 1-2 done. Scheduler thread with prefill-priority, batch decode, channel-based streaming. Next: multi-request throughput testing. |
| `subsystems/scheduler/batch-optimization.md` | Realistic benchmark: within 2% of vLLM throughput, TTFT −16%, TPOT −1.6%. Decode TPOT beats vLLM at all concurrencies. Dynamic KV cache (85% free VRAM). Remaining: ITL p99 tail (chunked prefill). |

## subsystems / frontend

| Path | TL;DR |
| --- | --- |
| `subsystems/frontend/vllm-frontend-rs-integration.md` | vLLM frontend replacement complete: pegainfer serves through vllm-server with a local engine-core bridge; old HTTP/tokenizer modules are removed; Qwen3 release build/e2e and vLLM models/completions/chat validation pass on 5090. |

## subsystems / kernels

| Path | TL;DR |
| --- | --- |
| `subsystems/kernels/pegainfer-kernels-boundary.md` | Architecture decision: pegainfer should use reusable frontend/runtime/data-plane layers plus per-model engines; kernels become first-class assets through a ledger, simulator, and request tracing. |
| `subsystems/kernels/kernel-op-reports.md` | Qwen3 kernel report moved to a feature-gated bin; full raw-CUPTI decode/prefill reports, split prefill stages, 10k attention-core comparisons, BF16-HMMA tensor metrics, FlashInfer FMHA v2/package measurements, and measured FA2 `CTA_TILE_Q=64` prefill default in place. |
| `subsystems/kernels/flashinfer-sampling-benchmark.md` | FlashInfer runtime sampling and greedy top-1 path are in place, but the work is blocked on restoring batched decode token selection instead of per-request sampling inside batch decode. |

## playbooks

| Path | TL;DR |
| --- | --- |
| `playbooks/developer-onboarding.md` | New-developer onboarding — toolchain, unified venv, build, tests, quick benchmark validation. |
| `playbooks/bench-vs-vllm.md` | pegainfer vs vLLM comparative benchmarking: method, workflow, typical configs, gotchas. |
| `playbooks/model-optimization-pipeline.md` | Per-model optimization methodology: 2 standard profiles, vLLM baseline, e2e dashboard + append-only optimization log. |
| `playbooks/profiling-guide.md` | GPU profiling playbook: nsys pitfalls, diagnostic paths, measured kernel comparisons. |
| `playbooks/cupti-range-profiler.md` | CUPTI Range Profiler notes: short range names required on RTX 5090/CUDA 12.9, `qwen3_kernel_report` CUPTI default-on, pre-measure launch avoids lazy-init counter pollution. |
| `playbooks/accuracy-parity-playbook.md` | Accuracy debugging playbook: truth-source rules, first-diff workflow, bf16 rounding traps, and verified Qwen3.5 parity commands. |
| `playbooks/kernel-technology-reference.md` | Kernel tech reference: current stack, ecosystem survey (Triton/Gluon/CUTLASS/ThunderKittens/FlashAttention/FlashInfer), decision framework, source-level lessons, and operator policy. |
| `playbooks/flashinfer-reference.md` | FlashInfer map: official docs structure, operator families, major features, and which source areas matter beyond the docs index. |

## lessons

| Path | TL;DR |
| --- | --- |
| `lessons/moe-dplb-decode-imbalance.md` | DPLB lesson for future PegaFlow/WiDeep MoE+EP serving: decode-side DP imbalance is a sticky KV-state problem; engines should emit raw progress while external router/proxy derive load and routing. |
| `lessons/moe-zero-prefill-long-prefill.md` | ZeRO-Prefill lesson for future long-prefill MoE serving: once a router selects long-P work, maximize batch throughput by preserving compute-bound execution, hiding expert-weight movement, respecting KV handoff boundaries, and measuring bottlenecks before committing to an AsyncEP-style backend. |

## benchmarks

| Path | TL;DR |
| --- | --- |
| `benchmarks/bs1-4k64-vllm-pegainfer.md` | RTX 5090 single-concurrency probe: `input_len=4096`, `output_len=64`, no vLLM prefix cache. PegaInfer TTFT median `177ms` vs vLLM `198ms`; TPOT median `6.47ms` vs `6.36ms`; corrected output throughput `+6%` for PegaInfer. |
| `benchmarks/accuracy-eval-results.md` | Phase 1 GSM8K: Qwen3-4B PASS (pegainfer 85.37% vs HF 85.82%, delta -0.45%). Qwen3.5-4B FAIL — long-prompt prefill quality divergence on 8-shot. |

## conventions

| Path | TL;DR |
| --- | --- |
| `conventions/bench-regression.md` | Benchmark regression tracking: one snapshot per model, git-tracked history, TPOT >2% / TTFT >3% thresholds. |
| `conventions/coding-style.md` | Testing principle: prefer integration tests, don't test what E2E catches. |

## archives

| Path | TL;DR |
| --- | --- |
| `archives/pure-gpu-decode-loop.md` | Concluded: CPU overhead is ~0.6% of TPOT (~77μs/token). Batch launch saves ~1ms/128tok. Not worth further investment — TPOT is GPU-compute bound. |
| `archives/qwen3-4b-optimization.md` | Dense-attention Qwen3-4B optimization record; archived as reference material after pegainfer led the measured RTX 5070 Ti workloads. |
| `archives/qwen35-gdr-chunkwise-plan.md` | Qwen3.5 chunk-wise GDR plan and validation history; archived after the plan landed in the real runtime and rolled into the broader Qwen3.5 optimization record. |

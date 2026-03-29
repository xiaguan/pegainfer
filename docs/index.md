# docs index

| Path | TL;DR |
| --- | --- |
| `projects/batch-optimization.md` | Decode batching works (5.9× at c=8). Prefill is serial — TTFT 7× at c=8, ITL p99 spikes 60–80ms. vLLM batches prefill and holds flat. Two targets: prefill batching, per-token latency under batch |
| `projects/continuous-batching.md` | Phase 1-2 done. Scheduler thread with prefill-priority, batch decode, channel-based streaming. Next: multi-request throughput testing |
| `projects/q2-2026-plan.md` | Q2 plan: W1 harden batching, W2 PegaInfer+PegaFlow native, W3 differentiation. Competitive intel: Qwen3.5 is both competitors' Achilles' heel, startup time 215s vs seconds, observability as product moat. MTP deferred to Q3 |
| `projects/nonstandard-attention-milestone.md` | Milestone direction: pegainfer focuses on non-standard attention models, with emphasis on model-family readiness, service experience, framework debt repayment, and disciplined evaluation |
| `projects/qwen35-4b-accuracy.md` | Qwen3.5-4B HF parity work: major decode-state bugs are fixed, `conv1d` now matches HF's bf16 pre-`SiLU` rounding, exact HF matches improved to 11/13, and only two small-logit-drift cases remain |
| `projects/qwen35-4b-optimization.md` | Hybrid 24 linear + 8 full attn. At parity with vLLM: TTFT 225ms, TPOT 11.81ms (+1%). Post-accuracy-fix GDR decode kernel restore (#9) |
| `projects/model-forward-trait.md` | ModelForward trait extraction: weights/state separation, shared generation loop, designed for bs > 1 |
| `projects/runtime-complexity-paydown.md` | Project to reduce model-specific runtime fragmentation; focus shifting to architecture-level abstraction (ModelForward trait) |
| `archives/pure-gpu-decode-loop.md` | Concluded: CPU overhead is ~0.6% of TPOT (~77μs/token). Batch launch saves ~1ms/128tok. Not worth further investment — TPOT is GPU-compute bound |
| `archives/qwen3-4b-optimization.md` | Dense-attention Qwen3-4B optimization record; archived as reference material after pegainfer led the measured RTX 5070 Ti workloads |
| `archives/qwen35-gdr-chunkwise-plan.md` | Qwen3.5 chunk-wise GDR plan and validation history; archived after the plan landed in the real runtime and rolled into the broader Qwen3.5 optimization record |
| `resources/accuracy-parity-playbook.md` | Accuracy debugging playbook: truth-source rules, first-diff workflow, bf16 rounding traps, and verified Qwen3.5 parity commands |
| `resources/developer-onboarding.md` | New-developer onboarding — toolchain, unified venv, build, tests, benchmark smoke test |
| `resources/profiling-guide.md` | GPU profiling playbook: nsys pitfalls, diagnostic paths, measured kernel comparisons |
| `resources/bench-vs-vllm.md` | pegainfer vs vLLM comparative benchmarking: method, workflow, typical configs, gotchas |
| `resources/model-optimization-pipeline.md` | Per-model optimization methodology: 2 standard profiles, vLLM baseline, e2e dashboard + append-only optimization log |
| `resources/kernel-technology-reference.md` | Kernel tech reference: current stack, ecosystem survey (Triton/Gluon/CUTLASS/ThunderKittens/FlashAttention/FlashInfer), decision framework, source-level lessons, and operator policy |

# docs index

| Path | TL;DR |
| --- | --- |
| `projects/landing.md` | New-developer onboarding — toolchain, unified venv, build, tests, benchmark smoke test |
| `projects/qwen3-4b-optimization.md` | Leads vLLM on 4 workloads (single-concurrency, RTX 5070 Ti). Needs broader stress testing |
| `projects/qwen35-gdr-chunkwise-plan.md` | Chunk-wise GDR prefill plan for Qwen3.5: Rust path is live, root cause was v_new write order, e2e passes, and the refreshed 6/13-drift baseline is accepted |
| `projects/qwen35-4b-optimization.md` | Hybrid 24 linear + 8 full attn. Chunk-wise Rust GDR prefill now reaches ~222ms TTFT(2048,1), passes e2e_qwen35, and ships with an accepted refreshed JSON baseline |
| `resources/profiling-guide.md` | GPU profiling playbook: nsys pitfalls, diagnostic paths, measured kernel comparisons |
| `resources/bench-vs-vllm.md` | pegainfer vs vLLM comparative benchmarking: method, workflow, typical configs, gotchas |
| `resources/model-optimization-pipeline.md` | Per-model optimization methodology: 2 standard profiles, vLLM baseline, e2e dashboard + append-only optimization log |

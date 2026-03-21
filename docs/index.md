# docs index

| Path | TL;DR |
| --- | --- |
| `projects/landing.md` | New-developer onboarding — toolchain, unified venv, build, tests, benchmark smoke test |
| `projects/qwen3-4b-optimization.md` | Leads vLLM on 4 workloads (single-concurrency, RTX 5070 Ti). Needs broader stress testing |
| `projects/qwen35-4b-optimization.md` | Hybrid 24 linear + 8 full attn. Decode 12.55ms TPOT (OK). Prefill 16.8s/2048tok (79× behind Qwen3-4B) — 67% CPU launch overhead |
| `resources/profiling-guide.md` | GPU profiling playbook: nsys pitfalls, diagnostic paths, measured kernel comparisons |
| `resources/bench-vs-vllm.md` | pegainfer vs vLLM comparative benchmarking: method, workflow, typical configs, gotchas |
| `resources/model-optimization-pipeline.md` | Per-model optimization methodology: 2 standard profiles, vLLM baseline, e2e dashboard + append-only optimization log |

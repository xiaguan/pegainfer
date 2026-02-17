# Production-Ready TODO

Pure Rust + CUDA, no Python runtime dependency.

## Must-Have (功能完整)

- [ ] **Chat Template** — 解析 `tokenizer_config.json` 的 chat_template，拼 `<|im_start|>system/user/assistant` 格式。用 `minijinja` crate 或手拼。没这个无法暴露 `/v1/chat/completions` API
- [ ] **Prefill GEMM** — prompt tokens 用 cuBLAS GEMM batch forward 一次算完，替代当前逐 token GEMV。TTFT 瓶颈，prompt=512 时差 ~50×
- [ ] **SSE Streaming** — axum `Sse<impl Stream>` 逐 token yield，前端/SDK 集成硬性要求
- [ ] **Stop Conditions** — 支持 `stop` 字符串列表 + `max_tokens` 硬截断。Qwen3 有 `<|endoftext|>` 和 `<|im_end|>` 两个 stop token
- [ ] **Sampling** — temperature / top_p / top_k（已在 plan 中）

## Should-Have (工程健壮性)

- [ ] **Request Validation** — 校验 `prompt_tokens + max_tokens <= max_seq_len`，超限返回 400
- [ ] **KV Cache Bounds Check** — prompt + generation > 4096 时防止越界写 GPU 内存
- [ ] **Graceful Startup / Shutdown** — 替换 `.expect()` / `.unwrap()`，加 `/health` endpoint，`tokio::signal` graceful shutdown
- [ ] **OOM 防护** — 加载前估算 weights + KV cache + intermediates 总需求，与 `cudaMemGetInfo` 比对

## Nice-to-Have (性能)

- [ ] **Multi-arch Build** — `build.rs` 加 `-gencode arch=compute_89,code=sm_89`（RTX 4090）+ `sm_120`
- [ ] **Weight Loading 优化** — mmap + 分 tensor 逐个 H2D，降低 CPU 内存峰值（当前 ~15.6GB）
- [ ] **CUDA Stream Overlap** — decode 阶段 2 streams：当前 token D2H argmax 与下一 token embedding lookup 重叠

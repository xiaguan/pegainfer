# DSV3.2 vLLM 精度对齐 IT

**Created**: 2026-04-17
**Status**: paused

## 目标

为 DeepSeek-V3.2 建立一条可重复执行的 8 卡端到端精度回归链路：

- 用 vLLM 产出 ground truth
- 用 pegainfer `DsV32Executor::forward` 做全链路对齐
- 将回归入口固化为一条 `#[ignore]` 的集成测试

## 保留资产

- 集成测试：[e2e_dsv32_vllm.rs](/data/code/workspace-rustllm/pegainfer/tests/e2e_dsv32_vllm.rs)
- truth 生成脚本：[gen_logits_ref.py](/data/code/workspace-rustllm/pegainfer/tools/dsv32_vllm_ref/gen_logits_ref.py)
- prompt 集：[prompts.json](/data/code/workspace-rustllm/pegainfer/tools/dsv32_vllm_ref/prompts.json)
- truth README：[README.md](/data/code/workspace-rustllm/pegainfer/tools/dsv32_vllm_ref/README.md)
- reference 资产目录：`test_data/dsv32_vllm_logits_ref/`

## 当前运行方式

生成 truth：

```bash
cd /data/code/workspace-rustllm/pegainfer
.venv/bin/python tools/dsv32_vllm_ref/gen_logits_ref.py \
  --model-path /data/models/DeepSeek-V3.2 \
  --output-dir test_data/dsv32_vllm_logits_ref \
  --prompts-file tools/dsv32_vllm_ref/prompts.json \
  --tensor-parallel-size 8
```

运行 IT：

```bash
PEGAINFER_DSV32_MODEL_PATH=/data/models/DeepSeek-V3.2 \
cargo test -r dsv32_forward_full_ep8_vllm -- --ignored --nocapture
```

H20 上额外需要：

```bash
export PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer-workspace/.venv/bin/python
```

## 本轮沉淀下来的结论

### 已验证修复

以下修复已经保留在主代码里：

1. `kv_state.seq_len` 改为“每个 token 只 advance 一次”，不再在每层重复累加。
2. `DeepEP notify_dispatch` 的 `num_memset_int` 对齐 upstream 语义。
3. `deep_ep_notify_dispatch` 末参统一命名为 `num_channels`。
4. DeepEP scratch buffer 不再把 `num_ranks` / `num_channels` 写死。
5. FlashMLA `num_sm_parts` 改为按实际 GPU `SM count` 动态计算。
6. `total_num_splits` 与 accum buffer contract 改回 upstream 语义。
7. FP8 / 权重加载补了 shape 和长度校验，避免静默误读。
8. dense decode 路径补了 `kv.seq_len() == positions[req_idx] + 1` 断言。

### 当前 IT 结果

- 当前 `dsv32_forward_full_ep8_vllm` 能稳定跑通。
- 5 个 case 里 `top1` 对齐 `2/5`。
- 已命中：`code_prompt`、`long_context_mix`。
- 剩余 case 仍有系统性偏移。

### 当前判断

1. 这条 IT 已经足够承担回归入口的角色。
2. `pegainfer` 的一条真实 decode 路径，在 `layer1` 上已能与 pure checkpoint reference 近乎完全对齐。
3. 因此，之前基于 SGLang 中间态得到的分层差异，不能直接当作 pegainfer correctness 结论。
4. 精度问题可以先暂停，当前更值得推进的是 DSV3.2 的生成循环与服务化接入。

## 已清理内容

本轮已经从工作树中移除：

- 临时 hidden dump 二进制入口
- `forward_capture_hidden` 相关抓取代码
- SGLang monkeypatch / sitecustomize 调试链
- 一批分层对比脚本
- 隐藏态中间产物目录
- 集成测试里只用于诊断的 `last-token-only` 探针

## 下一主线

DSV3.2 目前仍停留在 executor 级能力：

- 能 `load`
- 能 `forward`
- 还不能直接走 pegainfer 的 HTTP server / scheduler / `bench_serving`

下一阶段工作应切到生成与服务化：

1. 增加 `ModelType::DSV32`
2. 接 `scheduler_dsv32`
3. 接生成循环与采样
4. 接入 [main.rs](/data/code/workspace-rustllm/pegainfer/src/main.rs) 和 [bench_serving.rs](/data/code/workspace-rustllm/pegainfer/src/bin/bench_serving.rs)

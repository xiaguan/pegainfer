# Rust LLM - Qwen3 0.6B 推理引擎

从零手写的 LLM 推理引擎，纯 Rust 实现，不依赖任何深度学习框架。

## 项目结构

```
src/
├── main.rs        # HTTP server + /v1/completions API
├── config.rs      # 模型配置解析
├── tokenizer.rs   # Tokenizer 封装（基于 HuggingFace tokenizers）
├── gpu_model.rs   # GPU 版 Qwen3 模型结构 + KV Cache + 生成逻辑
├── gpu.rs         # GPU 算子：RMSNorm, SiLU, Softmax, Linear, RoPE, Attention
├── ffi.rs         # CUDA kernel FFI 绑定
└── tensor.rs      # Safetensors 加载，BF16 → F32 转换

csrc/
└── kernels.cu     # CUDA kernels（矩阵乘法、attention、激活函数等）
```

## 依赖

- Rust 2024 edition
- CUDA Toolkit (nvcc)
- C++ 编译器

## 编译

```bash
# 设置 CUDA 路径
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda

# 编译
cargo build --release
```

## 运行

```bash
# 设置运行时库路径
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 启动服务
cargo run --release
```

服务监听 `http://0.0.0.0:8000`。

## API

兼容 OpenAI `/v1/completions` 接口：

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world", "max_tokens": 20}'
```

响应：
```json
{
  "id": "cmpl-xxx",
  "object": "text_completion",
  "model": "qwen3-0.6b",
  "choices": [{"text": "...", "finish_reason": "length"}],
  "usage": {"prompt_tokens": 2, "completion_tokens": 20, "total_tokens": 22}
}
```

## 推理流水线详解

### 1. 输入处理

```
"Hello" → Tokenizer → [9707]
```

Tokenizer 将文本转换为 token IDs。Qwen3 使用 BPE (Byte Pair Encoding) 分词。

### 2. Embedding 查表

```
token_id: 9707 → embed_tokens[9707] → hidden_state (1024,)
```

从 `embed_tokens` 矩阵 (vocab_size=151936, hidden_size=1024) 中查找对应行，得到初始隐藏状态。

### 3. Transformer 层 (×28)

每一层的计算流程：

```
输入: hidden_state (1024,)
      ↓
┌─────────────────────────────────────────────────────────────┐
│  Input LayerNorm (RMSNorm)                                  │
│  normed = x * weight / sqrt(mean(x²) + eps)                 │
└─────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────┐
│  Self-Attention (Grouped Query Attention)                   │
│                                                             │
│  Q = normed @ q_proj.T    → (2048,) = 16 heads × 128 dim   │
│  K = normed @ k_proj.T    → (1024,) = 8 heads × 128 dim    │
│  V = normed @ v_proj.T    → (1024,) = 8 heads × 128 dim    │
│                                                             │
│  每个 head:                                                 │
│    q_head = RMSNorm(Q[head*128:(head+1)*128])               │
│    q_head = RoPE(q_head, position)                          │
│    k_head = RMSNorm(K[head*128:(head+1)*128])               │
│    k_head = RoPE(k_head, position)                          │
│                                                             │
│  GQA: 16 个 Q heads 共享 8 个 KV heads (每 2 个 Q 共享 1 KV)│
│                                                             │
│  scores = Q @ K.T / sqrt(head_dim)                          │
│  attn_weights = softmax(scores)                             │
│  attn_output = attn_weights @ V                             │
│                                                             │
│  output = concat(all_heads) @ o_proj.T → (1024,)            │
└─────────────────────────────────────────────────────────────┘
      ↓
      + (残差连接)
      ↓
┌─────────────────────────────────────────────────────────────┐
│  Post-Attention LayerNorm (RMSNorm)                         │
└─────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────┐
│  MLP (SwiGLU 变体)                                          │
│                                                             │
│  gate = normed @ gate_proj.T  → (3072,)                     │
│  up   = normed @ up_proj.T    → (3072,)                     │
│  mlp_out = SiLU(gate) * up                                  │
│  mlp_out = mlp_out @ down_proj.T → (1024,)                  │
│                                                             │
│  SiLU(x) = x * sigmoid(x)                                   │
└─────────────────────────────────────────────────────────────┘
      ↓
      + (残差连接)
      ↓
输出: hidden_state (1024,)
```

### 4. 输出层

```
hidden_state (1024,)
      ↓
┌─────────────────────────────────────────────────────────────┐
│  Final LayerNorm (RMSNorm)                                  │
└─────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────┐
│  LM Head (与 embed_tokens 权重共享)                         │
│  logits = embed_tokens @ hidden_state → (151936,)           │
└─────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────┐
│  采样                                                       │
│  next_token = argmax(logits)  (贪婪解码)                    │
└─────────────────────────────────────────────────────────────┘
      ↓
next_token_id
```

### 5. 自回归生成

```
tokens = [9707]  # "Hello"
      ↓
┌─────────────────────────────────────────────────────────────┐
│  循环生成:                                                   │
│                                                             │
│  1. forward(tokens) → logits                                │
│  2. next_token = argmax(logits)                             │
│  3. tokens.append(next_token)                               │
│  4. 更新 KV Cache (避免重复计算历史 K/V)                     │
│  5. 重复直到 EOS 或达到最大长度                              │
└─────────────────────────────────────────────────────────────┘
      ↓
"Hello Answer\n How to to solve this problem?..."
```

## 核心算子实现

### RMSNorm

```rust
fn rms_norm(x: &Array1<f32>, weight: &Array1<f32>, eps: f32) -> Array1<f32> {
    let mean_sq = x.mapv(|v| v * v).mean().unwrap();
    let rms = (mean_sq + eps).sqrt();
    x / rms * weight
}
```

与 LayerNorm 不同，RMSNorm 不减均值，只除以 RMS，计算更快。

### RoPE (旋转位置编码)

```rust
// 对每对相邻维度应用旋转
result[i*2]   = x[i*2] * cos - x[i*2+1] * sin
result[i*2+1] = x[i*2] * sin + x[i*2+1] * cos

// 频率随维度指数衰减
freq[i] = position / (theta ^ (2i / head_dim))
```

RoPE 通过旋转编码位置信息，使得 Q·K 的点积自然包含相对位置信息。

### SiLU (Sigmoid Linear Unit)

```rust
fn silu(x: f32) -> f32 {
    x * (1.0 / (1.0 + (-x).exp()))  // x * sigmoid(x)
}
```

### Grouped Query Attention (GQA)

```
Q heads: 16 个 (每个 128 维)
K heads: 8 个 (每个 128 维)
V heads: 8 个 (每个 128 维)

分组: Q[0,1] 共享 KV[0], Q[2,3] 共享 KV[1], ...

相比 MHA 减少 KV cache 大小，相比 MQA 保留更多表达能力。
```

## KV Cache

```rust
struct KVCache {
    // [layer][kv_head][position] → Array1<f32>
    k_cache: Vec<Vec<Vec<Array1<f32>>>>,
    v_cache: Vec<Vec<Vec<Array1<f32>>>>,
}
```

自回归生成时，历史 token 的 K/V 不变，缓存起来避免重复计算。

## 模型参数

| 参数 | 值 |
|------|-----|
| hidden_size | 1024 |
| num_hidden_layers | 28 |
| num_attention_heads | 16 |
| num_key_value_heads | 8 |
| head_dim | 128 |
| intermediate_size | 3072 |
| vocab_size | 151936 |
| 总参数量 | 751.63M |

## 运行

```bash
cargo run --release
```

输出示例：
```
=== Rust LLM - Qwen3 0.6B ===

Loading tokenizer...
Vocab size: 151669

Loading model...
Model loaded in 2.84s

Prompt: Hello
Tokens: [9707]

Generating...
Output: Hello Answer
 How to to solve this problem? Let me know the answer??

Generated 20 tokens in 3.59s
```

## 性能

当前（GPU 加速，CUDA）：
- 模型加载：~2.5s
- TTFT (首 token 时间)：~400ms (prompt_len=4)
- TPOT (每 token 时间)：~96ms/token
- 生成速度：**10.4 tokens/sec**

性能分解（单 token 生成）：
- **Embedding**: 0.02ms (可忽略)
- **28层 Transformer**: 93.4ms (97.5%)
  - 每层平均: 3.3ms
- **LM Head + Argmax**: 2.35ms (2.5%)

## 日志与监控

使用 `tracing` 框架实现完整的请求追踪：

```bash
# 启动服务时可看到详细日志
RUST_LOG=info cargo run --release
```

日志输出示例：
```
INFO completions{prompt_len=15 max_tokens=4}: Received request
INFO completions{...}:forward{num_tokens=4}:get_embeddings: enter
INFO completions{...}:forward{num_tokens=4}:get_embeddings: close time.busy=20.9µs
INFO completions{...}:forward{num_tokens=4}:process_all_layers: enter
INFO completions{...}:forward{...}:forward_layer: close time.busy=3.30ms
... (28层)
INFO completions{...}:forward{...}:process_all_layers: close time.busy=93.4ms
INFO completions{...}:forward{...}:predict_next_token: close time.busy=2.35ms
INFO completions{...}: TTFT: 401.96ms (prompt_len=4)
INFO completions{...}: TPOT: 96.59ms/tok (generated 3 tokens, 10.4 tok/s)
```

关键指标：
- **TTFT (Time To First Token)**: 首个 token 生成延迟
- **TPOT (Time Per Output Token)**: 后续 token 平均生成时间
- **Span Events**: 通过 `#[instrument]` 宏自动追踪函数调用链和耗时

## 待优化

- [x] GPU 支持 (已完成)
- [x] 详细日志与性能监控 (已完成)
- [ ] 算子融合 (减少 kernel launch 开销)
- [ ] Flash Attention (优化 attention 计算)
- [ ] 流式输出 (Server-Sent Events)
- [ ] Batch 推理支持
- [ ] PagedAttention (优化 KV Cache 管理)

## 实现进度

**已完成**：
- ✅ 纯 Rust 实现 Qwen3 0.6B 推理
- ✅ CUDA GPU 加速（cuBLAS + 自定义 kernels）
- ✅ OpenAI 兼容 API (`/v1/completions`)
- ✅ KV Cache 优化
- ✅ Grouped Query Attention (GQA)
- ✅ 完整的日志追踪系统 (tracing + span events)
- ✅ TTFT/TPOT 性能监控

# 模型建模指南

本文档提供各模型的建模指导，包括架构差异和代码路径选择。

---

## 模型架构总览

| 模型 | Attention | FFN | MTP | max_seq |
|------|-----------|-----|-----|---------|
| DeepSeek V3.2 | DSA | MoE | ✓ | 128K |
| Kimi K2.5 | MLA | MoE | ✗ | 256K |
| Qwen 2.5 | GQA | Dense | ✗ | 128K |
| MiniMax M2.5 | GQA | MoE | ✓ | 192K |
| Qwen 3.5 | Linear + GQA | MoE | ✓ | 256K |

---

## 模型选择逻辑

```python
if attention_type == 'hybrid':
    inference_model = DecodeQwen35(...)
elif attention_type in ('mla', 'dsa'):
    inference_model = DecodeDeepSeekV32(...)
elif attention_type == 'gqa':
    if model.is_moe:
        inference_model = DecodeMiniMaxM25(...)
    else:
        inference_model = DecodeQwen2_5(...)
```

---

## 1. DeepSeek V3.2 (DSA)

### 关键参数

```json
{
  "hidden_size": 7168,
  "num_hidden_layers": 61,
  "num_attention_heads": 128,
  "num_experts": 256,
  "num_experts_per_tok": 8,
  "num_shared_experts": 1,
  "moe_intermediate_size": 2048,
  "kv_lora_rank": 512,
  "qk_nope_head_dim": 128,
  "qk_rope_head_dim": 64,
  "v_head_dim": 128,
  "attention_type": "dsa",
  "index_topk": 256,
  "index_n_heads": 64,
  "index_head_dim": 128,
  "first_k_dense_replace": 3
}
```

### DSA vs MLA

| 特性 | DSA | MLA |
|------|-----|-----|
| Lightning Indexer | Decode 阶段启用 | 无 |
| Decode 稀疏 | index_topk 选择 | 完整 KV |
| KV 压缩 | kv_lora_rank | kv_lora_rank |

### DSA 关键实现

```python
# DSA 使用 topk_tokens 参数（绝对值）
if is_prefill:
    effective_kv_len = kv_seq_len
else:
    effective_kv_len = min(index_topk, kv_seq_len)
```

---

## 2. Kimi K2.5 (MLA)

### 关键参数

```json
{
  "hidden_size": 7168,
  "num_hidden_layers": 61,
  "num_attention_heads": 64,
  "num_experts": 384,
  "num_experts_per_tok": 8,
  "num_shared_experts": 1,
  "moe_intermediate_size": 2048,
  "kv_lora_rank": 512,
  "qk_nope_head_dim": 128,
  "qk_rope_head_dim": 64,
  "attention_type": "mla",
  "first_k_dense_replace": 1
}
```

### Kimi K2.5 vs DeepSeek V3.2

| 特性 | Kimi K2.5 | DeepSeek V3.2 |
|------|----------|---------------|
| attention_type | "mla" | "dsa" |
| Lightning Indexer | 无 | 有 |
| num_experts | 384 | 256 |
| max_seq | 256K | 128K |
| MTP | 无 | 有 |

### 建模注意

Kimi K2.5 使用 `ModuleMLAAttention`，使用 DSA 模块会因缺少 `index_n_heads` 参数报错。

---

## 3. Qwen 2.5 (GQA + Dense)

### 关键参数

```json
{
  "hidden_size": 8192,
  "num_hidden_layers": 80,
  "num_attention_heads": 64,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "intermediate_size": 29568,
  "attention_type": "gqa"
}
```

### GQA 特性

- 无 MLA KV 压缩
- KV cache = `2 × num_kv_heads × head_dim`
- GQA ratio = num_heads / num_kv_heads = 8

---

## 4. MiniMax M2.5 (GQA + MoE)

### 关键参数

```json
{
  "hidden_size": 3072,
  "num_hidden_layers": 62,
  "num_attention_heads": 48,
  "num_key_value_heads": 8,
  "num_experts": 256,
  "num_experts_per_tok": 8,
  "num_shared_experts": 0,
  "attention_type": "gqa"
}
```

### 关键差异

| 特性 | MiniMax M2.5 | Qwen 2.5 |
|------|-------------|----------|
| FFN | MoE | Dense |
| Shared Expert | 无 | N/A |
| MTP | 3 模块 | 无 |

### 无 Shared Expert 的影响

```python
# DeepEP overlap 失效
shared_expert_time_ms = 0.0  # n_shared = 0
# → dispatch 通信无法被隐藏
```

---

## 5. Qwen 3.5 (Hybrid + MoE)

### 关键参数

```json
{
  "hidden_size": 4096,
  "num_hidden_layers": 60,
  "num_attention_heads": 32,
  "num_key_value_heads": 2,
  "head_dim": 256,
  "linear_num_key_heads": 16,
  "linear_num_value_heads": 64,
  "linear_key_head_dim": 128,
  "linear_value_head_dim": 128,
  "num_experts": 512,
  "num_experts_per_tok": 10,
  "attention_type": "hybrid",
  "full_attention_interval": 4
}
```

### 混合注意力结构

```
Layer 0-2:   Linear Attention (Gated DeltaNet)
Layer 3:     Full Attention (GQA)
Layer 4-6:   Linear Attention
Layer 7:     Full Attention
...

总计: 45 层 Linear + 15 层 Full
```

### Linear Attention 特性

- 固定大小状态：O(d²) per token
- 无 growing KV cache
- Decode 比 Full Attention 快 ~100 倍 (8K context)

### 层类型判断

```python
interval = model_config.full_attention_interval or 4
layer_type = "full_attention" if (layer_idx + 1) % interval == 0 else "linear_attention"
```

### 建模注意

- Linear Attention 不需要 CP 通信
- MTP 模块使用 GQA (Full Attention)
- CP 通信只用于 Full Attention 层

---

## Prefill 建模

### Prefill vs Decode 差异

| 特性 | Decode | Prefill |
|------|--------|---------|
| seq_len | 1 | input_length × (1-cache_hit) / CP |
| kv_seq_len | input_length + 1 | input_length × (1-cache_hit) |
| DSA Sparse | index_topk | Full Attention |
| Lightning Indexer | 启用 | 禁用 |
| DeepEP Mode | low_latency | high_throughput |

### CP 建模

```python
effective_seq = input_length * (1 - prefix_cache_hit_rate)
seq_per_cp = effective_seq / CP

# CP 通信量
kv_bytes = seq_per_cp * kv_cache_dim * dtype
# MLA: kv_cache_dim = kv_lora_rank + qk_rope_head_dim
# GQA: kv_cache_dim = num_kv_heads * head_dim
```

### Prefix Cache

```python
# 命中的 token 不需要计算
effective_seq = input_length * (1 - prefix_cache_hit_rate)

# 典型场景
# 系统提示: 30-50%
# Few-shot: 50-70%
# 多轮对话: 70-90%
```

### CP 错误

```python
# ❌ 错误：kv_seq_len = seq_per_cp
# ✅ 正确：kv_seq_len = effective_seq_len (完整序列)
```
# 性能建模公式汇总

本文档汇总所有 FLOPs、访存量、通信量公式。

---

## 1. MoE FLOPs 公式

### 关键概念：moe_batch_size vs gate_batch_size

```python
gate_batch_size = micro_batch_size / attention_tp   # Router Gate
moe_batch_size = micro_batch_size / attention_tp * moe_tp  # Expert 投影
```

### Router Gate FLOPs

```python
flops = 2 * gate_batch_size * seq * hidden * num_experts
```

- 权重: `[hidden, num_experts]`
- 输出: 每个 token 对所有专家的分数

### Expert Gate/Up/Down FLOPs

**Gate 投影 (ColumnParallel)**:
```python
flops = 2 * hidden * (intermediate / moe_tp) * top_k * moe_batch_size * seq
vector = token_expert_pairs * intermediate_per_tp * 4  # SiLU (但归入 Down)
```

**Up 投影 (ColumnParallel)**:
```python
flops = 2 * hidden * (intermediate / moe_tp) * top_k * moe_batch_size * seq
```

**Down 投影 (RowParallel)**:
```python
# CUBE: 输入是 gate+up 融合，维度为 intermediate*2
flops = 2 * hidden * (intermediate / moe_tp) * 2 * top_k * moe_batch_size * seq
# Vector: 所有 SwiGLU 激活在此建模
vector = token_expert_pairs * intermediate_per_tp * 7
```

### Shared Expert

公式与 Routed Expert 相同，但 `top_k = 1`。

### EP 不影响 FLOPs

```python
# ❌ 错误
flops = 2 * batch * seq * top_k * intermediate * hidden / ep

# ✅ 正确：不除以 EP
flops = 2 * batch * seq * top_k * intermediate * hidden
```

---

## 2. MoE 权重访存公式

**核心原则：EP 切分专家，不切分权重！**

```python
num_experts_per_ep = ceil(num_experts / ep) + r_per_ep

# Routed Expert
read_weight = hidden * intermediate_per_tp * weight_bytes * num_experts_per_ep

# Shared Expert (不使用 EP)
read_weight = hidden * intermediate_per_tp * weight_bytes * num_shared_experts
```

**数值示例** (DeepSeek V3, ep=8, num_experts=256):
| 实现 | 权重访存 |
|------|---------|
| 错误: `/ ep` | 1.84 MB |
| 正确: `* num_experts_per_ep` | 469 MB |

---

## 3. Attention FLOPs 公式

### MLA/DSA Attention

```python
qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

# DSA Decode: effective_kv_len = min(index_topk, kv_seq_len)
# DSA Prefill: effective_kv_len = kv_seq_len

qk_flops = 2 * B * (H/TP) * seq * qk_head_dim * effective_kv_len
sv_flops = 2 * B * (H/TP) * seq * effective_kv_len * v_head_dim
total = qk_flops + sv_flops
```

### GQA Attention

```python
qk_flops = 2 * B * (H/TP) * seq * head_dim * kv_seq_len
sv_flops = 2 * B * (H/TP) * seq * kv_seq_len * head_dim
total = 4 * B * (H/TP) * seq * kv_seq_len * head_dim
```

### Linear Attention (DeltaNet)

```python
# 固定大小状态，O(d²) per token
state_update = 2 * num_key_heads * key_head_dim * value_head_dim
query = 2 * num_value_heads * value_head_dim * key_head_dim
per_token = state_update + query
total = B * seq * per_token
```

---

## 4. Attention 访存量公式

### MLA/DSA KV Cache 结构

```
KV cache per token = kv_lora_rank + qk_rope_head_dim
```

- `compressed_kv`: kv_lora_rank (512)
- `k_pe`: qk_rope_head_dim (64)

**DeepSeek-V3**: 512 + 64 = **576 values/token/layer**

### MLA/DSA 访存量

```python
kv_cache_dim = kv_lora_rank + qk_rope_head_dim

# DSA Decode
effective_kv_len = min(index_topk, kv_seq_len)

read_q = B * (H/TP) * seq * qk_head_dim * act_bytes
read_kv = B * effective_kv_len * kv_cache_dim * cache_bytes
write_out = B * (H/TP) * seq * v_head_dim * act_bytes
```

### GQA 访存量

```python
# 系数 2：K 和 V 都需要读取
read_kv = 2 * B * (num_kv_heads/TP) * kv_seq_len * head_dim * cache_bytes
```

### Linear Attention 访存量

```python
# 固定状态，无 growing KV cache
read_state = B * (num_key_heads/TP) * key_head_dim * value_head_dim * cache_bytes
```

### KV Cache 对比

| 类型 | KV Cache per token/layer | DeepSeek-V3 | Qwen 2.5 72B |
|------|-------------------------|-------------|--------------|
| MLA/DSA | kv_lora_rank + qk_rope_head_dim | 576 × 2 = 1,152 B | N/A |
| GQA | 2 × num_kv_heads × head_dim | N/A | 4,096 B |

---

## 5. 通信公式

### TP 通信

```python
# AllReduce
time = 2 * data / (bw * N) + rtt

# AllGather / ReduceScatter
time = data / (bw * N) + rtt

# data = batch * seq * hidden * act_bytes
```

### EP All-to-All

```python
tokens = batch * seq / attention_tp
data_size = tokens * top_k * (ep - 1) / ep * hidden * act_bytes
time = data_size / bw + rtt * sqrt(N-1)  # High-throughput
time = base_latency + transfer_time * log2(EP)  # Low-latency (DeepEP)
```

### PP P2P

```python
time = data / bw + rtt + static_overhead
```

---

## 6. Norm 公式

### RMSNorm

```python
vector_flops = 6 * batch * seq * hidden
mem_bytes = 2 * batch * seq * hidden * dtype
```

### LayerNorm

```python
vector_flops = 8 * batch * seq * hidden
mem_bytes = 2 * batch * seq * hidden * dtype
```

---

## 7. 性能指标公式

### Per-Card

```python
# Decode
tps_per_card = total_bs / tpot_s / num_cards
qps_per_card = tps_per_card / output_length

# Prefill
qps_per_card = total_bs / ttft_s / num_cards
tps_per_card = qps_per_card * input_length
```

### System

```python
system_tps = total_bs / tpot_s
system_qps = total_bs / tpot_s / output_length
```

### Batch Size

```python
num_tp_groups = total_chips / attention_tp / CP
global_batch = micro_batch * num_tp_groups
```

---

## 8. SwiGLU Vector FLOPs

**统一在 Down 层建模**：7 FLOPs/element

```
SiLU: 3 FLOPs (neg + exp + reciprocal)
silu * up: 2 FLOPs (2 reads + mul)
fusion prep: 2 FLOPs
Total: 7 FLOPs/element
```
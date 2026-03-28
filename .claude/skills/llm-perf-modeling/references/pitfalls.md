# 常见错误汇总

本文档记录 LLM 性能建模中常见的 35 个错误。

---

## 错误 1-5: 性能指标与 MoE Gate

### 错误 1：混淆 TPS 和 QPS

```python
# ❌ 错误：认为 Decode 时 QPS = TPS
# 实际上 Decode 时 QPS = TPS / output_length

# ✅ 正确
# Decode: QPS = TPS / output_length
# Prefill: TPS = QPS × input_length
```

### 错误 2：TPOT 没有考虑 batch_size

```python
# ❌ 错误
tpot = total_time  # total_time 包含 batch 维度

# ✅ 正确
tpot = total_time / micro_batch_size
```

### 错误 3：系统吞吐计算时乘以 num_tp_groups 而不是 num_cards

```python
# ❌ 错误
system_tps = tps_per_card * num_tp_groups

# ✅ 正确
system_tps = tps_per_card * num_cards
# 或: system_tps = total_bs / tpot_s
```

### 错误 4：Gate (Router) 的 batch 直接使用 micro_batch_size

```python
# ❌ 错误：Gate 是 replicated linear，每个 TP rank 独立计算
gate_flops = 2 * micro_bs * seq * hidden * num_experts

# ✅ 正确：有效 batch = micro_bs / attn_tp
gate_flops = 2 * (micro_bs / attn_tp) * seq * hidden * num_experts
```

### 错误 5：混淆 Router Gate 和 Expert Gate 的公式

| 类型 | 类 | 输出维度 | FLOPs 公式 |
|------|-----|---------|-----------|
| Router Gate | LayerMoEGate | num_experts (256) | `2 * batch * seq * hidden * num_experts` |
| Expert Gate | LayerMoEGateProj | intermediate (2048) | `2 * batch * seq * top_k * hidden * intermediate` |

---

## 错误 6-10: 通信算子

### 错误 6：认为 share_up 和 share_gate_proj 的 CUBE FLOPs 不同

它们相同！都是 hidden → intermediate 的投影。区别仅在于 Vector FLOPs（Gate 有 SiLU）。

### 错误 7：遗漏 Attention 前后的 TP 通信算子

```python
# ❌ 错误：模型中只有计算算子
self.add_module('attention', Attention(...))
self.add_module('moe', MoE(...))

# ✅ 正确：根据 TP 配置添加通信算子
if self.upstream_tp < self.attention_tp:
    self.add_layer('allgather_input', LayerAllGather(...))
# ... attention 计算 ...
if self.attention_tp > 1:
    self.add_layer('allreduce_output', LayerAllReduce(...))
if self.attention_tp > self.downstream_tp:
    self.add_layer('reduce_scatter_output', LayerReduceScatter(...))
```

### 错误 8：MoE 模块没有处理 TP 级别转换

```python
# ✅ 正确：MoE 模块的 TP 通信
if self.attention_tp < self.moe_tp:
    self.add_layer('allgather_moe_tp', LayerAllGather(...))
# ... MoE 计算 ...
if self.attention_tp > self.moe_tp:
    self.add_layer('reduce_scatter_moe_tp', LayerReduceScatter(...))
    self.add_layer('allgather_restore', LayerAllGather(...))
```

### 错误 9：认为 Embedding 后需要 AllGather 才能做 Attention

VocabParallelEmbedding 内部已有 AllReduce，输出是 replicated 状态。第一层 Attention 的 upstream_tp = attention_tp 是正确的。

### 错误 10：LayerEmbedding 没有建模内部 AllReduce 的通信代价

```python
class LayerEmbedding(LayerBase):
    def get_comm_bytes(self):
        if self.lm_head_tp <= 1:
            return 0.0
        return 2 * (self.lm_head_tp - 1) / self.lm_head_tp * self.allreduce_data_size
```

---

## 错误 11-15: FLOPs 与 EP 权重访存

### 错误 11：Down 投影没有 Vector FLOPs（遗漏 SwiGLU 激活）

```python
# ❌ 错误
def get_vector_flops(self):
    return 0.0

# ✅ 正确：SwiGLU 激活在 Down 层建模
def get_vector_flops(self):
    return token_expert_pairs * intermediate_per_tp * 7
```

### 错误 12：SwiGLU 激活 FLOPs 在 gate_proj 和 down_proj 重复计算

统一在 Down 层建模，gate_proj 的 vector 设为 0。

### 错误 13：Vector 算力根据 act_compute_bits 选择 FP16/FP32

Vector 单元**统一使用 FP16**，与 activation 计算精度无关。

### 错误 14：PP P2P 通信算子缺失

PP > 1 时，每个 stage 边界需要 P2P 通信。

### 错误 15：EP 权重访存计算错误

**核心原则：EP 切分的是专家，不是权重！**

```python
# ❌ 错误：将单个专家权重除以 EP
read_weight = hidden * intermediate * weight_bytes / ep

# ✅ 正确：每个 EP rank 存储 num_experts_per_ep 个专家的完整权重
read_weight = hidden * intermediate * weight_bytes * num_experts_per_ep
```

**Shared vs Routed Expert**:
- Routed Expert: `weight × num_experts_per_ep`
- Shared Expert: `weight × num_shared_experts` (不使用 EP)

---

## 错误 16-20: PP Bubble 与 EP 通信

### 错误 16：PP bubble 未在性能计算中体现

```python
if pp > 1:
    bubble_rate = deploy_config.pipeline_bubble_rate
    effective_time = total_time * (1 + bubble_rate)
else:
    effective_time = total_time
tpot = effective_time / micro_batch_size
```

### 错误 17：EP All-to-All 通信量公式错误

```python
# ✅ 正确：Per-rank 通信量
tokens = batch * seq / attention_tp
data_size_dispatch = tokens * top_k * (ep - 1) / ep * hidden * act_bytes
```

### 错误 18：未利用 DeepEP 的 Compute-Communication Overlap

```python
# DeepEP: dispatch 与 shared expert 并行
dispatch_effective = max(dispatch_time, shared_expert_time)
```

### 错误 19：Compute-Communication Overlap 导致 Double-Counting

从 total 中减去已添加的 accumulated_compute 再加 effective。

### 错误 20：忽略 EP 负载不均衡对计算时间的影响

```python
expert_compute_time = base_compute_time * ep_load_imbalance_factor
# 典型值: 1.0 ~ 1.5
```

---

## 错误 21-25: Attention FLOPs 与 GQA

### 错误 21：MLA/DSA Attention CUBE FLOPs 计算混淆

```python
# ✅ 正确：显式计算 Q@K 和 S@V
qk_flops = 2 * B * H/TP * S * qk_head_dim * effective_kv_len
sv_flops = 2 * B * H/TP * S * effective_kv_len * v_head_dim
return qk_flops + sv_flops
```

### 错误 22：Prefill CP 场景下 kv_seq_len 使用 seq_per_cp

```python
# ❌ 错误
attn_module = ModuleDSAAttention(..., seq_len=seq_per_cp, is_prefill=True)

# ✅ 正确：显式指定 kv_seq_len
attn_module = ModuleDSAAttention(
    ..., seq_len=seq_per_cp, is_prefill=True,
    kv_seq_len=effective_seq_len  # 完整序列
)
```

### 错误 23：Prefix Cache Hit Rate 仅减少 KV cache 读取

Prefix Cache 命中的 token 不需要经过任何计算，不仅仅是跳过 KV cache 读取。

### 错误 24：GQA 模型的 CP 通信使用 kv_lora_rank

```python
# ❌ 错误：GQA 模型没有 kv_lora_rank
cp_comm = LayerCPComm(..., kv_lora_rank=512)

# ✅ 正确：GQA 使用 num_kv_heads × head_dim
kv_cache_size = num_kv_heads * head_dim
cp_comm = LayerCPComm(..., kv_cache_size=kv_cache_size)
```

### 错误 25：GQA Attention 使用 MLA 的 head_dim 公式

```python
# ❌ 错误
qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

# ✅ 正确：GQA 使用标准的 head_dim
attn_flops = 2 * B * H * S * head_dim * KV
```

---

## 错误 26-32: 模型选择与配置

### 错误 26：Kimi K2.5 模型使用 DSA 模块

Kimi K2.5 使用 `attention_type: "mla"`，不是 "dsa"。应使用 ModuleMLAAttention。

### 错误 27：ModelConfig 的 Optional 字段默认值处理

```python
# ❌ 错误：getattr 的默认值无法覆盖 dataclass 的 None 默认值
first_k = getattr(model_config, 'first_k_dense_replace', 0)  # 返回 None！

# ✅ 正确
first_k = getattr(model_config, 'first_k_dense_replace', None) or 0
```

### 错误 28：GQA + MoE 模型使用 Qwen 2.5 的 Dense 实现

```python
if attention_type == 'gqa':
    if model.is_moe:
        inference_model = DecodeMiniMaxM25(...)  # MoE
    else:
        inference_model = DecodeQwen2_5(...)      # Dense
```

### 错误 29：MTP 模块硬编码 MLA Attention

```python
attention_type = getattr(model_config, 'attention_type', 'mla')
if attention_type in ('mla', 'dsa'):
    attn = ModuleMLAAttention(...)
elif attention_type == 'gqa':
    attn = ModuleGQAAttention(...)
```

### 错误 30：Linear Attention 层误用 GQA 的投影维度

Linear Attention 使用独立的维度：`linear_num_key_heads`, `linear_key_head_dim` 等。

### 错误 31：Linear Attention 层添加 CP 通信

Linear Attention 状态大小固定，不需要 CP 通信。

### 错误 32：MTP 模块使用 Linear Attention

Qwen3.5 MTP 使用 GQA (Full Attention)，不是 Linear Attention。

---

## 错误 33-35: Attention 访存量

### 错误 33：MLA/DSA KV cache 访存遗漏 qk_rope_head_dim

```python
# ❌ 错误：只读取 kv_lora_rank
read_kv = batch * kv_seq_len * kv_lora_rank * cache_bytes

# ✅ 正确：KV cache = compressed_kv + k_pe
read_kv = batch * kv_seq_len * (kv_lora_rank + qk_rope_head_dim) * cache_bytes
```

**数值影响**: DeepSeek-V3 遗漏 `qk_rope_head_dim=64`，低估 12.5%。

### 错误 34：GQA KV cache 访存遗漏 K 或 V

```python
# ❌ 错误：系数 1
read_kv = batch * num_kv_heads/TP * kv_seq_len * head_dim * cache_bytes

# ✅ 正确：系数 2（K 和 V 都需要读取）
read_kv = 2 * batch * num_kv_heads/TP * kv_seq_len * head_dim * cache_bytes
```

**数值影响**: 低估 50% KV cache 访存量。

### 错误 35：将用户公式中的 `(nope*2+rope)` 误解为权重访存

`(nope*2+rope)` 是 CUBE FLOPs 公式中的维度，不是访存公式。访存使用 `kv_lora_rank + qk_rope_head_dim`。
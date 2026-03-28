---
name: llm-perf-modeling
description: LLM性能建模工具 - 基于Roofline模型分析大语言模型性能。当用户需要评估模型性能、分析瓶颈、搜索最优并行策略时使用。
---

# LLM Roofline 性能建模

你是一个 LLM 性能建模专家，基于 Roofline 模型分析大语言模型在不同硬件上的性能表现。

## 核心职责

1. **配置获取与验证**: 硬件、模型、量化、部署配置
2. **代码生成**: 算子层 (LayerBase)、模块层 (ModuleBase)、模型层 (InferenceBase)
3. **性能仿真**: 计算 TPOT/TTFT/TPS/QPS 等指标
4. **瓶颈分析**: 识别计算/访存/通信瓶颈
5. **系统寻优**: 搜索最优并行策略

---

## 工作流程

### 阶段 1: 配置获取

| 配置类型 | 来源 | 关键参数 |
|----------|------|----------|
| hardware_config | `configs/hardware/` | CUBE/Vector 算力、HBM 带宽、通信带宽 |
| model_config | HuggingFace | hidden_size, num_experts, attention_type |
| quant_config | `configs/quantization/` | weight_bits, cache_read_bits |
| deploy_config | `configs/deployment/` | TP/EP/PP/CP, micro_batch_size |

**必须参考 vLLM 代码验证模型结构！**

### 阶段 2: 代码生成

```python
class LayerXXX(LayerBase):
    def get_cube_flops(self): ...    # CUBE 计算量
    def get_vector_flops(self): ...  # Vector 计算量
    def get_mem_bytes(self): ...     # 访存量
    def get_comm_bytes(self): ...    # 通信量 (通信算子)
```

**时延公式**: `max(cube_time + vector_time, mem_time) + op_overhead + comm_time`

**Vector 统一使用 FP16 算力**，与 activation 计算精度无关。

### 阶段 3: 性能计算

```bash
python main.py --hardware ... --model ... --quant ... --deploy ...
```

输出: `op_details.csv`, `single_card_perf.json`, `system_perf.json`

### 阶段 4: 瓶颈分析

| 瓶颈类型 | 占比阈值 | 优化方向 |
|----------|----------|----------|
| CUBE 低 | <10% | 增大 batch/seq |
| Memory 高 | >40% | 量化、优化访存 |
| Comm 高 | >30% | 调整并行策略 |

---

## 并行策略影响

| 策略 | FLOPs | 权重访存 | 通信 | 关键影响 |
|------|-------|----------|------|----------|
| **TP** | ÷TP | ÷TP | AllReduce | 切分权重矩阵 |
| **EP** | 不变 | ×(num_experts/ep) | All-to-All | 存储部分专家 |
| **PP** | ÷PP | ÷PP | P2P | 切分层数+bubble |
| **CP** | ÷CP | 不变 | Ring Attn | 切分序列 |
| **moe_tp** | 不变 | ÷moe_tp | AG/RS | 切分 expert intermediate |

**EP 是唯一不影响 FLOPs 但影响权重访存的策略！**

详见 [references/parallelism.md](references/parallelism.md)

---

## 模型架构

| 模型 | Attention | FFN | attention_type | 代码路径 |
|------|-----------|-----|----------------|----------|
| DeepSeek V3.2 | DSA | MoE | "dsa" | DecodeDeepSeekV32 |
| Kimi K2.5 | MLA | MoE | "mla" | DecodeDeepSeekV32 |
| Qwen 2.5 | GQA | Dense | "gqa" | DecodeQwen2_5 |
| MiniMax M2.5 | GQA | MoE | "gqa" | DecodeMiniMaxM25 |
| Qwen 3.5 | Linear+GQA | MoE | "hybrid" | DecodeQwen35 |

详见 [references/model_guide.md](references/model_guide.md)

---

## 关键公式速查

### MoE FLOPs

```python
# Router Gate
flops = 2 * (micro_bs/attn_tp) * seq * hidden * num_experts

# Expert Gate/Up/Down
flops = 2 * hidden * (intermediate/moe_tp) * top_k * moe_batch * seq
# moe_batch = micro_bs / attn_tp * moe_tp
```

### EP 权重访存

```python
num_experts_per_ep = ceil(num_experts / ep) + r_per_ep
read_weight = hidden * intermediate * weight_bytes * num_experts_per_ep
# Shared Expert: 不使用 EP，用 num_shared_experts
```

### Attention KV Cache

| 类型 | KV Cache per token | 公式 |
|------|-------------------|------|
| MLA/DSA | kv_lora_rank + qk_rope_head_dim | `batch * kv_seq * 576 * cache_bytes` |
| GQA | 2 × num_kv_heads × head_dim | `2 * batch * kv_seq * ...` |

**GQA 系数 2 不能省略！K 和 V 都需要读取。**

详见 [references/formulas.md](references/formulas.md)

---

## 常见错误速查

| 错误 | 描述 |
|------|------|
| #4 | Gate batch 使用 micro_bs 而非 micro_bs/attn_tp |
| #11 | Down 投影遗漏 SwiGLU Vector FLOPs (7/element) |
| #15 | EP 权重访存错误除以 ep |
| #17 | EP All-to-All 遗漏 top_k 或 (EP-1)/EP |
| #33 | MLA/DSA KV cache 遗漏 qk_rope_head_dim |
| #34 | GQA KV cache 遗漏系数 2 |

详见 [references/pitfalls.md](references/pitfalls.md)

---

## 性能指标

### Per-Card

```python
# Decode
tps = total_bs / tpot_s / num_cards
qps = tps / output_length

# Prefill
qps = total_bs / ttft_s / num_cards
tps = qps * input_length
```

### System

```python
system_tps = total_bs / tpot_s
system_qps = total_bs / tpot_s / output_length
```

### Batch Size

```python
num_tp_groups = total_chips / attn_tp / CP
global_batch = micro_batch * num_tp_groups
```

---

## 算子序列概览

### Dense FFN

`gate_proj` → `up_proj` → `down_proj` → `allreduce`

### MoE

`e_topk` → `[allgather_tp]` → `shared_expert` → `[dispatch]` → `routed_expert` → `[combine]` → `[rs_tp]`

### DSA Attention (Decode)

`q_a_proj` → `q_b_proj` → `kv_a_proj` → **`indexer_*`** → `kv_b_proj` → `dsa_attention` → `o_proj`

**Lightning Indexer 只在 Decode 运行，无 TP 切分。**

详见 [references/architecture.md](references/architecture.md)

---

## DeepEP 优化

| 技术 | 效果 |
|------|------|
| 双模式内核 | Prefill/Decode 分别优化 |
| Pure RDMA | 延迟降低 50%+ |
| Hook 机制 | 通信与计算重叠 |
| Overlap | `max(dispatch, shared_expert)` |

**配置**: `deepep_base_latency_us=50`, `deepep_overlap_efficiency=0.9`

详见 [references/parallelism.md](references/parallelism.md#5-deepep-优化技术)

---

## 代码参考

- `references/layer_base.py` - 算子基类
- `references/module_base.py` - 模块基类
- `references/inference_base.py` - 模型基类
- `vllm/model_executor/models/` - vLLM 模型实现

---

## 参考文档

- [架构与算子序列](references/architecture.md)
- [公式汇总](references/formulas.md)
- [并行策略详解](references/parallelism.md)
- [模型建模指南](references/model_guide.md)
- [常见错误汇总](references/pitfalls.md)
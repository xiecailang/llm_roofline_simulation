# 架构与算子序列

本文档描述 Transformer 层结构、算子序列和 Forward 流程。

---

## 1. Transformer 层结构

### Dense FFN 层 (前 first_k_dense_replace 层)

| 序号 | 算子名称 | 类型 | 说明 |
|------|----------|------|------|
| 1 | `gate_proj` | CUBE | hidden → intermediate (ColumnParallel + SiLU) |
| 2 | `up_proj` | CUBE | hidden → intermediate (ColumnParallel) |
| 3 | `down_proj` | CUBE | intermediate → hidden (RowParallel) |
| 4 | `allreduce` | 通信 | attention_tp > 1 时 |

**Dense FFN FLOPs**: `4 × hidden × (intermediate/TP) × batch × seq`

### MoE 层

| 序号 | 算子名称 | 类型 | 触发条件 | 说明 |
|------|----------|------|----------|------|
| 1 | `e_topk_weight` | CUBE | 始终 | Router Gate |
| 2 | `allgather_moe_tp` | 通信 | attn_tp < moe_tp | TP 级别转换 |
| 3 | `share_up` | CUBE | n_shared > 0 | Shared expert up |
| 4 | `share_gate_proj` | CUBE | n_shared > 0 | Shared expert gate |
| 5 | `share_down` | CUBE | n_shared > 0 | Shared expert down |
| 6 | `dispatch` | 通信 | EP > 1 | All-to-All |
| 7 | `moe_up` | CUBE | 始终 | Routed expert up |
| 8 | `moe_gate_proj` | CUBE | 始终 | Routed expert gate |
| 9 | `moe_down` | CUBE | 始终 | Routed expert down |
| 10 | `combine` | 通信 | EP > 1 | All-to-All |
| 11 | `reduce_scatter_moe_tp` | 通信 | attn_tp > moe_tp | TP 级别转换 |

### Attention 通信算子

| 序号 | 算子名称 | 触发条件 | 位置 | 说明 |
|------|----------|----------|------|------|
| 1 | `allgather_input` | upstream_tp < attention_tp | Attention 前 | 恢复完整激活 |
| 2 | `allreduce_output` | attention_tp > 1 | o_proj 后 | 聚合各 head |
| 3 | `reduce_scatter_output` | attention_tp > downstream_tp | Attention 后 | 切分给下游 |

---

## 2. DSA 完整算子序列 (DeepSeek-V3.2)

DSA = MLA + Lightning Indexer

| 序号 | 算子名称 | 类型 | TP | 触发条件 | 说明 |
|------|----------|------|-----|----------|------|
| 1 | `allgather_input` | 通信 | ✓ | upstream_tp < attention_tp | |
| 2 | `input_norm` | Vector | - | 始终 | RMSNorm |
| 3 | `q_a_proj` | CUBE | ✓ | 始终 | hidden → q_lora_rank |
| 4 | `q_a_norm` | Vector | ✓ | 始终 | RMSNorm |
| 5 | `q_b_proj` | CUBE | ✓ | 始终 | q_lora_rank → heads × qk_dim |
| 6 | `kv_a_proj` | CUBE | - | 始终 | hidden → kv_lora_rank + rope |
| 7 | `kv_a_norm` | Vector | - | 始终 | RMSNorm |
| **8** | `indexer_wq_b` | CUBE | **✗** | **Decode only** | Lightning Indexer |
| **9** | `indexer_wk` | CUBE | **✗** | **Decode only** | |
| **10** | `indexer_k_norm` | Vector | **✗** | **Decode only** | LayerNorm |
| **11** | `indexer_weights_proj` | CUBE | **✗** | **Decode only** | 无量化 |
| **12** | `sparse_attn_indexer` | CUBE+Vector | **✗** | **Decode only** | FP8 MQA + TopK |
| 13 | `kv_b_proj` | CUBE | ✓ | 始终 | |
| 14 | `dsa_attention` | CUBE | ✓ | 始终 | FlashMLA 稀疏 |
| 15 | `o_proj` | CUBE | ✓ | 始终 | |
| 16 | `allreduce_output` | 通信 | ✓ | attention_tp > 1 | |
| 17 | `reduce_scatter_output` | 通信 | ✓ | attn_tp > downstream_tp | |

**Lightning Indexer 关键特性**：
- 只在 Decode 阶段运行
- 所有线性层使用 ReplicatedLinear，无 TP 切分
- `indexer_weights_proj` 不使用量化
- 维护独立的 FP8 K cache

---

## 3. Forward 流程

```
PP Stage 0:
  Embedding (VocabParallel + AllReduce → replicated)
    │
    ▼
  ┌─ Transformer Layer ─────────────────────────────────────┐
  │  [AllGather]          ← upstream_tp < attention_tp      │
  │  Attention (at attention_tp)                             │
  │  [AllReduce]          ← attention_tp > 1                │
  │  [ReduceScatter]      ← attention_tp > downstream_tp    │
  │                                                          │
  │  if MoE layer:                                           │
  │    [AllGather]        ← attention_tp < moe_tp           │
  │    Shared Expert (可与 Dispatch 重叠)                    │
  │    [Dispatch]        ← EP All-to-All                    │
  │    Routed Expert                                         │
  │    [Combine]         ← EP All-to-All                    │
  │    [RS + AG]          ← attention_tp > moe_tp           │
  │  else Dense FFN:                                         │
  │    Dense FFN + [AllReduce]                              │
  └──────────────────────────────────────────────────────────┘
    │
    ▼
  [P2P Send]  ← PP stage 边界
```

---

## 4. 层类型判断

```python
def is_moe_layer(layer_idx: int) -> bool:
    return (
        config.num_experts is not None
        and layer_idx >= config.first_k_dense_replace
        and layer_idx % config.moe_layer_freq == 0
    )
```

**DeepSeek V3.2** (61层):
- Layer 0-2: Dense FFN (3层)
- Layer 3-60: MoE (58层)

**Kimi K2.5** (61层):
- Layer 0: Dense FFN (1层)
- Layer 1-60: MoE (60层)

---

## 5. 算子命名规范

| 层类型 | 类名前缀 | 文件名前缀 |
|--------|----------|------------|
| Router Gate | `LayerMoEGate` | `layer_moe_gate.py` |
| Expert 投影 | `LayerExpert*` | `layer_expert_*.py` |
| Dense FFN | `LayerDense*` | `layer_dense_*.py` |

**Expert 投影层同时用于 Shared 和 Routed Expert**：
- `LayerExpertUp`
- `LayerExpertGateProj`
- `LayerExpertDown`
- 通过 `top_k` 参数区分：Shared (top_k=1) vs Routed (top_k=8)
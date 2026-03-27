---
name: llm-perf-modeling
description: LLM性能建模工具 - 基于Roofline模型的大语言模型性能仿真
---

# LLM Roofline 性能建模 Skill

你是一个专业的 LLM 性能建模专家，负责基于 Roofline 模型分析大语言模型在不同硬件上的性能表现。

## 核心职责

1. **获取和验证配置**: 硬件配置、模型结构、量化策略、部署方案
2. **生成性能建模代码**: 算子层、模块层、模型层的实现
3. **运行性能仿真**: 计算时延、吞吐、内存占用等指标
4. **分析性能瓶颈**: 识别计算、访存、通信的瓶颈
5. **系统寻优**: 搜索最优并行策略和部署配置

## 工作流程

### 阶段 1: 配置获取与验证

#### 1.1 硬件配置 (hardware_config)
检查 `configs/hardware/` 目录，如果配置不存在则获取：
- **单芯片算力**: CUBE(FP4/FP8/FP16/FP32)、Vector、利用率
- **多级缓存**: HBM、Host Memory、SSD 的大小、带宽、利用率
- **通信**: 4GPU/8GPU/框内/框间带宽、P2P、RTT、静态开销
- **芯片配置**: 单卡芯片数、单框最大芯片数、总芯片数、芯片名称

#### 1.2 模型结构配置 (model_config)
从 HuggingFace 下载或用户提供，**必须参考 vllm 代码验证**：
- 基础参数: hidden_size, num_hidden_layers, num_attention_heads, vocab_size
- MoE参数: num_experts, num_experts_per_tok, num_shared_experts, moe_intermediate_size
- Attention类型: attention_type (mha/gqa/mla/dsa)
- MLA/DSA参数: qk_nope_head_dim, qk_rope_head_dim, v_head_dim, kv_lora_rank, q_lora_rank
- **DSA特有**: topk_tokens (稀疏注意力选择的token数量，绝对值而非比例)

**关键**: 不同模型字段名可能不同，需结合 vllm 代码确认。

#### 1.3 量化配置 (quant_config)
每个算子可配置不同精度(4/8/16/32 bit):
- weight_bits, activation_compute_bits, activation_transfer_bits
- cache_read_bits, cache_write_bits

#### 1.4 部署配置 (deploy_config)
- 部署方式: pd_mixed, pd_separated, afd_separated
- 投机解码: mtp_length, mtp_acceptance_rate
- 并行策略: CP, attention_tp, PP, EP, moe_tp, lm_head_tp
- 业务负载: input_length, output_length, prefix_cache_hit_rate

### 阶段 2: 代码生成

#### 2.1 算子层 (layers/)
继承 `LayerBase`，实现：
```python
class LayerXXX(LayerBase):
    def get_cube_flops(self):
        # 返回 CUBE 计算量 (FLOPs)
        pass

    def get_vector_flops(self):
        # 返回 Vector 计算量 (FLOPs)
        pass

    def get_mem_bytes(self):
        # 返回访存量 (Bytes)
        pass

    def get_comm_bytes(self):  # 通信算子
        # 返回通信量 (Bytes)
        pass
```

**时延计算公式**:
```
latency = max(cube_time + vector_time, mem_time) + op_overhead + comm_time

其中:
- cube_time: 根据 activation 计算精度选择 FP4/FP8/FP16/FP32 算力
- vector_time: **统一使用 FP16 算力**（激活函数在 FP16 下执行）
- mem_time: HBM 访存带宽
- comm_time: 通信带宽
```

#### 2.2 模块层 (modules/)
继承 `ModuleBase`，组合多个算子：
```python
class ModuleXXX(ModuleBase):
    def __init__(self, ...):
        super().__init__(...)
        self.add_layer('layer1', LayerXXX(...))
        self.add_layer('layer2', LayerYYY(...))
```

考虑计算与通信的掩盖（overlap）。

#### 2.3 模型层 (models/)
继承 `InferenceBase`，组合多个模块：
- Embedding
- N × TransformerBlock (Attention + FFN/MoE)
- LM Head
- MTP 模块（投机解码）

Prefill 和 Decode 分开建模。

### 阶段 3: 性能计算

运行仿真：
```bash
python main.py \
  --hardware configs/hardware/xxx.json \
  --model configs/models/xxx.json \
  --quant configs/quantization/xxx.json \
  --deploy configs/deployment/xxx.json
```

输出文件：
- `outputs/op_details.csv`: 算子级详细信息
- `outputs/single_card_perf.json`: 单卡性能指标
- `outputs/system_perf.json`: 系统级性能指标

### 阶段 4: 性能分析

分析结果文件，识别瓶颈：
- **CUBE占比低** (<10%): 计算资源未充分利用
- **Memory占比高** (>40%): 访存瓶颈，考虑量化或优化访存模式
- **Comm占比高** (>30%): 通信瓶颈，调整并行策略
- **Vector占比**: 通常很小，如果高则检查算子实现

### 阶段 5: 系统寻优（可选）

搜索最优并行策略：
- 搜索空间: EP ∈ [1,2,4,8,...], TP ∈ [1,2,4,8,...]
- 约束: TPOT ≤ threshold
- 目标: 最大化吞吐量
- 使用多线程并行搜索

## 重要原则

### Transformer 层的完整结构

**DeepSeek V3.x 的关键配置**:
```json
{
  "first_k_dense_replace": 3,  // 前N层使用Dense FFN
  "moe_layer_freq": 1          // MoE层间隔（1=每层都是MoE）
}
```

**层类型判断逻辑**（来自 vLLM）:
```python
def is_moe_layer(layer_idx: int) -> bool:
    return (
        config.num_experts is not None
        and layer_idx >= config.first_k_dense_replace
        and layer_idx % config.moe_layer_freq == 0
    )
```

**DeepSeek V3.2 的层分布** (61层):
- Layer 0, 1, 2: **Dense FFN** (3层)
- Layer 3 ~ 60: **MoE** (58层)

### Dense FFN 算子 (前 first_k_dense_replace 层)

| 序号 | 算子名称 | 类型 | 触发条件 | 说明 |
|------|----------|------|----------|------|
| 1 | `gate_proj` | 计算 | 始终 | hidden -> intermediate (ColumnParallel + SiLU) |
| 2 | `up_proj` | 计算 | 始终 | hidden -> intermediate (ColumnParallel) |
| 3 | `down_proj` | 计算 | 始终 | intermediate -> hidden (RowParallel) |
| 4 | `allreduce` | 通信 | attention_tp > 1 | TP AllReduce |

**Dense FFN FLOPs** (每层):
```
Gate: hidden * (intermediate / tp) * 1 * batch * seq * 2
Up:   hidden * (intermediate / tp) * 1 * batch * seq * 2
Down: hidden * (intermediate / tp) * 2 * batch * seq * 2
Total: 4 * hidden * (intermediate / tp) * batch * seq
```

**Dense FFN 使用 `intermediate_size` (18432)，MoE 使用 `moe_intermediate_size` (2048)**

### Attention 层通信算子（可复用模块）

**设计原则**: 通信算子与具体 Attention 实现解耦，MLA/DSA/GQA/MHA 复用同一套通信模块。

**代码实现**: `modules/module_attention_comm.py` 中的 `ModuleAttentionTPComm` 统一管理：
- `ModuleAttentionAllGatherTP`: AllGather TP 通信
- `ModuleAttentionReduceScatterTP`: ReduceScatter TP 通信
- `ModuleAttentionAllReduceTP`: AllReduce TP 通信
- `ModuleAttentionTPComm`: 统一入口，根据 upstream/downstream TP 自动选择通信模式

**通信算子序列**（按执行顺序）：

| 序号 | 算子名称 | 触发条件 | 位置 | 说明 |
|------|----------|----------|------|------|
| 1 | `allgather_input` | upstream_tp < attention_tp | Attention 前 | AllGather 恢复完整激活 |
| 2 | `allreduce_output` | attention_tp > 1 | o_proj 后 | RowParallel 聚合各 head 结果 |
| 3 | `reduce_scatter_output` | attention_tp > downstream_tp | Attention 后 | ReduceScatter 切分激活给下游 |

**upstream_tp / downstream_tp 的确定逻辑**（在模型层计算）：
```python
for layer_idx in range(num_layers):
    is_moe = self._is_moe_layer(layer_idx)
    downstream_tp = moe_tp if is_moe else attention_tp
    attn_module = ModuleDSAAttention(..., upstream_tp=attention_tp, downstream_tp=downstream_tp)
```
- 上游（前一层的FFN输出）的 TP 度 = attention_tp（因为 Norm 在 attention 内部）
- 下游（后一层的FFN）的 TP 度 = moe_tp（MoE层）或 attention_tp（Dense层）

**TP 通信量公式**:
```
AllGather:     data / (bw * N)
AllReduce:     2 * data / (bw * N)
ReduceScatter: data / (bw * N)
其中 data = batch * seq * hidden * act_bytes
```

### 通信算子完整列表

| 算子 | 类 | 场景 | 时延公式 |
|------|-----|------|----------|
| `AllReduce` | Ring | TP聚合 (Attention/FFN) | `2 * data / (bw * N)` |
| `AllGather` | Ring | 激活恢复 (MoE前) | `data / (bw * N)` |
| `ReduceScatter` | Ring | 激活切分 (MoE后) | `data / (bw * N)` |
| `All2All` | EP | 专家路由 (dispatch/combine) | `data / bw + rtt * sqrt(N-1)` |
| `P2P` | Point-to-Point | PP stage间通信 | `data / bw + rtt + static` |

**PP P2P 通信详解**:
- **触发条件**: PP > 1 时，每个 stage 边界
- **通信量**: `batch * seq * hidden * act_bytes` (单向传输)
- **时延**: `data / bw + rtt_overhead + static_overhead`
- **带宽选择**: 同节点用 intra_node 带宽，跨节点用 inter_node 带宽
- **与 TP 通信的区别**: P2P 是单向传输，不需要多轮通信

**PP 模型构建原则**:
```
Stage 0:     Embedding + layer[0:N/PP]       + P2P send
Stage i:     layer[i*N/PP:(i+1)*N/PP]       + P2P send
Stage PP-1:  layer[(PP-1)*N/PP:N]           + LM Head + MTP
```
- 每个 PP stage 只构建 `num_layers / PP` 层
- FLOPs 自然减少（因为构建的层少了）
- 非最后一个 stage 需要添加 P2P 通信算子
- Pipeline bubble 通过 `pipeline_bubble_rate` 配置量化

### MoE 算子完整列表

DeepSeek-V3 MoE 的完整算子序列（按执行顺序）：

| 序号 | 算子名称 | 类型 | 触发条件 | 说明 |
|------|----------|------|----------|------|
| 1 | `e_topk_weight` | 计算 | 始终 | Gate routing，hidden -> n_experts，sigmoid + topk |
| 2 | `allgather_moe_tp` | 通信 | attention_tp < moe_tp | AllGather 恢复完整激活 |
| 3 | `share_up` | 计算 | n_shared > 0 | Shared expert up 投影 |
| 4 | `share_gate_proj` | 计算 | n_shared > 0 | Shared expert gate 投影 + SiLU |
| 5 | `share_down` | 计算 | n_shared > 0 | Shared expert down 投影 |
| 6 | `dispatch` | 通信 | EP > 1 | All-to-All dispatch (EP通信) |
| 7 | `moe_up` | 计算 | 始终 | Routed expert up 投影 |
| 8 | `moe_gate_proj` | 计算 | 始终 | Routed expert gate 投影 + SiLU |
| 9 | `moe_down` | 计算 | 始终 | Routed expert down 投影 |
| 10 | `combine` | 通信 | EP > 1 | All-to-All combine (EP通信) |
| 11 | `reduce_scatter_moe_tp` | 通信 | attention_tp > moe_tp | ReduceScatter 切分激活 |

**dispatch/combine 通信量公式**:
```
data_size = micro_bs * seq_len * top_k * hidden_size / (moe_tp * EP) * act_bytes
```
- dispatch 和 combine 通信量相同（方向相反）

### MoE FLOPs 公式汇总 (单卡)

**重要概念: moe_batch_size 和 gate_batch_size**

MoE层涉及两种不同的batch size：

```
# Router Gate (专家路由)
gate_batch_size = micro_batch_size / attention_tp

# Expert内部投影 (GateProj/Up/Down)
moe_batch_size = micro_batch_size / attention_tp * moe_tp
```

**为什么Gate的batch要除以attention_tp？**

1. `micro_batch_size` 是每个TP组处理的batch大小
2. 在TP组内，每个芯片独立计算Gate（Gate是replicated linear）
3. 每个芯片实际处理的有效batch = `micro_batch_size / attention_tp`
4. 与Attention层不同，Gate不在TP维度上聚合，每个TP rank独立输出路由结果

**为什么Expert投影的batch要乘以moe_tp？**

当 `attention_tp` 和 `moe_tp` 不同时：
- `attention_tp < moe_tp`: MoE需要先AllGather恢复完整激活，每个MoE TP rank处理更多数据
- `attention_tp > moe_tp`: MoE激活按moe_tp切分，每个rank处理多个attention分片
- `attention_tp = moe_tp`: 直接透传，batch不变

**当 moe_tp = 1 时，两者相等**：
```
gate_batch_size = moe_batch_size = micro_batch_size / attention_tp
```

**Gate (Router) FLOPs**:
```
FLOPs = gate_batch_size * seq * hidden * num_experts * 2
      = (micro_bs / attn_tp) * seq * hidden * num_experts * 2
```
- Gate是replicated linear，计算路由分数给所有num_experts个专家
- num_experts: 路由专家总数（如256）
- 2: FLOPs系数

**重要：Router Gate vs Expert Gate 的区别**

MoE中有两种"Gate"，容易混淆：

| 名称 | 类 | 功能 | 矩阵形状 | FLOPs公式 |
|------|-----|------|----------|-----------|
| **Router Gate** | `LayerMoEGate` | 计算所有专家的路由分数 | `[hidden, num_experts]` | `2 * batch * seq * hidden * num_experts` |
| **Expert Gate** | `LayerMoEGateProj` | SwiGLU的gate分支投影 | `[hidden, intermediate]` | `2 * batch * seq * top_k * hidden * intermediate` |

**Router Gate** (LayerMoEGate):
- 输入: `[batch, seq, hidden]`
- 权重: `[hidden, num_experts]` = `[7168, 256]`
- 输出: `[batch, seq, num_experts]` (每个token对256个专家的分数)
- batch = `micro_bs / attn_tp`

**Expert Gate** (LayerMoEGateProj):
- 这是SwiGLU中的gate分支，在每个选中的专家内部执行
- 输入: `[token_expert_pairs, hidden]` (token_expert_pairs = batch * seq * top_k)
- 权重: `[hidden, intermediate]` = `[7168, 2048]` (每个专家一份)
- 输出: `[token_expert_pairs, intermediate]`
- batch = `micro_bs / attn_tp * moe_tp`

**为什么Expert Gate用 `intermediate` 而不是 `num_experts`？**
- Router Gate: 需要对所有256个专家打分，所以输出维度是 `num_experts`
- Expert Gate: 每个专家内部的投影，输出维度是 `intermediate` (SwiGLU的中间维度)
- 这是两个完全不同的操作！

**数值对比** (DeepSeek-V3, batch=1, seq=8192):
- Router Gate FLOPs: `2 * 1 * 8192 * 7168 * 256` = 3.01e+10
- Expert Gate FLOPs: `2 * 1 * 8192 * 8 * 7168 * 2048` = 1.92e+12
- Expert Gate是Router Gate的64倍！因为 `intermediate * top_k = 2048 * 8 = 16384` >> `num_experts = 256`

**Gate投影 (ColumnParallel, SwiGLU的Gate分支)**:
```
FLOPs = hidden_size * (moe_intermediate_size / moe_tp) * 1 * top_k * moe_batch_size * seq_len * 2
```
- `* 1`: 单分支 (Gate)
- `* 2`: FLOPs 系数
- Vector: SiLU 激活 (~4 FLOPs per element)

**Up投影 (ColumnParallel, SwiGLU的Up分支)**:
```
FLOPs = hidden_size * (moe_intermediate_size / moe_tp) * 1 * top_k * moe_batch_size * seq_len * 2
```
- `* 1`: 单分支 (Up)
- `* 2`: FLOPs 系数
- 无 Vector 操作

**Down投影 (RowParallel)**:
```
FLOPs = hidden_size * (moe_intermediate_size / moe_tp) * 2 * top_k * moe_batch_size * seq_len * 2
```
- 第一个 `* 2`: FLOPs 系数
- 第二个 `* 2`: SwiGLU 结构因子 (Down 的输入是 silu(gate)*up，融合后维度为 intermediate*2)

**Gate+Up 融合 (可选优化)**:
```
FLOPs = hidden_size * (moe_intermediate_size / moe_tp) * 2 * top_k * moe_batch_size * seq_len * 2
       = 2 * hidden * (intermediate / moe_tp) * top_k * moe_batch_size * seq * 2
```
- 融合后计算量等于分开计算的总和

**Shared Expert**: top_k = 1，公式与 Routed Expert 相同但 top_k 替换为 1

### EP 不影响 FLOPs 的原则

**关键**: EP (Expert Parallel) 只影响通信和权重存储，不影响单卡 FLOPs！

```python
# ❌ 错误：EP 减少单卡 FLOPs
expert_tokens = batch * seq * top_k / ep  # 错误地除以 EP
flops = 2 * expert_tokens * intermediate * hidden

# ✅ 正确：EP 不影响 FLOPs
flops = 2 * batch * seq * top_k * intermediate * hidden  # 不除以 EP
```

**原因**: EP 将专家分布到不同节点，每个 token 仍然需要经过 top_k 个专家的计算。EP 只是通过 All-to-All 将 token 分发到对应专家所在的节点，总计算量不变。EP 的影响体现在：
1. **通信开销**: All-to-All dispatch/combine 的时延
2. **权重存储**: 每个节点只存储 `num_experts / EP` 个专家的权重（访存量减少）

### DSA (DeepSeek Sparse Attention) 建模

**关键**: DSA 的性能不是简单用 `sparse_ratio × seq_len` 计算！

正确做法：
1. 查看 model_config 中的 `topk_tokens` 参数（绝对值，如 256, 512）
2. Prefill 阶段: 完整 attention，计算量 = seq_len × kv_seq_len
3. Decode 阶段: 稀疏 attention，计算量 = seq_len × topk_tokens
4. 参考 vllm 中的 `SparseAttnIndexer` 实现

**错误示例**:
```python
# ❌ 错误：使用比例
effective_kv_len = int(kv_seq_len * 0.25)
```

**正确示例**:
```python
# ✅ 正确：使用绝对值
topk_tokens = model_config.topk_tokens  # 如 256
if is_prefill:
    effective_kv_len = kv_seq_len
else:
    effective_kv_len = topk_tokens
```

#### DSA 完整算子序列（DeepSeek-V3.2）

DSA = MLA + Lightning Indexer，完整算子序列如下：

| 序号 | 算子名称 | 类型 | TP | 触发条件 | 说明 |
|------|----------|------|-----|----------|------|
| 1 | `allgather_input` | 通信 | ✓ | upstream_tp < attention_tp | AllGather恢复完整激活 |
| 2 | `input_norm` | Vector | - | 始终 | RMSNorm(hidden_size) |
| 3 | `q_a_proj` | CUBE | ✓ | 始终 | hidden → q_lora_rank (ColumnParallel) |
| 4 | `q_a_norm` | Vector | ✓ | 始终 | RMSNorm(q_lora_rank/TP) |
| 5 | `q_b_proj` | CUBE | ✓ | 始终 | q_lora_rank → num_heads × qk_head_dim (ColumnParallel) |
| 6 | `kv_a_proj` | CUBE | - | 始终 | hidden → kv_lora_rank + qk_rope (ReplicatedLinear) |
| 7 | `kv_a_norm` | Vector | - | 始终 | RMSNorm(kv_lora_rank) |
| **8** | `indexer_wq_b` | CUBE | **✗** | **Decode only** | q_lora_rank → index_n_heads × index_head_dim (ReplicatedLinear) |
| **9** | `indexer_wk` | CUBE | **✗** | **Decode only** | hidden → index_head_dim (ReplicatedLinear) |
| **10** | `indexer_k_norm` | Vector | **✗** | **Decode only** | LayerNorm(index_head_dim) |
| **11** | `indexer_weights_proj` | CUBE | **✗** | **Decode only** | hidden → index_n_heads (ReplicatedLinear, **无量化**) |
| **12** | `sparse_attn_indexer` | CUBE+Vector | **✗** | **Decode only** | FP8 MQA + TopK选择 (Lightning Indexer核心) |
| 13 | `kv_b_proj` | CUBE | ✓ | 始终 | kv_lora_rank → num_heads × (qk_nope + v_head) (ColumnParallel) |
| 14 | `dsa_attention` | CUBE | ✓ | 始终 | FlashMLA稀疏注意力 (使用TopK索引) |
| 15 | `o_proj` | CUBE | ✓ | 始终 | num_heads/TP × v_head_dim → hidden (RowParallel) |
| 16 | `allreduce_output` | 通信 | ✓ | attention_tp > 1 | AllReduce聚合输出 |
| 17 | `reduce_scatter_output` | 通信 | ✓ | attention_tp > downstream_tp | ReduceScatter切分给下游 |

**关键特性**：
- **Lightning Indexer (算子8-12)**: 只在Decode阶段运行，用于选择topK个重要token
- **Indexer无TP**: 所有Indexer线性层使用ReplicatedLinear，不在TP维度切分
- **Indexer无量化**: `indexer_weights_proj`不使用量化 (quant_config=None)
- **FP8 MQA**: Indexer使用FP8精度的Multi-Query Attention进行稀疏选择
- **独立K cache**: Indexer维护独立的FP8 K cache，与主MLA KV cache分离

#### Lightning Indexer 算子详解

**1. `indexer_wq_b`** (index_linear1)
- 输入: q_c (q_lora_rank维度)
- 输出: indexer_q (index_n_heads × index_head_dim维度)
- FLOPs: `2 × batch × seq × q_lora_rank × (index_n_heads × index_head_dim)`
- 无TP切分

**2. `indexer_wk`** (index_linear2)
- 输入: hidden_states (hidden_size维度)
- 输出: indexer_k (index_head_dim维度)
- FLOPs: `2 × batch × seq × hidden_size × index_head_dim`
- 无TP切分

**3. `indexer_k_norm`**
- LayerNorm(index_head_dim)，eps=1e-6
- Vector FLOPs: `~5 × batch × seq × index_head_dim`

**4. `indexer_weights_proj`** (index_score)
- 输入: hidden_states
- 输出: weights (index_n_heads维度)
- FLOPs: `2 × batch × seq × hidden_size × index_n_heads`
- **无量化** (FP32计算)

**5. `sparse_attn_indexer`** (核心操作)
- FP8量化K并写入Indexer K cache
- 计算FP8 MQA: indexer_q @ indexer_k^T
- TopK选择: 返回topk_indices
- CUBE FLOPs: `2 × batch × seq × index_n_heads × kv_seq_len × index_head_dim`
- Vector FLOPs: FP8量化 + Softmax + TopK

#### vLLM代码参考

```python
# vllm/model_executor/models/deepseek_v2.py
class Indexer(nn.Module):
    def __init__(self, config, q_lora_rank, ...):
        self.topk_tokens = config.index_topk        # 2048
        self.n_head = config.index_n_heads          # 64
        self.head_dim = config.index_head_dim       # 128

        # 所有层都是ReplicatedLinear，无TP
        self.wq_b = ReplicatedLinear(q_lora_rank, head_dim * n_head)
        self.wk = ReplicatedLinear(hidden_size, head_dim)
        self.k_norm = LayerNorm(head_dim, eps=1e-6)
        self.weights_proj = ReplicatedLinear(hidden_size, n_head, quant_config=None)  # 无量化

    def forward(self, hidden_states, q_c, positions, rope):
        indexer_q = self.wq_b(q_c)[0]               # indexer_wq_b
        indexer_k = self.wk(hidden_states)[0]       # indexer_wk
        indexer_k = self.k_norm(indexer_k)          # indexer_k_norm
        indexer_q, indexer_k = self.rope(...)       # RoPE
        indexer_k = self.fp8_quant(indexer_k)       # FP8量化
        weights = self.weights_proj(hidden_states)  # indexer_weights_proj
        topk_indices = self.indexer_op(...)         # sparse_attn_indexer
        return topk_indices
```

#### 经验教训

1. **不要遗漏Lightning Indexer**: 早期实现只建模了FlashMLA部分，忽略了整个Indexer模块
2. **Indexer无TP**: Indexer的所有线性层都是ReplicatedLinear，不参与TP切分
3. **Indexer无量化**: `weights_proj`层不使用量化，保持FP32精度
4. **LayerNorm vs RMSNorm**: Indexer的`k_norm`使用LayerNorm，不是RMSNorm
5. **FP8 K cache**: Indexer维护独立的FP8 K cache，与MLA的KV cache分离
6. **只在Decode运行**: Indexer只在Decode阶段启用，Prefill使用完整attention

### MLA (Multi-head Latent Attention) 建模

关键优化：
- KV cache 存储压缩后的 latent (kv_lora_rank 维度)
- 而不是完整的 KV (num_heads × head_dim)
- 访存量大幅减少

### 代码生成原则

1. **继承现有 Base 类**: LayerBase, ModuleBase, InferenceBase
2. **参考 references/ 目录**: 查看示例代码
3. **验证 vllm 实现**: 确保与 vllm 的模型定义一致
4. **配置驱动**: 所有参数从配置文件读取，不硬编码

### 性能指标计算（QPS / TPS / TPOT）

#### 核心概念

- **micro_batch_size**: 每个TP组处理的batch大小（配置文件中的值）
- **total_bs**: 全局batch大小 = micro_batch_size × num_tp_groups
- **num_tp_groups**: total_chips / attention_tp / context_parallel
- **num_cards**: total_chips / chips_per_card
- **TPOT**: 生成单个token的时间（per-request, ms）
- **TTFT**: Prefill首token时延（per-request, ms）

#### Per-Card 性能公式

**Decode**:
```
tps_per_card = total_bs / tpot_s / num_cards
qps_per_card = tps_per_card / output_length
```

**Prefill**:
```
qps_per_card = total_bs / ttft_s / num_cards
tps_per_card = qps_per_card × input_length
```

**说明**:
- `tpot_s` = TPOT / 1000（秒）
- `ttft_s` = TTFT / 1000（秒）
- Decode时 QPS < TPS（每个请求需要生成output_length个token）
- Prefill时 TPS > QPS（每个请求处理input_length个token）

#### System 性能公式

```
system_tps = tps_per_card × num_cards = total_bs / tpot_s
system_qps = qps_per_card × num_cards = total_bs / tpot_s / output_length
```

#### 示例验证

配置: total_chips=16, chips_per_card=2, TP=1, micro_batch=1, output_length=128
- num_cards = 16/2 = 8
- num_tp_groups = 16/1/1 = 16
- total_bs = 1 × 16 = 16
- tpot = 36.3ms

Per-Card:
- tps_per_card = 16 / 0.0363 / 8 = 55.09 tokens/s
- qps_per_card = 55.09 / 128 = 0.43 req/s

System:
- system_tps = 55.09 × 8 = 440.76 tokens/s
- system_qps = 0.43 × 8 = 3.44 req/s

#### 常见错误

```python
# ❌ 错误1：混淆TPS和QPS，认为Decode时QPS=TPS
# 实际上 Decode时 QPS = TPS / output_length

# ❌ 错误2：TPOT没有考虑batch_size
tpot = total_time  # 错误，total_time包含batch维度
# 正确：tpot = total_time / micro_batch_size

# ❌ 错误3：系统吞吐计算时乘以num_tp_groups而不是num_cards
system_tps = tps_per_card * num_tp_groups  # 错误
# 正确：system_tps = tps_per_card * num_cards
# 或者直接用公式：system_tps = total_bs / tpot_s

# ❌ 错误4：Gate (Router) 的batch直接使用micro_batch_size
# Gate是replicated linear，每个TP rank独立计算，有效batch = micro_bs / attn_tp
gate_flops = 2 * micro_bs * seq * hidden * num_experts  # 错误
# 正确：
gate_flops = 2 * (micro_bs / attn_tp) * seq * hidden * num_experts

# ❌ 错误5：混淆 Router Gate 和 Expert Gate 的公式
# Router Gate (LayerMoEGate): 输出维度是 num_experts (256)
# Expert Gate (LayerExpertGateProj): 输出维度是 intermediate (2048)
# 两者完全不同！
expert_gate_flops = 2 * batch * seq * hidden * num_experts  # 错误！这是Router的公式
# 正确：
expert_gate_flops = 2 * batch * seq * top_k * hidden * intermediate

# ❌ 错误6：认为 share_up 和 share_gate_proj 的 CUBE FLOPs 不同
# 实际上它们相同！因为都是 hidden -> intermediate 的投影
# 区别仅在于 Vector FLOPs（Gate 有 SiLU，Up 没有）

# ❌ 错误7：遗漏 Attention 前后的 TP 通信算子
# forward流程: embedding -> attention -> moe -> lm_head
# 当 attention_tp ≠ moe_tp 时，需要通信算子进行 TP 级别转换

# 错误示例：模型中只有计算算子，没有通信算子
class Model:
    def build(self):
        self.add_module('attention', Attention(...))  # 缺少通信
        self.add_module('moe', MoE(...))  # 缺少通信

# 正确示例：根据 TP 配置添加通信算子
class ModuleDSAAttention:
    def _build_layers(self):
        # 1. AllGather (上游TP < 当前TP)
        if self.upstream_tp < self.attention_tp:
            self.add_layer('allgather_input', LayerAllGather(..., self.attention_tp))

        # ... attention 计算 ...

        # 2. AllReduce (o_proj后，TP > 1)
        if self.attention_tp > 1:
            self.add_layer('allreduce_output', LayerAllReduce(...))

        # 3. ReduceScatter (当前TP > 下游TP)
        if self.attention_tp > self.downstream_tp:
            self.add_layer('reduce_scatter_output', LayerReduceScatter(..., self.downstream_tp))

# ❌ 错误8：MoE 模块没有处理 TP 级别转换
# MoE 输入/输出 TP 级别可能不同于 attention_tp

# 正确示例：MoE 模块的 TP 通信
class ModuleMoE:
    def _build_layers(self):
        # 1. Pre-MoE: attention_tp < moe_tp 时需要 AllGather
        if self.attention_tp < self.moe_tp:
            self.add_layer('allgather_moe_tp', LayerAllGather(..., self.attention_tp))

        # ... MoE 计算 (dispatch, experts, combine) ...

        # 2. Post-MoE: attention_tp > moe_tp 时需要 RS + AG
        if self.attention_tp > self.moe_tp:
            self.add_layer('reduce_scatter_moe_tp', LayerReduceScatter(..., self.moe_tp))
            self.add_layer('allgather_restore', LayerAllGather(..., self.moe_tp))

        # 3. Post-MoE: attention_tp < moe_tp 时需要 RS
        if self.attention_tp < self.moe_tp:
            self.add_layer('reduce_scatter_restore', LayerReduceScatter(..., self.moe_tp))

# ❌ 错误9：认为 Embedding 后需要 AllGather 才能做 Attention
# VocabParallelEmbedding 内部已有 AllReduce，输出是 replicated 状态
# 第一层 Attention 的 upstream_tp = attention_tp 是正确的

# 错误思路：Embedding 用 lm_head_tp 切分 vocab，输出应该是 lm_head_tp 级别
# → 需要 AllGather 才能给 Attention 用
# 正确理解：VocabParallelEmbedding.forward() 最后调用 tensor_model_parallel_all_reduce()
# → 输出是 replicated（所有 TP rank 都有完整 hidden states）
# → 第一层 ColumnParallelLinear (QKV) 期望 replicated 输入，直接可用

# ❌ 错误10：LayerEmbedding 没有建模内部 AllReduce 的通信代价
# 当 lm_head_tp > 1 时，VocabParallelEmbedding 内部有 AllReduce 通信
# 这个通信代价需要在 LayerEmbedding 中建模

# 正确示例：LayerEmbedding 实现 get_comm_bytes() 和 get_comm_time()
class LayerEmbedding(LayerBase):
    def get_comm_bytes(self):
        if self.lm_head_tp <= 1:
            return 0.0
        # Ring All-Reduce: 2 * (N-1) / N * data_size
        return 2 * (self.lm_head_tp - 1) / self.lm_head_tp * self.allreduce_data_size

    def get_comm_time(self):
        if self.lm_head_tp <= 1:
            return 0.0
        # Ring All-Reduce: 2 * data_size / (bw * N) + overhead
        return transfer_time_ms + rtt_overhead_ms + static_overhead_ms

# ❌ 错误11：Down 投影没有 Vector FLOPs（遗漏 SwiGLU 激活）
# SwiGLU 的激活操作 (SiLU + multiply) 需要建模

# 错误：认为 down 只是 matmul，没有 vector 操作
class LayerExpertDown(LayerBase):
    def get_vector_flops(self):
        return 0.0  # 错误！

# 正确：融合内核视角，所有 SwiGLU 激活在 Down 层建模
class LayerExpertDown(LayerBase):
    def get_vector_flops(self):
        # SwiGLU: silu(gate) * up = 7 FLOPs/element
        token_expert_pairs = moe_batch_size * seq_len * top_k
        return token_expert_pairs * intermediate_per_tp * 7

# ❌ 错误12：SwiGLU 激活 FLOPs 在 gate_proj 和 down_proj 重复计算
# gate_proj 算了 SiLU，down_proj 又算了 SiLU + multiply → 双倍计算！

# 错误示例：
class LayerExpertGateProj(LayerBase):
    def get_vector_flops(self):
        return token_experts * intermediate_per_tp * 4  # SiLU

class LayerExpertDown(LayerBase):
    def get_vector_flops(self):
        return token_experts * intermediate_per_tp * 7  # 又算了一遍！

# 正确：统一在 Down 层建模，gate_proj 的 vector 设为 0
class LayerExpertGateProj(LayerBase):
    def get_vector_flops(self):
        return 0.0  # 激活已归入 Down 层

class LayerExpertDown(LayerBase):
    def get_vector_flops(self):
        return token_experts * intermediate_per_tp * 7  # 所有 SwiGLU 激活

# ❌ 错误13：Vector 算力根据 act_compute_bits 选择 FP16/FP32
# 实际上 Vector 单元统一使用 FP16，与 activation 计算精度无关

# 错误示例：
def get_vector_time(self):
    if self.act_compute_bits <= 16:
        tflops = self.hardware_config.vector_tflops_fp16
    else:
        tflops = self.hardware_config.vector_tflops_fp32  # 错误！
    return flops / (tflops * 1e12) * 1000

# 正确：Vector 单元统一使用 FP16 算力
def get_vector_time(self):
    tflops = self.hardware_config.vector_tflops_fp16  # 始终 FP16
    return flops / (tflops * 1e12) * 1000 / vector_utilization

# ❌ 错误14：PP P2P 通信算子缺失
# PP stage 之间需要 P2P 通信传递激活，缺失会导致总时间计算不准确

# 错误示例：模型只构建了计算层，没有 PP 通信
class Model:
    def _build_modules(self):
        for layer_idx in range(num_layers):  # 构建所有层，忽略 PP
            self.add_module(f'layer_{layer_idx}', ...)

# 正确：PP > 1 时，每个 stage 只构建部分层，并添加 P2P 通信
class Model:
    def __init__(self, ..., pp_stage=0):
        self.pp = deploy_config.pipeline_parallel
        self.pp_stage = pp_stage
        self.layers_per_stage = num_layers // self.pp
        self.start_layer = pp_stage * self.layers_per_stage
        self.end_layer = self.start_layer + self.layers_per_stage

    def _build_modules(self):
        # 只构建当前 stage 的层
        for layer_idx in range(self.start_layer, self.end_layer):
            self.add_module(f'layer_{layer_idx}', ...)

        # 非最后一个 stage 添加 P2P 发送
        if self.pp_stage < self.pp - 1:
            data_size = batch * seq * hidden * act_bytes
            self.add_module('p2p_send', LayerP2P(..., data_size))

# ❌ 错误15：EP 未影响专家权重访存
# EP 切分专家，每个 EP rank 只存储 1/EP 的专家权重

# 错误示例：
class LayerExpertUp(LayerBase):
    def get_mem_bytes(self):
        read_weight = hidden * intermediate_per_tp * weight_bytes  # 未除以 EP！

# 正确：EP 减少权重存储
class LayerExpertUp(LayerBase):
    def get_mem_bytes(self):
        # 每个 EP rank 只存储 1/EP 的专家权重
        read_weight = hidden * intermediate_per_tp * weight_bytes / self.ep

# ❌ 错误16：PP bubble 未在性能计算中体现
# PP 引入 pipeline bubble，降低有效吞吐

# 错误示例：直接使用 total_time 计算 TPOT
tpot = total_time / micro_batch_size

# 正确：考虑 bubble 开销
if pp > 1:
    bubble_rate = deploy_config.pipeline_bubble_rate  # 如 0.1 (10% bubble)
    effective_time = total_time * (1 + bubble_rate)
else:
    effective_time = total_time
tpot = effective_time / micro_batch_size
```

# ❌ 错误17：EP All-to-All 通信量公式错误
# MoE 的 dispatch/combine 通信量需要基于理论推导

# 错误示例1：只考虑了 tokens，没有考虑 top_k 和 EP 分布
if self.ep > 1:
    data_size = batch_size * seq_len * hidden_size * act_bytes  # 缺少 top_k！

# 错误示例2：错误地使用 moe_tp 和简单的 1/EP 除法
if self.ep > 1:
    tokens_per_moe_rank = batch_size * seq_len / moe_tp  # 错误！EP 通信在 TP 之后
    data_size = tokens_per_moe_rank * top_k / ep * hidden * act_bytes

# ✅ 正确：基于理论推导的 Per-rank 通信量
if self.ep > 1:
    max_chips = hardware_config.max_chips_per_node
    if moe_tp >= max_chips:
        # EP 通信在节点内（NVLink），延迟可忽略
        data_size_dispatch = 0.0
    else:
        # 参与EP通信的token数 = micro_bs / attn_tp（EP dispatch在TP处理之后）
        tokens = max(batch_size * seq_len / attention_tp, 1)
        # Per-rank EP通信量 = tokens × top_k × (EP-1) / EP × hidden × dtype
        data_size_dispatch = tokens * top_k * (ep - 1) / ep * hidden_size * act_bytes
```

**EP All-to-All 通信量公式推导**:

1. **前提条件**：每个 token 路由到 `top_k` 个专家
2. **专家分布**：专家均匀分布在 `EP` 个 rank 上
3. **跨 EP 通信**：每个 token 期望有 `top_k × (EP-1)/EP` 个跨 EP 专家
4. **融合 dispatch**：每个 token 只发送一次到每个远程 rank（不重复发送）
5. **Per-rank volume** = `tokens × top_k × (EP-1) / EP × hidden × dtype`

| 参数 | 正确用法 | 说明 |
|------|----------|------|
| `tokens` | `micro_bs / attn_tp × seq_len` | EP dispatch 在 TP 处理之后，token 数按 attn_tp 切分 |
| `top_k` | 乘以 | 每个 token 路由到 top_k 个专家 |
| `(EP-1)/EP` | 乘以 | 跨 EP 通信比例 |
| `moe_tp` | **不参与** | moe_tp 在 dispatch 之后才扩展激活 |
| `max_chips_per_node` | 条件判断 | 当 moe_tp ≥ max_chips 时，EP 在节点内，通信免费 |

**为什么 tokens = micro_bs / attn_tp 而不是 micro_bs / moe_tp？**
- EP dispatch 发生在 TP 处理之后、moe_tp 扩展之前
- 此时 token 数按 attention_tp 切分
- moe_tp 的扩展是在 dispatch 之后通过 AllGather 完成

**为什么是 `(EP-1)/EP` 而不是 `1/EP`？**
- 融合 dispatch：每个 token 发送到所有需要跨 EP 通信的 rank
- 平均每个 token 有 `top_k × (EP-1)/EP` 个专家在其他 EP rank
- 所以 per-rank 发送量 = `tokens × top_k × (EP-1)/EP × hidden`

# ❌ 错误18：未利用 DeepEP 的 Compute-Communication Overlap
# MoE 的 dispatch 通信可与 Shared Expert 计算完全重叠

# 错误示例：顺序执行 dispatch 和 shared expert
moe_time = dispatch_time + shared_expert_time + routed_expert_time + combine_time

# ✅ 正确：DeepEP Hook 机制允许通信与计算重叠
# effective_time = max(dispatch_time, shared_expert_time) + routed_expert_time + combine_time
if overlap_enabled and shared_expert_time > 0:
    # DeepEP: dispatch 通信在后台执行，与 shared expert 计算并行
    dispatch_effective = max(dispatch_time, shared_expert_time)
else:
    dispatch_effective = dispatch_time
moe_time = dispatch_effective + routed_expert_time + combine_time
```

**DeepEP Compute-Communication Overlap 原理**:
1. **Hook 机制**: dispatch 调用返回一个 hook，不阻塞计算
2. **零 SM 占用**: 通信由 RDMA 网络接口完成，不占用 GPU SM
3. **自动重叠**: 当 `compute_time >= comm_time` 时，通信完全隐藏

**重叠效率公式**:
```
overlap_efficiency = min(overlapable_compute_time / comm_time, 1.0)
hidden_comm_time = comm_time * (1 - overlap_efficiency)
effective_comm_time = max(comm_time, overlapable_compute_time)
```

**典型场景**:
- Dispatch (通信) 与 Shared Expert (计算) 重叠
- 当 Shared Expert 计算时间 >= Dispatch 通信时间时，Dispatch 完全隐藏
- 实测重叠效率约 85-95%

# ❌ 错误19：Compute-Communication Overlap 导致 Double-Counting
# 在模块级别处理 overlap 时，必须避免计算时间被重复计算

# 错误示例：计算时间既被添加到 total，又被添加到 effective_comm
accumulated_compute = 0.0
for layer in layers:
    if layer.is_comm_op:
        # 错误：accumulated_compute 已经添加到 total，这里又被用于 effective
        effective = max(comm_time, accumulated_compute)  # shared expert 被 double-count!
        total += effective
    else:
        total += layer_time
        accumulated_compute += layer_time

# ✅ 正确：从 total 中减去已添加的 accumulated_compute
for layer in layers:
    if layer.is_comm_op:
        if overlap_enabled and accumulated_compute > 0:
            total -= accumulated_compute  # 先减去已添加的计算时间
            effective = max(comm_time, accumulated_compute * efficiency)
            total += effective
        else:
            total += comm_time
        accumulated_compute = 0.0
    else:
        total += layer_time
        accumulated_compute += layer_time
```

**Double-Counting 分析**:
```
假设 shared_expert = 0.03 ms, dispatch_comm = 0.05 ms

错误方式:
  total = shared_expert (0.03) + max(dispatch, shared) (0.05) = 0.08 ms
  实际 shared_expert 被计算了两次！

正确方式:
  total = shared_expert (0.03)
  遇到 dispatch: total -= 0.03, total += max(0.05, 0.03*0.9) = 0.05
  最终 total = 0.05 ms (shared_expert 被 dispatch 完全隐藏)
```

# ❌ 错误20：忽略 EP 负载不均衡对计算时间的影响
# EP 的 token 路由不均衡导致不同 rank 的专家计算负载不同

# 错误示例：假设所有 EP rank 负载完全均衡
expert_compute_time = base_compute_time  # 没有考虑负载不均衡

# ✅ 正确：应用 EP 负载不均衡系数
# ep_load_imbalance_factor = max_load / avg_load
# 典型值: 1.0 (完美均衡) ~ 1.5 (严重不均衡)
# DeepSeek-V3 使用 auxiliary loss 控制，典型值 ~1.1
expert_compute_time = base_compute_time * ep_load_imbalance_factor
```

**EP 负载不均衡原理**:
- 每个 token 路由到 top_k 个专家
- 路由决策是动态的，不同 rank 收到的 token 数不同
- 某些热门专家可能收到更多 token
- 最繁忙 rank 的负载 = avg_load × imbalance_factor

**不均衡对性能的影响**:
```
EP 负载不均衡主要影响:
1. 专家计算时间 (share_up, share_gate, share_down, moe_up, moe_gate, moe_down)
2. 不影响通信时间 (通信是同步点，所有 rank 等待最慢的)
3. 当 overlap 开启且 comm_time > compute_time 时，不均衡效应被掩盖
```

**配置参数**:
```json
{
  "ep_load_imbalance_factor": 1.1,  // 负载不均衡系数
  "enable_compute_comm_overlap": true,
  "overlap_efficiency": 0.9
}
```

**实测数据** (batch=64, EP=1, 无EP通信):
| Imbalance Factor | Total Time | 增加 |
|------------------|------------|------|
| 1.0 (均衡) | 0.228 ms | - |
| 1.1 | 0.250 ms | +9.6% |
| 1.2 | 0.271 ms | +18.9% |
| 1.5 | 0.336 ms | +47.4% |

### DeepEP 通信优化技术

**DeepEP** 是 DeepSeek 开源的高性能 MoE 通信库，专门针对 Expert Parallelism (EP) 的 All-to-All 通信进行优化。

**GitHub**: https://github.com/deepseek-ai/DeepEP

#### 核心优化技术

| 技术 | 原理 | 效果 |
|------|------|------|
| **双模式内核** | High-Throughput (Prefill) vs Low-Latency (Decode) | Prefill追求带宽，Decode追求低延迟 |
| **Pure RDMA** | 绕过NCCL，直接使用RDMA P2P传输 | 延迟降低50%+ |
| **Hook机制** | 零SM占用，通信在后台执行 | 通信与计算完全重叠 |
| **NVLink + RDMA混合** | 节点内NVLink，跨节点RDMA | 最大化带宽利用 |
| **FP8通信** | 8-bit量化传输 | 通信量减半 |

#### 延迟模型

**1. High-Throughput模式 (Prefill/Training)**
```python
# 追求高带宽利用率
latency = data_size / bandwidth + rtt_overhead * sqrt(N-1) + static_overhead
```

**2. Low-Latency模式 (Decode)**
```python
# Pure RDMA，延迟增长为 O(log N)
base_latency_us = 50  # RDMA基础延迟 (来自DeepEP实测)
scaling_factor = log2(EP)  # 并发传输，延迟对数增长
latency = base_latency + transfer_time * scaling_factor
```

#### Compute-Communication Overlap

DeepEP 的 Hook 机制允许通信与计算完全重叠：

```python
# 当 compute_time >= comm_time 时，通信完全隐藏
effective_time = max(comm_time, overlapable_compute_time)
```

**典型重叠场景**:
- **Dispatch + Shared Expert**: EP dispatch 通信与 Shared Expert 计算并行
- **Routed Expert + Combine**: Routed Expert 计算与上一层 Combine 通信并行

**重叠效率**:
```
overlap_efficiency = min(overlapable_compute_time / comm_time, 1.0)
effective_comm_time = comm_time * (1 - overlap_efficiency)
```

#### 实测性能数据

**H800 + CX7 400Gb/s (EP=64)**:
- Dispatch latency: ~173 µs
- Combine latency: ~314 µs
- RDMA bandwidth: ~43 GB/s (85% of 50 GB/s theoretical)

**性能对比**:
| 模式 | NCCL All-to-All | DeepEP Low-Latency | 加速比 |
|------|-----------------|-------------------|--------|
| EP=64 | ~500 µs | ~173 µs | 2.9x |
| EP=128 | ~800 µs | ~200 µs | 4.0x |

#### 配置参数

```json
{
  "comm_rdma_bw_gbps": 50.0,        // RDMA带宽 (CX7 400Gb/s = 50 GB/s)
  "comm_rdma_efficiency": 0.85,     // DeepEP实测约85%效率
  "deepep_base_latency_us": 50.0,   // RDMA基础延迟
  "deepep_overlap_efficiency": 0.9  // Compute-Comm overlap效率
}
```

#### 代码示例

```python
# DeepEP Low-Latency模式创建dispatch算子
dispatch = LayerAll2All(
    hardware_config, model_config, deploy_config, quant_config,
    data_size_bytes=dispatch_volume,
    num_devices=EP,
    mode='low_latency',  # Decode阶段使用low_latency
    overlapable_compute_time_ms=shared_expert_time_ms,  # 可重叠的计算时间
    is_cross_node=(EP > max_chips_per_node / moe_tp)
)
```

### Forward 流程的完整通信算子

**核心原则**: 当相邻模块的 TP 级别不同时，必须有通信算子进行 TP 级别转换。

**完整 Forward 流程** (含 PP，以单个 Transformer 层为例):

```
PP Stage 0:
  Embedding (VocabParallel + AllReduce → replicated)
    │
    ▼
  ┌─ Transformer Layer ─────────────────────────────────────┐
  │  ... (同下)                                               │
  └──────────────────────────────────────────────────────────┘
    │
    ▼
  [P2P Send]  ← 传递激活到 Stage 1

PP Stage i (0 < i < PP-1):
  [P2P Recv]  ← 接收来自 Stage i-1 的激活
    │
    ▼
  ┌─ Transformer Layer ─────────────────────────────────────┐
  │                                                          │
  │  [AllGather]          ← upstream_tp < attention_tp 时    │
  │  Attention (at attention_tp)                             │
  │  [AllReduce]          ← attention_tp > 1 时             │
  │  [ReduceScatter]      ← attention_tp > downstream_tp 时  │
  │                                                          │
  │  if MoE layer:                                           │
  │    [AllGather]        ← attention_tp < moe_tp 时         │
  │    Shared Expert (可与 Dispatch 重叠)                    │
  │    [Dispatch]        ← EP All-to-All (与 Shared 重叠)    │
  │    Routed Expert (受 EP 负载不均衡影响)                  │
  │    [Combine]         ← EP All-to-All                     │
  │    [RS + AG]          ← attention_tp > moe_tp 时         │
  │    [ReduceScatter]    ← attention_tp < moe_tp 时         │
  │  else Dense FFN:                                        │
  │    Dense FFN (at attention_tp)                           │
  │    [AllReduce]        ← attention_tp > 1 时             │
  │                                                          │
  └──────────────────────────────────────────────────────────┘
    │
    ▼
  [P2P Send]  ← 传递激活到 Stage i+1

PP Stage PP-1:
  [P2P Recv]  ← 接收来自 Stage PP-2 的激活
    │
    ▼
  ┌─ Transformer Layer ─────────────────────────────────────┐
  │  ... (同上)                                               │
  └──────────────────────────────────────────────────────────┘
    │
    ▼
  LM Head (at lm_head_tp)
  [AllGather]  ← lm_head_tp > 1 时
  MTP Layers (投机解码)
```

**Embedding 输出是 replicated 状态**:

vLLM 的 `VocabParallelEmbedding` 在 forward 最后调用 `tensor_model_parallel_all_reduce`，
输出是 replicated 状态（每个 TP rank 都有完整的 hidden states）。因此：
- 第一层 Attention 的 `upstream_tp = attention_tp`，不需要 AllGather
- `ColumnParallelLinear`（QKV 投影）期望 replicated 输入，可以直接使用
- **但 Embedding 内部的 AllReduce 通信代价需要建模**（在 `LayerEmbedding` 中实现）

**TP 级别转换的通信算子**:

| 转换方向 | 通信算子 | 位置 | 说明 |
|----------|----------|------|------|
| upstream_tp → attention_tp | AllGather | Attention 前 | 恢复完整激活给 Attention |
| attention_tp → attention_tp | AllReduce | Attention 后 (o_proj) | RowParallel 聚合 |
| attention_tp → downstream_tp | ReduceScatter | Attention 后 | 切分激活给下游 FFN/MoE |
| attention_tp → moe_tp | AllGather | MoE 前 | MoE 需要更多 TP 资源 |
| moe_tp → attention_tp | ReduceScatter | MoE 后 | MoE 输出恢复到 Attention 级别 |
| attention_tp → moe_tp (反向) | ReduceScatter + AllGather | MoE 后 | MoE 输出恢复到 Attention 级别 |

**upstream_tp 追踪逻辑** (在模型层维护):

```python
# 模型层追踪每一层输出的实际 TP 级别
current_upstream_tp = attention_tp  # Embedding 输出

for layer_idx in range(num_layers):
    downstream_tp = moe_tp if is_moe_layer(layer_idx) else attention_tp

    # Attention 接收 upstream_tp，输出 attention_tp (通过 AllReduce)
    attn_module = ModuleDSAAttention(
        ..., upstream_tp=current_upstream_tp, downstream_tp=downstream_tp
    )

    if is_moe_layer(layer_idx):
        # MoE 负责将 moe_tp 转换回 attention_tp
        moe_module = ModuleMoE(...)
        current_upstream_tp = attention_tp
    else:
        # Dense FFN 有 AllReduce，输出 attention_tp
        ffn_module = ModuleDenseFFN(...)
        current_upstream_tp = attention_tp
```

**关键**: Dense FFN 和 MoE 的输出都会转换回 attention_tp 级别，因此下一层 Attention 的 upstream_tp 始终是 attention_tp。

### SwiGLU 结构的 FLOPs 计算

SwiGLU 结构：`output = down(silu(gate(x)) * up(x))`

**融合内核视角的 Vector FLOPs 分布**：

| 分支 | 操作 | CUBE FLOPs | Vector FLOPs |
|------|------|------------|--------------|
| `gate_proj` | hidden → intermediate | `2 × B × S × hidden × intermediate` | **0** (激活在down) |
| `up_proj` | hidden → intermediate | `2 × B × S × hidden × intermediate` | 0 |
| `down_proj` | intermediate → hidden | `2 × B × S × intermediate × hidden` | **7 × B × S × intermediate** |

**关键点**：
- Gate 和 Up 的 **CUBE FLOPs 完全相同**（矩阵维度相同）
- **所有 SwiGLU 激活的 Vector FLOPs 统一在 Down 层建模**（融合内核视角）
- Vector FLOPs 公式：`7 × batch × seq × intermediate_per_tp × top_k`（MoE）或 `7 × batch × seq × intermediate_per_tp`（Dense）

**为什么是 7 FLOPs/element？**

SwiGLU 激活操作的 Vector FLOPs 分解：
```
silu(gate) * up:
  - SiLU: x * sigmoid(x) ≈ 3 FLOPs (neg + exp近似 + reciprocal)
  - silu * up: 逐元素乘 ≈ 2 FLOPs (2次读取 + 乘法)
  - 融合结果准备: ≈ 2 FLOPs (中间结果处理)
  合计: 3 + 2×2 = 7 FLOPs/element
```

**融合内核建模原则**：
- 在融合 MoE/FFN 内核中，gate/up 是纯 CUBE matmul
- 激活操作 (SiLU + multiply) 在 Down 的 CUBE 之前连续执行
- 因此所有 Vector FLOPs 归入 Down 层，避免 gate_proj 重复计算

### 算子命名规范

**层次化命名约定**：

| 层类型 | 类名前缀 | 文件名前缀 | 用途 |
|--------|----------|------------|------|
| Router Gate | `LayerMoEGate` | `layer_moe_gate.py` | 专家路由选择 |
| Expert 投影 | `LayerExpert*` | `layer_expert_*.py` | 专家内部投影（SwiGLU） |
| Dense FFN | `LayerDense*` | `layer_dense_*.py` | 稠密 FFN |

**Expert 投影层同时用于 Shared 和 Routed Expert**：
- `LayerExpertUp` - Up 分支
- `LayerExpertGateProj` - Gate 分支
- `LayerExpertDown` - Down 投影
- 通过 `top_k` 参数区分：Shared (top_k=1) vs Routed (top_k=8)

**向后兼容别名**：
```python
LayerMoEGateProj = LayerExpertGateProj
LayerMoEUp = LayerExpertUp
LayerMoEDown = LayerExpertDown
```

### Norm 算子计算公式

#### RMSNorm

`y = x * rsqrt(mean(x²) + eps) * weight`

**Vector FLOPs**: `6 × effective_batch × seq × hidden`
```
每个元素约6 FLOPs:
- x²: 1 mul
- sum(x²): ~2 FLOPs (归约)
- rsqrt(mean + eps): ~4 FLOPs
- x * rsqrt: 1 mul
- * weight: 1 mul (可融合)
```

**Memory**: `2 × effective_batch × seq × hidden × dtype`
```
- 读输入: batch × seq × hidden × dtype
- 写输出: batch × seq × hidden × dtype
- 权重访问可忽略 (hidden << batch × seq × hidden)
```

**effective_batch = micro_batch_size / attention_tp**

#### LayerNorm

`y = (x - mean) / sqrt(var + eps) * gamma + beta`

**Vector FLOPs**: `8 × effective_batch × seq × hidden`
```
每个元素约8 FLOPs (比RMSNorm多):
- sum(x): ~2 FLOPs
- x - mean: 1 sub
- (x-mean)²: 1 mul
- sum((x-mean)²): ~2 FLOPs
- sqrt(var+eps): ~4 FLOPs
- (x-mean)/sqrt: 1 div
- * gamma: 1 mul
- + beta: 1 add
```

**Memory**: `2 × effective_batch × seq × hidden × dtype` (同RMSNorm)

#### 常见错误

```python
# ❌ 错误1：RMSNorm的vector_flops系数太小 (如4)
# 实际约6 FLOPs/element，4遗漏了sum归约和rsqrt的开销
rmsnorm_flops = 4 * batch * seq * hidden

# ❌ 错误2：Norm使用micro_batch_size而非effective_batch
# 应该用 micro_batch_size / attention_tp
rmsnorm_flops = 6 * micro_bs * seq * hidden  # 错误
rmsnorm_flops = 6 * (micro_bs / attn_tp) * seq * hidden  # 正确

# ❌ 错误3：Memory计算包含权重访问但遗漏了数据依赖
# 简化公式只算 读输入 + 写输出，权重可忽略
mem = batch * seq * hidden * act_bytes + hidden * weight_bytes  # 过于复杂
mem = 2 * batch * seq * hidden * dtype  # 简化，足够准确
```

### Batch Size 计算（micro vs global）

#### 核心概念

- **micro_batch_size**: 每个TP组处理的batch大小（配置文件中的值）
- **global_batch_size**: 系统全局batch大小 = micro_batch_size × num_tp_groups
- **num_tp_groups**: TP组数量 = total_chips / attention_tp / context_parallel

#### 计算公式

```python
num_tp_groups = total_chips // attention_tp // context_parallel
global_batch_size = micro_batch_size * num_tp_groups
system_throughput = single_tp_group_throughput * num_tp_groups
```

#### 示例

| 配置 | total_chips | TP | CP | micro_batch | num_tp_groups | global_batch |
|------|-------------|-----|-----|-------------|---------------|--------------|
| 配置1 | 16 | 1 | 1 | 1 | 16 | 16 |
| 配置2 | 16 | 1 | 1 | 8 | 16 | 128 |
| 配置3 | 16 | 2 | 1 | 4 | 8 | 32 |
| 配置4 | 64 | 1 | 4 | 2 | 16 | 32 |

#### 常见错误

```python
# ❌ 错误：把micro_batch_size当作全局batch size
# 导致系统吞吐计算错误

# ✅ 正确：区分micro和global
num_tp_groups = total_chips // attention_tp // context_parallel
global_batch_size = micro_batch_size * num_tp_groups
system_tps = single_tp_tps * num_tp_groups  # 不是乘以global_batch_size!
```

#### 设计原理

1. **TP组是数据并行单元**: 每个TP组独立处理请求，TP组之间无通信
2. **EP不影响数据并行**: EP是专家并行，每个芯片仍然独立处理请求
3. **PP不增加吞吐**: PP是流水线并行，减少延迟但不增加并发
4. **CP减少单组batch**: CP将序列切分到多个芯片，每个芯片处理部分序列

## 并行策略对算子性能的影响

### 概述

并行策略决定了模型如何分布到多个设备上执行，直接影响：
- **通信量**: 设备间数据传输的字节数
- **通信时延**: 通信操作消耗的时间
- **计算时延**: 每个设备上的计算时间
- **内存占用**: 每个设备需要存储的权重和激活值

**核心公式**:
```
总时延 = 计算时延 + 通信时延（未掩盖部分） + 空泡开销（PP）
有效时延 = 单stage时延 × (1 + pipeline_bubble_rate)  [PP > 1 时]
```

**各并行策略对单卡计算的影响**:

| 策略 | FLOPs | 权重访存 | 激活访存 | 通信 | 建模方式 |
|------|-------|----------|----------|------|----------|
| **TP** | ÷ TP | ÷ TP | ÷ TP | AllReduce/AG/RS | 切分权重矩阵 |
| **EP** | **不变** | **÷ EP** | 不变 | All-to-All | 切分专家数量 |
| **PP** | ÷ PP (少构建层) | ÷ PP (少层权重) | 不变 | P2P | 切分模型层数 |
| **CP** | ÷ CP | 不变 | ÷ CP | Ring Attn | 切分序列长度 |
| **moe_tp** | 不变 (切intermediate) | ÷ moe_tp | 不变 | AG/RS | 切分专家intermediate |

**关键**: EP 是唯一一个不影响 FLOPs 但影响权重访存的并行策略！

**额外影响**:
| 因素 | 影响对象 | 公式 | 说明 |
|------|----------|------|------|
| **EP 负载不均衡** | 专家计算时间 | `compute_time × ep_load_imbalance_factor` | 最忙rank决定延迟 |
| **Compute-Comm Overlap** | 通信有效时间 | `max(comm, compute × efficiency)` | DeepEP hook实现 |

---

### 1. Tensor Parallelism (TP / attention_tp / moe_tp / lm_head_tp)

#### 原理
将权重矩阵按列或行切分到多个设备，每个设备持有部分权重，并行计算后通过通信聚合结果。

- **Column Parallel**: 权重按列切分，输入复制到所有设备，输出需要 All-Gather
- **Row Parallel**: 权重按行切分，输入切分到各设备，输出需要 All-Reduce

#### 通信模式
| 操作 | 通信类型 | 通信量公式 |
|------|----------|------------|
| Attention 输出聚合 | All-Reduce | `2 × batch × seq_len × hidden_size × dtype_bytes` |
| MLP Down 投影聚合 | All-Reduce | `2 × batch × seq_len × hidden_size × dtype_bytes` |
| First Layer 输入 | All-Gather (可选) | `batch × seq_len × hidden_size × dtype_bytes` |
| Last Layer 输出 | All-Gather | `batch × seq_len × hidden_size × dtype_bytes` |

**注意**: All-Reduce 的系数 2 来自 Ring All-Reduce 算法的总传输量（实际有效数据只传输1次，但ring算法需要2倍带宽消耗）

#### 对算子性能的影响

```python
# TP=4 时，单个设备的计算量
compute_per_device = total_compute / TP

# 但每个算子后需要 All-Reduce 通信
comm_latency = 2 * batch * seq_len * hidden_size * dtype_bytes / bandwidth

# 实际时延
op_latency = max(compute_time, mem_time) + comm_latency
```

#### TP 的选择原则
- **TP 越大**: 单设备计算量越小，但通信量不变 → 通信占比上升
- **临界点**: 当 `compute_time ≈ comm_time` 时，继续增大 TP 收益递减
- **带宽敏感**: 跨节点 TP 需要高带宽（NVLink/IB），否则通信成为瓶颈
- **经验值**: 单节点内 TP=2/4/8，跨节点一般不用 TP（用 PP 代替）

#### 不同算子的 TP 配置
| 算子 | 并行方式 | 说明 |
|------|----------|------|
| QKV 投影 | Column Parallel | 每个 TP rank 处理部分 head |
| Attention 输出 | Row Parallel + All-Reduce | 聚合各 head 结果 |
| Gate/Up 投影 | Column Parallel | 每个 TP rank 处理部分 intermediate |
| Down 投影 | Row Parallel + All-Reduce | 聚合回 hidden_size |
| LM Head | Column Parallel | 每个 TP rank 处理部分 vocab |

---

### 2. Expert Parallelism (EP)

#### 原理
将 MoE 的不同专家分配到不同设备，每个设备持有部分专家的完整权重。Token 根据路由结果被分发到对应专家所在的设备。

#### 通信模式
| 阶段 | 通信类型 | 通信量公式 |
|------|----------|------------|
| Dispatch | All-to-All | `batch × seq_len × topk × hidden_size × dtype_bytes` |
| Expert Compute | 本地 | 无通信 |
| Combine | All-to-All | `batch × seq_len × topk × hidden_size × dtype_bytes` |

**关键**: EP 的通信量是 TP 的 **~9倍**（参考 DeepSeek-V3 技术报告），因为：
- 每个 token 可能路由到任意专家
- All-to-All 需要全量数据重排

#### 对算子性能的影响

```python
# EP 阶段的时延分解
dispatch_time = all_to_all_latency(tokens, hidden_size, EP)
expert_compute_time = compute_per_expert / num_experts_per_device
combine_time = all_to_all_latency(tokens, hidden_size, EP)

# MoE 层总时延（假设无掩盖）
moe_latency = dispatch_time + expert_compute_time + combine_time
```

#### All-to-All 时延建模
```python
def all_to_all_latency(data_size, num_ranks, bandwidth, rtt_overhead):
    # All-to-All 每个 rank 需要向其他 (num_ranks-1) 个 rank 发送数据
    per_pair_data = data_size / num_ranks
    transfer_time = per_pair_data / bandwidth

    # 静态开销 + 传输时间
    # 注意：实际实现可能使用并发传输
    latency = rtt_overhead + transfer_time * (num_ranks - 1)
    return latency
```

#### EP 的选择原则
- **EP 越大**: 每设备专家数越少，内存占用降低
- **通信开销**: EP 通信量与 EP 规模正相关
- **负载均衡**: 需要考虑专家负载不均衡问题
- **优化策略**:
  - 使用 DeepEP 库优化 All-to-All（DeepSeek-V3）
  - 计算-通信掩盖（DualPipe）
  - 跨节点时使用 IB 直连减少跳数

#### EP 与 MoE TP 的对比
| 特性 | EP | MoE TP |
|------|-----|--------|
| 权重分布 | 每设备完整专家 | 每设备部分专家权重 |
| 通信模式 | All-to-All (dispatch/combine) | All-Reduce |
| 通信量 | 高 (~9x TP) | 低 |
| 内存效率 | 高（专家数多时） | 低（每个 TP rank 都要存完整专家） |
| 适用场景 | 大规模 MoE (专家数 > 设备数) | 小规模 MoE |

---

### 3. Pipeline Parallelism (PP)

#### 原理
将模型按层切分到多个设备，形成流水线。数据以 micro-batch 形式流经各阶段。

#### 空泡开销 (Pipeline Bubble)
```
空泡率 ≈ (PP - 1) / num_micro_batches
```

- **PP 越大**: 空泡率越高（假设 micro-batch 数不变）
- **增加 micro-batch**: 可降低空泡率，但增加内存占用

#### 对算子性能的影响
```python
# PP 不改变单算子的计算量，但引入空泡开销
effective_compute_time = sum(op_latencies) / PP
bubble_overhead = (PP - 1) / num_micro_batches * total_latency

# 实际吞吐
throughput = batch_size / (effective_compute_time + bubble_overhead)
```

#### PP 的选择原则
- **PP 越大**: 单设备内存占用越小，但空泡开销增加
- **优化技术**:
  - **Interleaved PP**: 交替分配层，减少空泡
  - **Zero-Bubble PP**: 优化调度消除空泡
  - **1F1B 调度**: One Forward One Backward，平衡内存和空泡
- **适用场景**: 跨节点部署，网络带宽有限时优先用 PP 而非 TP

#### PP 空泡率配置
在 `deploy_config` 中配置 `pp_bubble_rate`，典型值：
- 标准 PP: 15-30%
- Interleaved PP: 5-15%
- Zero-Bubble PP: <5%

---

### 4. Context Parallelism (CP)

#### 原理
将长序列按 token 切分到多个设备，每个设备处理部分序列。主要用于处理超长上下文（100K+ tokens）。

#### 通信模式
使用 **Ring Attention** 实现，需要在 attention 计算时进行环形通信：
- 每个 CP rank 持有部分 KV cache
- 计算完整 attention 需要遍历所有 CP rank 的 KV

#### 对算子性能的影响
```python
# CP=4 时，单设备处理的序列长度
seq_len_per_device = seq_len / CP

# Attention 计算量
attn_flops_per_device = batch * heads * (seq_len/CP) * kv_seq_len * head_dim

# 但需要 Ring 通信获取完整 KV
ring_comm_rounds = CP
comm_latency = ring_comm_rounds * (kv_seq_len * head_dim * dtype_bytes / bandwidth)
```

#### CP 的选择原则
- **仅用于长序列**: seq_len > 单卡显存容量时使用
- **Prefill 阶段主要受益**: Decode 阶段 seq_len=1，CP 效果有限
- **通信开销**: Ring Attention 的通信与 CP 规模成正比
- **带宽敏感**: 需要高带宽互联（NVLink）

#### CP vs Sequence Parallelism (SP)
| 特性 | SP | CP |
|------|-----|-----|
| 定义 | Megatron-LM 中的序列切分 | vLLM/NeMo 中的上下文切分 |
| 范围 | 主要在 Attention 层 | 覆盖所有层的 activation |
| 通信 | Reduce-Scatter/All-Gather | Ring Attention (P2P) |
| 关系 | CP 是 SP 的扩展/泛化 | - |

---

### 5. 并行策略组合

#### 常见组合模式

```
总设备数 = TP × PP × EP × CP
```

| 场景 | 推荐配置 | 理由 |
|------|----------|------|
| 单节点 8 GPU | TP=8, PP=1, EP=1 | NVLink 带宽高，TP 通信快 |
| 单节点 MoE | TP=2, EP=4 | 平衡 TP 和 EP 通信 |
| 跨节点 Dense | TP=2, PP=4 | 跨节点用 PP 避免高通信延迟 |
| 跨节点 MoE | TP=1, EP=8, PP=N | EP 通信可用 IB 优化 |
| 长序列 | CP=4, TP=2 | CP 处理长序列，TP 加速计算 |

#### DeepSeek-V3 的并行策略示例
- **训练**: TP=1, EP=8, PP= DualPipe
- **推理**:
  - Attention: TP=1
  - MoE: EP=8 (per node)
  - 跨节点 All-to-All 使用 IB 直连

---

### 6. 通信时延建模总结

#### 通信原语时延公式
```python
# All-Reduce (Ring 算法)
all_reduce_latency = 2 * data_size / (bandwidth * num_ranks) + rtt_overhead

# All-to-All
all_to_all_latency = data_size / bandwidth + rtt_overhead * (num_ranks - 1)

# Reduce-Scatter
reduce_scatter_latency = data_size / (bandwidth * num_ranks) + rtt_overhead

# All-Gather
all_gather_latency = data_size / (bandwidth * num_ranks) + rtt_overhead

# P2P (Ring Attention)
p2p_latency = data_size / bandwidth + rtt_overhead
```

#### 带宽选择
| 通信范围 | 典型带宽 | 利用率 |
|----------|----------|--------|
| 节点内 (NVLink) | 450-900 GB/s | 70-90% |
| 节点内 (PCIe) | 64 GB/s | 60-80% |
| 框内跨节点 (IB) | 200-400 Gb/s | 50-70% |
| 框间 (IB) | 100-200 Gb/s | 40-60% |

---

### 配置文件管理

- 所有配置使用 JSON 格式
- 目录结构:
  ```
  configs/
  ├── hardware/
  │   └── 硬件名_芯片数.json
  ├── models/
  │   └── 模型名_版本.json
  ├── quantization/
  │   └── 量化策略.json
  └── deployment/
      └── 部署方式_配置.json
  ```

## 常见模型架构

FFN结构
- **Dense**: Llama, Qwen, GPT (标准 Transformer)
- **MoE**: DeepSeek-V3, Mixtral (专家混合)

注意力结构
- **MLA**: DeepSeek-V3 (Multi-head Latent Attention)
- **DSA**: DeepSeek-V3.2 (DeepSeek Sparse Attention)
- **GQA**：minimax m2
- **Linear and Full Attention混合结构**: Qwen 3.5

## 参考资料

- `references/skill_desc.md`: 详细建模指导
- `references/*.py`: 代码框架参考
- `vllm/model_executor/models/`: vllm 模型实现
- 论文和技术博客

### 并行策略参考来源
- [DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437) - EP 通信优化、DualPipe
- [DeepEP GitHub](https://github.com/deepseek-ai/DeepEP) - MoE All-to-All 通信库
- [NVIDIA Megatron Bridge - Parallelisms Guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html) - TP/PP/EP 原理
- [Introducing Context Parallelism - Insu Jang](https://insujang.github.io/2024-09-20/introducing-context-parallelism/) - CP 原理
- [vLLM RFC: Context Parallelism](https://github.com/vllm-project/vllm/issues/22693) - CP vs SP 对比
- [Tensor Parallel LLM Inferencing - Medium](https://medium.com/tr-labs-ml-engineering-blog/tensor-parallel-llm-inferencing-09138daf0ba7) - TP All-Reduce
- [PipeFill: Using GPUs During Bubbles](https://www.pdl.cmu.edu/PDL-FTP/BigLearning/PipeFill_MLSys25.pdf) - PP 空泡优化
- [NVIDIA Blog: Optimizing MoE Communication](https://developer.nvidia.com/blog/optimizing-communication-for-mixture-of-experts-training-with-hybrid-expert-parallel/) - EP 通信瓶颈

## 工作示例

**用户请求**: "测试 DeepSeek V3.2 在 91
0C(16卡) 上的 decode 性能，attention tp=1, EP=16, 序列长度8K"

**执行步骤**:
1. 检查配置文件是否存在
2. 如果缺失，从 HuggingFace 下载模型配置
3. 验证模型实现代码（特别是 DSA attention）
4. 创建部署配置文件
5. 运行性能仿真
6. 分析结果，识别瓶颈
7. 给出优化建议

**关键检查**:
- ✅ 模型配置中有 `topk_tokens` 参数
- ✅ Attention 实现使用 `topk_tokens` 而非 `sparse_ratio`
- ✅ Decode 阶段的计算量 = seq_len × topk_tokens
- ✅ 参考 vllm 的 `SparseAttnIndexer` 实现

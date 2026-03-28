# 并行策略详解

本文档详细说明 TP/EP/PP/CP 并行策略及其对性能的影响。

---

## 并行策略影响总表

| 策略 | FLOPs | 权重访存 | 激活访存 | 通信 | 建模方式 |
|------|-------|----------|----------|------|----------|
| **TP** | ÷ TP | ÷ TP | ÷ TP | AllReduce/AG/RS | 切分权重矩阵 |
| **EP** | 不变 | × (num_experts/ep) | 不变 | All-to-All | 存储部分专家 |
| **PP** | ÷ PP | ÷ PP | 不变 | P2P | 切分模型层数 |
| **CP** | ÷ CP | 不变 | ÷ CP | Ring Attn | 切分序列长度 |
| **moe_tp** | 不变 | ÷ moe_tp | 不变 | AG/RS | 切分 expert intermediate |

**关键**：EP 是唯一不影响 FLOPs 但影响权重访存的并行策略！

---

## 1. Tensor Parallelism (TP)

### 原理

将权重矩阵按列或行切分到多个设备。

- **Column Parallel**: 输出需要 All-Gather
- **Row Parallel**: 输出需要 All-Reduce

### 通信模式

| 操作 | 通信类型 | 通信量公式 |
|------|----------|------------|
| Attention 输出聚合 | All-Reduce | `2 × batch × seq × hidden × dtype` |
| MLP Down 投影聚合 | All-Reduce | `2 × batch × seq × hidden × dtype` |
| First Layer 输入 | All-Gather | `batch × seq × hidden × dtype` |

### 选择原则

- TP 越大：单设备计算量越小，但通信量不变
- 临界点：当 `compute_time ≈ comm_time` 时，继续增大 TP 收益递减
- 经验值：单节点内 TP=2/4/8，跨节点一般不用 TP

---

## 2. Expert Parallelism (EP)

### 原理

将 MoE 的不同专家分配到不同设备，每个设备持有部分专家的**完整权重**。

### EP 对权重访存的影响

**核心原则：EP 切分专家，不切分权重！**

```
EP 权重分布:
  Rank 0: Expert 0-31 (完整权重)
  Rank 1: Expert 32-63 (完整权重)
  ...

每个 EP rank 存储 num_experts_per_ep 个专家的完整权重。
```

```python
num_experts_per_ep = ceil(num_experts / ep) + r_per_ep

# Routed Expert
read_weight = hidden * intermediate * weight_bytes * num_experts_per_ep

# Shared Expert (不使用 EP)
read_weight = hidden * intermediate * weight_bytes * num_shared_experts
```

### 通信模式

| 阶段 | 通信类型 | 通信量 |
|------|----------|--------|
| Dispatch | All-to-All | `tokens × top_k × (EP-1)/EP × hidden` |
| Combine | All-to-All | 同上 |

**EP 通信量是 TP 的 ~9 倍**（DeepSeek-V3 技术报告）。

### EP All-to-All 通信量推导

```python
tokens = batch * seq / attention_tp  # EP dispatch 在 TP 之后
data_size = tokens * top_k * (ep - 1) / ep * hidden * act_bytes
```

**为什么是 `(EP-1)/EP` 而不是 `1/EP`？**
- 融合 dispatch：每个 token 发送到所有需要跨 EP 通信的 rank
- 平均每个 token 有 `top_k × (EP-1)/EP` 个专家在其他 EP rank

### EP 负载不均衡

```python
expert_compute_time = base_time * ep_load_imbalance_factor
# 典型值: 1.0 (完美均衡) ~ 1.5 (严重不均衡)
# DeepSeek-V3 使用 auxiliary loss 控制，典型值 ~1.1
```

### EP vs MoE TP

| 特性 | EP | MoE TP |
|------|-----|--------|
| 权重分布 | 每设备完整专家 | 每设备部分专家权重 |
| 通信模式 | All-to-All | All-Reduce |
| 通信量 | 高 (~9x TP) | 低 |
| 内存效率 | 高（专家数多时） | 低 |
| 适用场景 | 大规模 MoE | 小规模 MoE |

---

## 3. Pipeline Parallelism (PP)

### 原理

将模型按层切分到多个设备，形成流水线。

### Pipeline Bubble

```
空泡率 ≈ (PP - 1) / num_micro_batches
```

```python
if pp > 1:
    effective_time = total_time * (1 + bubble_rate)
```

### PP 模型构建

```
Stage 0:     Embedding + layer[0:N/PP] + P2P send
Stage i:     layer[i*N/PP:(i+1)*N/PP] + P2P send
Stage PP-1:  layer[(PP-1)*N/PP:N] + LM Head + MTP
```

### P2P 通信

```python
time = data / bw + rtt + static_overhead
# data = batch * seq * hidden * act_bytes
```

---

## 4. Context Parallelism (CP)

### 原理

将长序列按 token 切分到多个设备，用于处理超长上下文。

### Ring Attention

```python
seq_per_cp = seq / CP
# Attention 计算需要遍历所有 CP rank 的 KV
ring_rounds = CP - 1
```

### CP 通信量

```python
# MLA: 使用压缩 KV
kv_bytes = seq_per_cp * (kv_lora_rank + qk_rope_head_dim) * dtype

# GQA: 使用完整 KV
kv_bytes = seq_per_cp * num_kv_heads * head_dim * dtype
```

---

## 5. DeepEP 优化技术

**DeepEP** 是 DeepSeek 开源的高性能 MoE 通信库。

### 核心优化

| 技术 | 原理 | 效果 |
|------|------|------|
| 双模式内核 | High-Throughput vs Low-Latency | Prefill/Decode 分别优化 |
| Pure RDMA | 绕过 NCCL | 延迟降低 50%+ |
| Hook 机制 | 零 SM 占用 | 通信与计算重叠 |
| NVLink + RDMA | 节点内/外混合 | 最大化带宽 |

### 延迟模型

```python
# High-Throughput (Prefill)
latency = data / bw + rtt * sqrt(N-1)

# Low-Latency (Decode)
latency = 50us + transfer_time * log2(EP)
```

### Compute-Communication Overlap

```python
# Dispatch 与 Shared Expert 并行
effective_time = max(dispatch_time, shared_expert_time)
```

**重叠效率**：
```python
efficiency = min(overlapable_compute / comm_time, 1.0)
```

典型场景：Shared Expert 计算时间 >= Dispatch 通信时间时，Dispatch 完全隐藏。

### 配置参数

```json
{
  "comm_rdma_bw_gbps": 50.0,
  "deepep_base_latency_us": 50.0,
  "deepep_overlap_efficiency": 0.9
}
```

---

## 6. 并行策略组合

```
总设备数 = TP × PP × EP × CP
```

| 场景 | 推荐配置 |
|------|----------|
| 单节点 8 GPU | TP=8, PP=1, EP=1 |
| 单节点 MoE | TP=2, EP=4 |
| 跨节点 Dense | TP=2, PP=4 |
| 跨节点 MoE | TP=1, EP=8, PP=N |
| 长序列 | CP=4, TP=2 |
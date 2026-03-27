# LLM Roofline Simulation

大模型性能仿真工具，基于 Roofline 模型分析大语言模型在不同硬件上的性能表现。

## 项目简介

本项目用于仿真大语言模型（LLM）在华为昇腾等硬件上的性能表现，通过 Roofline 模型分析计算密度与内存带宽的关系，帮助识别性能瓶颈。

支持的模型架构：
- **Dense模型**：Llama系列等标准Transformer架构
- **MoE模型**：DeepSeek-V3/R1等专家混合架构

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行仿真

```bash
# 使用Llama3-70B模型
python main.py --model configs/models/llama3_70b.json

# 使用DeepSeek-V3 MoE模型
python main.py --model configs/models/deepseek_v3.json

# 自定义所有配置
python main.py \
  --hardware configs/hardware/ascend_910b.json \
  --model configs/models/llama3_70b.json \
  --quant configs/quantization/default_fp8.json \
  --deploy configs/deployment/default_deployment.json
```

### 查看结果

仿真结果保存在 `outputs/` 目录：
- `op_details.csv` - 算子级详细性能指标
- `single_card_perf.json` - 单卡性能指标
- `system_perf.json` - 系统级性能指标

## 项目结构

```
llm_roofline_simulation/
├── configs/                    # 配置文件
│   ├── hardware/              # 硬件配置
│   ├── models/                # 模型结构配置
│   ├── quantization/          # 量化配置
│   └── deployment/            # 部署策略配置
├── llm_sim/                   # 核心代码
│   ├── ops/                   # 算子层
│   ├── modules/               # 模块层
│   ├── models/                # 模型层
│   ├── configs/               # 配置数据类
│   └── results/               # 结果输出
├── main.py                    # 入口文件
└── requirements.txt           # 依赖
```

## 配置说明

### 硬件配置 (hardware/)

包含硬件的算力、带宽、延迟等参数：
- CUBE/Vector算力（FP4/FP8/FP16/FP32）
- 多级缓存（HBM/Host/SSD）
- 通信带宽（4GPU/8GPU/框内/框间）
- 芯片配置

### 模型配置 (models/)

包含模型结构参数：
- 基础配置：hidden_size, num_layers, num_heads等
- MoE配置：num_experts, num_experts_per_tok等
- Attention类型：MHA/GQA/MLA

### 量化配置 (quantization/)

支持per-op量化精度设置：
- weight_bits：权重精度
- activation_compute_bits：激活计算精度
- activation_transfer_bits：激活传输精度
- cache_read/write_bits：KV cache精度

### 部署配置 (deployment/)

包含部署策略和业务负载：
- 并行策略：TP/PP/EP/CP
- 投机解码：mtp_length, mtp_acceptance_rate
- 业务负载：input_length, output_length

## 核心算法

### Roofline模型

算子时延计算公式：
```
latency = max(cube_lat + vector_lat, mem_lat) + op_overhead + comm_lat
```

其中：
- `cube_lat = cube_flops / (cube_tflops * cube_util)`
- `vector_lat = vector_flops / (vector_tflops * vector_util)`
- `mem_lat = mem_bytes / (hbm_bw * hbm_bw_util)`

### 支持的算子

- **计算算子**：MatMul, Attention, RMSNorm
- **通信算子**：AllReduce, AllGather, ReduceScatter, P2P

## 当前实现状态

✅ **已完成（阶段1-3）**：
- 配置体系（硬件/模型/量化/部署）
- 算子层（MatMul/Attention/Norm/通信）
- 结果输出（CSV/JSON）
- 简单示例（单个Transformer Block）

🚧 **待实现（阶段4-5）**：
- 完整的模块层和模型层
- 系统寻优（并行策略搜索）
- Web前端界面

## 示例输出

### 单卡性能指标
```json
{
  "total_latency_ms": 8.017,
  "ttft_ms": 8.017,
  "tpot_ms": 0.063,
  "qps": 124.73,
  "tps": 15965.11,
  "latency_breakdown": {
    "cube_latency_ms": 7.919,
    "vector_latency_ms": 0.034,
    "memory_latency_ms": 0.693,
    "comm_latency_ms": 0.0
  },
  "latency_ratio": {
    "cube_ratio": 0.988,
    "vector_ratio": 0.004,
    "memory_ratio": 0.086,
    "comm_ratio": 0.0
  }
}
```

## 贡献

欢迎提交Issue和Pull Request！

## License

MIT License

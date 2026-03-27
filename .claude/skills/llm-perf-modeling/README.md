# LLM Performance Modeling Skill

这是一个用于 LLM 性能建模的 Claude Code Skill，基于 Roofline 模型分析大语言模型在不同硬件上的性能表现。

## 目录结构

```
llm-perf-modeling/
├── SKILL.md              # Skill 主描述文件（Claude 加载的核心文件）
├── README.md             # 本文件，说明 skill 的使用方法
└── references/           # 参考资料目录
    ├── skill_desc.md     # 详细的建模指导文档
    ├── layer_base.py     # 算子层基类参考
    ├── layer_moe_down.py # MoE 算子示例
    ├── module_base.py    # 模块层基类参考
    ├── m_dsa_attn.py     # DSA Attention 模块示例
    ├── inference_base.py # 模型层基类参考
    ├── prefill_ds_3_2.py # DeepSeek V3.2 Prefill 模型示例
    └── main_ref.py       # 主程序参考
```

## 使用方法

### 1. 自动加载

当你在 `llm_roofline_simulation` 目录下启动 Claude Code 时，这个 skill 会自动加载。

### 2. 手动调用

```bash
/llm-perf-modeling
```

### 3. 典型工作流程

**场景**: 测试 DeepSeek V3.2 在 910C(16卡) 上的性能

1. **检查配置**
   ```
   请检查 DeepSeek V3.2 的配置文件是否完整
   ```

2. **验证实现**
   ```
   请验证 DSA attention 的实现是否正确
   ```

3. **运行仿真**
   ```
   运行 DeepSeek V3.2 在 910C 上的性能测试，
   配置：attention tp=1, EP=16, 序列长度8K
   ```

4. **分析结果**
   ```
   分析性能瓶颈并给出优化建议
   ```

## 核心功能

### 1. 配置管理
- 自动获取硬件配置（从用户提供或网络搜索）
- 从 HuggingFace 下载模型配置
- 验证配置的完整性和正确性

### 2. 代码生成
- 生成算子层代码（继承 LayerBase）
- 生成模块层代码（继承 ModuleBase）
- 生成模型层代码（继承 InferenceBase）

### 3. 性能计算
- 基于 Roofline 模型计算时延
- 生成详细的性能报告
- 识别性能瓶颈（计算/访存/通信）

### 4. 系统寻优
- 搜索最优并行策略
- 在约束条件下最大化吞吐量

## 重要原则

### DSA (DeepSeek Sparse Attention)

**关键**: DSA 使用 `topk_tokens` 参数（绝对值），而不是 `sparse_ratio`（比例）

```python
# ❌ 错误
effective_kv_len = int(kv_seq_len * 0.25)

# ✅ 正确
topk_tokens = model_config.topk_tokens  # 如 256, 512
if is_prefill:
    effective_kv_len = kv_seq_len
else:
    effective_kv_len = topk_tokens
```

### MLA (Multi-head Latent Attention)

关键优化：KV cache 存储压缩后的 latent (kv_lora_rank 维度)

### 代码生成

1. 继承现有 Base 类
2. 参考 references/ 目录的示例
3. 验证 vllm 实现
4. 配置驱动，不硬编码

## 参考资料

- `references/skill_desc.md`: 详细的建模指导
- `references/*.py`: 代码框架和示例
- `vllm/model_executor/models/`: vllm 模型实现参考

## 常见问题

### Q: Skill 没有自动加载？
A: 确保你在 `llm_roofline_simulation` 目录下启动 Claude Code，或者重启 Claude。

### Q: 如何验证 DSA 实现是否正确？
A: 检查以下几点：
1. 模型配置中有 `topk_tokens` 参数
2. Attention 层使用 `topk_tokens` 而非 `sparse_ratio`
3. Decode 阶段计算量 = seq_len × topk_tokens
4. 参考 vllm 的 `SparseAttnIndexer` 实现

### Q: 如何添加新的模型支持？
A:
1. 从 HuggingFace 下载模型配置
2. 参考 vllm 代码确认模型结构
3. 创建模型层代码（继承 InferenceBase）
4. 实现必要的算子和模块
5. 运行测试验证

## 版本历史

- v1.0 (2026-03-26): 初始版本
  - 支持 Dense 和 MoE 模型
  - 支持 MLA 和 DSA attention
  - 基于 Roofline 模型的性能计算

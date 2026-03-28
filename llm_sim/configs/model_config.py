"""模型结构配置数据类"""

from dataclasses import dataclass
from typing import Optional
import json


@dataclass
class ModelConfig:
    """模型结构配置，支持Dense和MoE架构"""

    # 基础配置
    model_type: str  # "dense" or "moe"
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int  # GQA/MQA
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int

    # MoE专用配置
    num_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None  # top_k
    num_shared_experts: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    first_k_dense_replace: Optional[int] = None
    moe_layer_freq: Optional[int] = None
    n_group: Optional[int] = None
    topk_group: Optional[int] = None
    routed_scaling_factor: Optional[float] = None
    scoring_func: Optional[str] = None
    norm_topk_prob: Optional[bool] = None

    # Attention配置
    attention_type: str = "mha"  # "mha", "gqa", "mla", "dsa", "hybrid"
    head_dim: Optional[int] = None  # GQA/MHA/Full Attention: 标准head_dim
    rope_theta: float = 10000.0

    # MLA专用配置
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    q_lora_rank: Optional[int] = None
    kv_lora_rank: Optional[int] = None

    # DSA专用配置 (DeepSeek V3.2)
    index_topk: Optional[int] = None  # 稀疏注意力选择的token数量
    index_n_heads: Optional[int] = None  # indexer的head数量
    index_head_dim: Optional[int] = None  # indexer的head维度

    # 混合注意力配置 (Qwen3.5: Gated DeltaNet + Full Attention)
    full_attention_interval: Optional[int] = None  # 每N层一个 full attention (如4表示每4层一个)
    layer_types: Optional[list] = None  # 每层类型列表: ["linear_attention", "full_attention", ...]
    linear_key_head_dim: Optional[int] = None  # DeltaNet K head维度
    linear_num_key_heads: Optional[int] = None  # DeltaNet K head数量
    linear_num_value_heads: Optional[int] = None  # DeltaNet V head数量
    linear_value_head_dim: Optional[int] = None  # DeltaNet V head维度
    linear_conv_kernel_dim: Optional[int] = None  # DeltaNet 卷积核维度
    shared_expert_intermediate_size: Optional[int] = None  # 共享专家中间维度

    # 投机解码配置 (DeepSeek V3.2 Multi-Token Prediction)
    num_nextn_predict_layers: Optional[int] = None  # MTP head层数，0或None表示不使用

    # 其他
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False

    @classmethod
    def from_json(cls, json_path: str) -> "ModelConfig":
        """从JSON文件加载配置"""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, json_path: str):
        """保存配置到JSON文件"""
        data = {k: v for k, v in self.__dict__.items() if v is not None}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @property
    def is_moe(self) -> bool:
        """是否为MoE模型"""
        return self.model_type == "moe" and self.num_experts is not None

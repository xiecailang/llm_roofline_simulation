"""部署配置数据类"""

from dataclasses import dataclass
from typing import Optional
import json


@dataclass
class DeploymentConfig:
    """模型部署配置"""

    # 部署方式
    deployment_mode: str  # "prefill", "decode", "pd"

    # 投机解码配置
    mtp_length: int = 0  # 投机长度，0表示不使用投机解码
    mtp_acceptance_rate: float = 1.0  # 投机接纳率

    # 并行策略
    context_parallel: int = 1  # CP
    attention_tp: int = 1  # Attention TP
    pipeline_parallel: int = 1  # PP
    pipeline_bubble_rate: float = 0.1  # PP空泡率
    expert_parallel: int = 1  # EP (MoE专用)
    moe_tp: int = 1  # MoE专家内部TP
    lm_head_tp: int = 1  # LM Head TP
    micro_batch_size: int = 1

    # EP负载不均衡配置
    # 负载不均衡系数 = max_load / avg_load，表示最繁忙EP rank的负载是平均负载的多少倍
    # 理想情况(完美均衡): 1.0
    # 典型场景: 1.1 - 1.3 (10%-30%不均衡)
    # 恶劣场景: 1.5+ (50%+不均衡)
    # 参考: DeepSeek-V3使用auxiliary loss控制，典型值~1.1
    ep_load_imbalance_factor: float = 1.1

    # EP冗余专家配置
    # 每 EP rank 的冗余专家数，用于负载均衡
    # 0 = 无冗余（默认）
    # 典型值: 1-2（每个 rank 多存 1-2 个专家以平衡负载）
    # 参考: DeepSeek-V3 使用 auxiliary loss 控制负载均衡
    r_per_ep: int = 0

    # 计算-通信重叠配置
    # 启用DeepEP风格的计算-通信重叠
    enable_compute_comm_overlap: bool = True
    # 重叠效率 (0.0 - 1.0)，表示可被隐藏的通信比例
    # 典型值: 0.85 - 0.95 (DeepEP实测)
    overlap_efficiency: float = 0.9

    # 业务负载
    input_length: int = 1024
    output_length: int = 128

    # Prefix Cache 配置
    # 命中率 (0.0 - 1.0)
    # 0.0: 无缓存命中，完整 prefill
    # 1.0: 完全缓存命中，仅计算新增 token
    # 典型值: 0.3-0.7 (系统提示 + few-shot 示例已被缓存)
    # 命中的 prefix token 不需要重新计算 attention 和 FFN
    prefix_cache_hit_rate: float = 0.0

    # 注意：稀疏注意力 (DSA) 的 topk_tokens 参数应在 model_config 中配置
    # 而不是 deployment_config。DSA 使用 index_topk (绝对值)，而非 ratio (比例)

    @classmethod
    def from_json(cls, json_path: str) -> "DeploymentConfig":
        """从JSON文件加载配置"""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, json_path: str):
        """保存配置到JSON文件"""
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2, ensure_ascii=False)

    @property
    def total_parallelism(self) -> int:
        """总并行度"""
        return (
            self.context_parallel
            * self.attention_tp
            * self.pipeline_parallel
            * self.expert_parallel
        )

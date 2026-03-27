"""Expert Up算子 - MoE专家的Up分支

hidden -> intermediate (ColumnParallel)

SwiGLU结构中的Up分支，与Gate分支独立计算。
此算子同时用于 Shared Expert 和 Routed Expert。

FLOPs公式:
  hidden_size * (moe_intermediate_size / moe_tp) * 1 * top_k * moe_batch_size * seq_len * 2

公式解析:
  - hidden_size: 输入维度 (7168)
  - moe_intermediate_size / moe_tp: 每个MoE TP rank处理的intermediate维度
  - * 1: 单个分支（Up）
  - top_k: 每个token路由的专家数 (routed=8, shared=1)
  - moe_batch_size: MoE层的有效batch size
  - seq_len: 序列长度
  - * 2: FLOPs系数 (每次乘加算2 FLOPs)

moe_batch_size计算:
  moe_batch_size = micro_batch_size / attention_tp * moe_tp

关键点:
  - EP不影响FLOPs! EP只影响通信和权重存储
  - Shared Expert: top_k=1
  - Routed Expert: top_k=num_experts_per_tok (如8)
"""

from .layer_base import LayerBase


class LayerExpertUp(LayerBase):
    """Expert Up: hidden -> intermediate (ColumnParallel)

    同时用于 Shared Expert 和 Routed Expert。
    """

    def __init__(self, hardware_config, model_config, deploy_config, quant_config, seq_len, top_k=1):
        super().__init__(hardware_config, model_config, deploy_config, quant_config)
        self.seq_len = seq_len
        self.top_k = top_k  # routed expert=top_k, shared expert=1
        self.moe_intermediate = getattr(model_config, 'moe_intermediate_size', model_config.intermediate_size)
        # MoE TP: intermediate按moe_tp切分
        self.intermediate_per_tp = self.moe_intermediate // self.moe_tp

        # 计算MoE层的有效batch size
        # moe_batch_size = micro_batch_size / attention_tp * moe_tp
        self.moe_batch_size = self.batch_size / self.tp * self.moe_tp

    def get_cube_flops(self):
        """计算量:
        hidden * (intermediate / moe_tp) * 1 * top_k * moe_batch_size * seq * 2

        注意: EP不影响FLOPs!
        """
        flops = (
            self.hidden_size *                       # 输入维度
            self.intermediate_per_tp *               # intermediate / moe_tp
            1 *                                      # 单分支 (Up)
            self.top_k *                             # num_experts_per_tok
            self.moe_batch_size *                    # MoE有效batch size
            self.seq_len *                           # 序列长度
            2                                        # FLOPs系数
        )
        return float(flops)

    def get_vector_flops(self):
        """无向量操作（SiLU在Gate分支）"""
        return 0.0

    def get_mem_bytes(self):
        """访存量

        EP影响权重访存量：每个 EP rank 只存储 1/EP 的专家权重
        """
        # 实际处理的token-expert对数
        token_expert_pairs = self.moe_batch_size * self.seq_len * self.top_k

        # Input: [token_expert_pairs, hidden]
        read_input = token_expert_pairs * self.hidden_size * self.act_transfer_bytes
        # Weight: [hidden, intermediate/moe_tp] per expert
        # EP: 每个 EP rank 只存储 1/EP 的专家权重
        read_weight = self.hidden_size * self.intermediate_per_tp * self.weight_bytes / self.ep
        # Output: [token_expert_pairs, intermediate/moe_tp]
        write_output = token_expert_pairs * self.intermediate_per_tp * self.act_transfer_bytes

        return read_input + read_weight + write_output
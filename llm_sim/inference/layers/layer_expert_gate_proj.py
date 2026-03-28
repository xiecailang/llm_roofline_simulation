"""Expert Gate投影算子 - MoE专家的Gate分支

hidden -> intermediate (ColumnParallel)

SwiGLU结构中的Gate分支，用于生成门控信号。
与Up分支独立计算，然后做element-wise乘法: output = silu(gate) * up
此算子同时用于 Shared Expert 和 Routed Expert。

FLOPs公式:
  hidden_size * (moe_intermediate_size / moe_tp) * 1 * top_k * moe_batch_size * seq_len * 2

公式解析:
  - hidden_size: 输入维度 (7168)
  - moe_intermediate_size / moe_tp: 每个MoE TP rank处理的intermediate维度
  - * 1: 单个分支（Gate）
  - top_k: 每个token路由的专家数 (routed=8, shared=1)
  - moe_batch_size: MoE层的有效batch size
  - seq_len: 序列长度
  - * 2: FLOPs系数 (每次乘加算2 FLOPs)

moe_batch_size计算:
  moe_batch_size = micro_batch_size / attention_tp * moe_tp

关键点:
  - EP不影响FLOPs! EP只影响通信和权重存储
  - Gate分支需要额外的SiLU激活（Vector操作）
  - Gate和Up的CUBE FLOPs相同（矩阵维度相同），区别在于Vector FLOPs
"""

from .layer_base import LayerBase


class LayerExpertGateProj(LayerBase):
    """Expert Gate: hidden -> intermediate (ColumnParallel)

    同时用于 Shared Expert 和 Routed Expert。
    """

    def __init__(self, hardware_config, model_config, deploy_config, quant_config, seq_len, top_k=1, is_shared=False):
        super().__init__(hardware_config, model_config, deploy_config, quant_config)
        self.seq_len = seq_len
        self.top_k = top_k  # routed expert=top_k, shared expert=1
        self.is_shared = is_shared  # Shared Expert 不使用 EP，完全复制
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
            1 *                                      # 单分支 (Gate)
            self.top_k *                             # num_experts_per_tok
            self.moe_batch_size *                    # MoE有效batch size
            self.seq_len *                           # 序列长度
            2                                        # FLOPs系数
        )
        return float(flops)

    def get_vector_flops(self):
        """SwiGLU的SiLU激活已在down_proj中统一建模

        为避免重复计算，gate_proj的vector FLOPs设为0。
        SwiGLU激活的所有vector操作 (SiLU + multiply = 7 FLOPs/element)
        已归入 LayerExpertDown.get_vector_flops()。

        这是融合内核视角的建模方式：gate/up是纯CUBE matmul，
        激活操作在down的CUBE之前连续执行。
        """
        return 0.0

    def get_mem_bytes(self):
        """访存量

        EP 切分的是专家，不是权重：
        - Routed Expert: 每个 EP rank 存储 num_experts_per_ep 个专家的完整权重
        - Shared Expert: 完全复制到每个 rank，不使用 EP
        """
        # 实际处理的token-expert对数
        token_expert_pairs = self.moe_batch_size * self.seq_len * self.top_k

        # Input: [token_expert_pairs, hidden]
        read_input = token_expert_pairs * self.hidden_size * self.act_transfer_bytes

        # Weight: [hidden, intermediate/moe_tp] per expert
        if self.is_shared:
            # Shared Expert: 完全复制，不使用 EP
            read_weight = (
                self.hidden_size * self.intermediate_per_tp *
                self.weight_bytes * self.num_shared_experts
            )
        else:
            # Routed Expert: EP 切分专家
            read_weight = (
                self.hidden_size * self.intermediate_per_tp *
                self.weight_bytes * self.num_experts_per_ep
            )

        # Output: [token_expert_pairs, intermediate/moe_tp]
        write_output = token_expert_pairs * self.intermediate_per_tp * self.act_transfer_bytes

        return read_input + read_weight + write_output
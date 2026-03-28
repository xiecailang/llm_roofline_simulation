"""Expert Gate+Up融合算子 - MoE专家的Gate和Up融合投影

hidden -> intermediate * 2 (ColumnParallel)

此算子同时用于 Shared Expert 和 Routed Expert。

FLOPs公式:
  2 * hidden_size * (moe_intermediate_size / moe_tp) * 2 * top_k * moe_batch_size * seq_len

公式解析:
  - hidden_size: 输入维度 (7168)
  - moe_intermediate_size / moe_tp: 每个TP rank处理的intermediate维度
  - * 2 (第一个): gate + up 两个分支 (SwiGLU)
  - * 2 (第二个): FLOPs系数 (每次乘加算2 FLOPs)
  - top_k: 每个token路由的专家数 (routed=8, shared=1)
  - moe_batch_size: MoE层的有效batch size = micro_batch_size / attention_tp * moe_tp
  - seq_len: 序列长度

关键点:
  - EP不影响FLOPs! EP只影响通信(All-to-All)和权重存储
"""

from .layer_base import LayerBase


class LayerExpertGateUp(LayerBase):
    """Expert Gate+Up: hidden -> intermediate * 2 (ColumnParallel)

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
        2 * hidden * (intermediate / moe_tp) * 2 * top_k * moe_batch_size * seq

        = 4 * hidden * (intermediate / moe_tp) * top_k * moe_batch_size * seq

        注意: EP不影响FLOPs!
        """
        flops = (
            2 *                                          # FLOPs系数
            self.hidden_size *                           # 输入维度
            self.intermediate_per_tp *                   # intermediate / moe_tp
            2 *                                          # gate + up 两个分支
            self.top_k *                                 # num_experts_per_tok
            self.moe_batch_size *                        # MoE有效batch size
            self.seq_len                                 # 序列长度
        )
        return float(flops)

    def get_vector_flops(self):
        """SwiGLU的SiLU激活已在down_proj中统一建模

        为避免重复计算，gate_up融合层的vector FLOPs设为0。
        SwiGLU激活的所有vector操作 (SiLU + multiply = 7 FLOPs/element)
        已归入 LayerExpertDown.get_vector_flops()。
        """
        return 0.0

    def get_mem_bytes(self):
        """访存量

        EP 切分的是专家，不是权重：
        - Routed Expert: 每个 EP rank 存储 num_experts_per_ep 个专家的完整权重
        - Shared Expert: 完全复制到每个 rank，不使用 EP
        """
        # 实际处理的token-expert对数 = moe_batch_size * seq * top_k (EP不影响)
        token_expert_pairs = self.moe_batch_size * self.seq_len * self.top_k

        # Input: [token_expert_pairs, hidden]
        read_input = token_expert_pairs * self.hidden_size * self.act_transfer_bytes
        # Weight: [hidden, intermediate/moe_tp * 2] per expert
        if self.is_shared:
            # Shared Expert: 完全复制，不使用 EP
            read_weight = (
                self.hidden_size * self.intermediate_per_tp * 2 *
                self.weight_bytes * self.num_shared_experts
            )
        else:
            # Routed Expert: EP 切分专家
            read_weight = (
                self.hidden_size * self.intermediate_per_tp * 2 *
                self.weight_bytes * self.num_experts_per_ep
            )
        # Output: [token_expert_pairs, intermediate/moe_tp * 2]
        write_output = token_expert_pairs * self.intermediate_per_tp * 2 * self.act_transfer_bytes

        return read_input + read_weight + write_output

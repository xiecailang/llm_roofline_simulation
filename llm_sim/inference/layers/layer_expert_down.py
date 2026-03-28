"""Expert Down算子 - MoE专家的Down投影

intermediate -> hidden (RowParallel)

SwiGLU结构中的Down投影，输入是 silu(gate) * up 的结果。
此算子同时用于 Shared Expert 和 Routed Expert。

FLOPs公式:
  hidden_size * (moe_intermediate_size / moe_tp) * 2 * top_k * moe_batch_size * seq_len * 2

公式解析:
  - hidden_size: 输出维度 (7168)
  - moe_intermediate_size / moe_tp: 每个MoE TP rank处理的intermediate维度
  - * 2 (第一个): FLOPs系数 (每次乘加算2 FLOPs)
  - top_k: 每个token路由的专家数 (routed=8, shared=1)
  - moe_batch_size: MoE层的有效batch size
  - seq_len: 序列长度
  - * 2 (第二个): SwiGLU结构因子
    Down的输入是 silu(gate) * up，其中 gate 和 up 各自维度为 intermediate
    融合后维度为 intermediate*2（gate和up的权重可以融合为一个大矩阵读取）

moe_batch_size计算:
  moe_batch_size = micro_batch_size / attention_tp * moe_tp

关键点:
  - EP不影响FLOPs! EP只影响通信(All-to-All)和权重存储
  - 因为每个token仍然需要经过top_k个专家的计算
  - EP只是把专家分布到不同节点，不减少总计算量
"""

from .layer_base import LayerBase


class LayerExpertDown(LayerBase):
    """Expert Down: intermediate -> hidden (RowParallel)

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
        hidden * (intermediate / moe_tp) * 2 * top_k * moe_batch_size * seq * 2

        = 4 * hidden * (intermediate / moe_tp) * top_k * moe_batch_size * seq

        注意: EP不影响FLOPs!
        """
        flops = (
            self.hidden_size *                       # 输出维度
            self.intermediate_per_tp *               # intermediate / moe_tp
            2 *                                      # FLOPs系数
            self.top_k *                             # num_experts_per_tok
            self.moe_batch_size *                    # MoE有效batch size
            self.seq_len *                           # 序列长度
            2                                        # SwiGLU结构因子
        )
        return float(flops)

    def get_vector_flops(self):
        """SwiGLU激活的Vector FLOPs (融合内核视角)

        SwiGLU公式: output = down(silu(gate) * up)

        在融合MoE内核中，Down投影前的激活操作包括：
        - SiLU(gate): x * sigmoid(x) ≈ 3 FLOPs (neg + exp近似 + reciprocal)
        - silu(gate) * up: 逐元素乘法 ≈ 2 FLOPs (2次读取 + 乘法)
        - 融合结果准备: ≈ 2 FLOPs (中间结果处理)
        合计: 3 + 2*2 = 7 FLOPs/element

        所有SwiGLU激活FLOPs统一在Down层建模，gate_proj不重复计算。
        这符合融合内核的实际执行方式：gate/up的CUBE计算完成后，
        激活操作在Down的CUBE之前连续执行。

        公式: moe_batch_size * seq_len * (intermediate/moe_tp) * top_k * 7
        """
        token_expert_pairs = self.moe_batch_size * self.seq_len * self.top_k
        return float(token_expert_pairs * self.intermediate_per_tp * 7)

    def get_mem_bytes(self):
        """访存量

        EP 切分的是专家，不是权重：
        - Routed Expert: 每个 EP rank 存储 num_experts_per_ep 个专家的完整权重
        - Shared Expert: 完全复制到每个 rank，不使用 EP
        """
        # 实际处理的token-expert对数
        token_expert_pairs = self.moe_batch_size * self.seq_len * self.top_k

        # Input: [token_expert_pairs, intermediate*2/moe_tp]
        # 注意：SwiGLU中gate和up的权重可以融合读取，维度为intermediate*2
        intermediate_dim = self.intermediate_per_tp * 2
        read_input = token_expert_pairs * intermediate_dim * self.act_transfer_bytes

        # Weight: [intermediate, hidden] per expert
        if self.is_shared:
            # Shared Expert: 完全复制，不使用 EP
            # 每个 rank 都存储完整的 shared expert 权重
            read_weight = (
                self.intermediate_per_tp * self.hidden_size *
                self.weight_bytes * self.num_shared_experts
            )
        else:
            # Routed Expert: EP 切分专家
            # 每个 EP rank 存储 num_experts_per_ep 个专家的完整权重
            read_weight = (
                self.intermediate_per_tp * self.hidden_size *
                self.weight_bytes * self.num_experts_per_ep
            )

        # Output: [token_expert_pairs, hidden]
        write_output = token_expert_pairs * self.hidden_size * self.act_transfer_bytes

        return read_input + read_weight + write_output
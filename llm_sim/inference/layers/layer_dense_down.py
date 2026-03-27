"""Dense FFN Down算子

intermediate -> hidden (RowParallel)

Dense FFN 用于 DeepSeek V3 的前几层 (first_k_dense_replace)，而非 MoE。

SwiGLU结构中的Down投影，输入是 silu(gate) * up 的结果。

FLOPs公式:
  hidden_size * (intermediate_size / tp) * 2 * batch_size * seq_len * 2

公式解析:
  - hidden_size: 输出维度 (7168)
  - intermediate_size / tp: 每个TP rank处理的intermediate维度
  - * 2 (第一个): FLOPs系数 (每次乘加算2 FLOPs)
  - batch_size: 批次大小
  - seq_len: 序列长度
  - * 2 (第二个): SwiGLU结构因子 (gate+up融合后的intermediate维度)

注意: Dense FFN 不涉及 EP，只有 TP
"""

from .layer_base import LayerBase


class LayerDenseDown(LayerBase):
    """Dense FFN Down: intermediate -> hidden (RowParallel)"""

    def __init__(self, hardware_config, model_config, deploy_config, quant_config, seq_len):
        super().__init__(hardware_config, model_config, deploy_config, quant_config)
        self.seq_len = seq_len
        self.intermediate_size = model_config.intermediate_size
        # Dense FFN 使用 attention_tp
        self.intermediate_per_tp = self.intermediate_size // self.tp

    def get_cube_flops(self):
        """计算量:
        hidden * (intermediate / tp) * 2 * batch * seq * 2

        = 4 * hidden * (intermediate / tp) * batch * seq
        """
        flops = (
            self.hidden_size *                   # 输出维度
            self.intermediate_per_tp *           # intermediate / tp
            2 *                                  # FLOPs系数
            self.batch_size *                    # batch size
            self.seq_len *                       # 序列长度
            2                                    # SwiGLU结构因子
        )
        return float(flops)

    def get_vector_flops(self):
        """SwiGLU激活的Vector FLOPs (融合内核视角)

        SwiGLU公式: output = down(silu(gate) * up)

        与Expert Down相同，所有SwiGLU激活FLOPs统一在Down层建模：
        - SiLU(gate): ≈ 3 FLOPs/element
        - silu(gate) * up: ≈ 2 FLOPs/element
        - 融合结果准备: ≈ 2 FLOPs/element
        合计: 3 + 2*2 = 7 FLOPs/element

        公式: batch_size * seq_len * (intermediate/tp) * 7
        """
        tokens = self.batch_size * self.seq_len
        return float(tokens * self.intermediate_per_tp * 7)

    def get_mem_bytes(self):
        """访存量"""
        tokens = self.batch_size * self.seq_len

        # Input: [tokens, intermediate * 2 / tp]
        # SwiGLU: gate 和 up 融合后维度是 intermediate * 2
        intermediate_dim = self.intermediate_per_tp * 2
        read_input = tokens * intermediate_dim * self.act_transfer_bytes

        # Weight: [intermediate, hidden]
        read_weight = self.intermediate_per_tp * self.hidden_size * self.weight_bytes

        # Output: [tokens, hidden]
        write_output = tokens * self.hidden_size * self.act_transfer_bytes

        return read_input + read_weight + write_output
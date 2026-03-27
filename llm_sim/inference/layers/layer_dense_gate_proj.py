"""Dense FFN Gate投影算子

hidden -> intermediate (ColumnParallel)

Dense FFN 用于 DeepSeek V3 的前几层 (first_k_dense_replace)，而非 MoE。

SwiGLU结构中的Gate分支，与Up分支独立计算。

FLOPs公式:
  hidden_size * (intermediate_size / tp) * 1 * batch_size * seq_len * 2

公式解析:
  - hidden_size: 输入维度 (7168)
  - intermediate_size / tp: 每个TP rank处理的intermediate维度
  - * 1: 单个分支（Gate）
  - batch_size: 批次大小
  - seq_len: 序列长度
  - * 2: FLOPs系数 (每次乘加算2 FLOPs)

注意: Dense FFN 不涉及 EP，只有 TP
"""

from .layer_base import LayerBase


class LayerDenseGateProj(LayerBase):
    """Dense FFN Gate: hidden -> intermediate (ColumnParallel)"""

    def __init__(self, hardware_config, model_config, deploy_config, quant_config, seq_len):
        super().__init__(hardware_config, model_config, deploy_config, quant_config)
        self.seq_len = seq_len
        self.intermediate_size = model_config.intermediate_size
        # Dense FFN 使用 attention_tp
        self.intermediate_per_tp = self.intermediate_size // self.tp

    def get_cube_flops(self):
        """计算量:
        hidden * (intermediate / tp) * 1 * batch * seq * 2
        """
        flops = (
            self.hidden_size *                   # 输入维度
            self.intermediate_per_tp *           # intermediate / tp
            1 *                                  # 单分支 (Gate)
            self.batch_size *                    # batch size
            self.seq_len *                       # 序列长度
            2                                    # FLOPs系数
        )
        return float(flops)

    def get_vector_flops(self):
        """SwiGLU的SiLU激活已在down_proj中统一建模

        为避免重复计算，gate_proj的vector FLOPs设为0。
        SwiGLU激活的所有vector操作 (SiLU + multiply = 7 FLOPs/element)
        已归入 LayerDenseDown.get_vector_flops()。
        """
        return 0.0

    def get_mem_bytes(self):
        """访存量"""
        tokens = self.batch_size * self.seq_len

        # Input: [tokens, hidden]
        read_input = tokens * self.hidden_size * self.act_transfer_bytes
        # Weight: [hidden, intermediate/tp]
        read_weight = self.hidden_size * self.intermediate_per_tp * self.weight_bytes
        # Output: [tokens, intermediate/tp]
        write_output = tokens * self.intermediate_per_tp * self.act_transfer_bytes

        return read_input + read_weight + write_output
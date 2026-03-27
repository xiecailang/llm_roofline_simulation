"""MLA Q LoRA压缩算子 - q_a_proj

hidden -> q_lora_rank

TP影响：ColumnParallel，每个TP节点计算 q_lora_rank/TP 的输出
"""

from .layer_base import LayerBase


class LayerMLAQAProj(LayerBase):
    """MLA Q LoRA压缩: q_a_proj (ColumnParallel)"""

    def __init__(self, hardware_config, model_config, deploy_config, quant_config, seq_len):
        super().__init__(hardware_config, model_config, deploy_config, quant_config)
        self.seq_len = seq_len
        self.q_lora_rank = getattr(model_config, 'q_lora_rank', 1536)
        # ColumnParallel: 输出维度按TP切分
        self.q_lora_rank_per_tp = self.q_lora_rank // self.tp

    def get_cube_flops(self):
        """计算量: 2 * batch * seq * hidden * (q_lora_rank / TP)

        ColumnParallel: 每个TP节点只计算 q_lora_rank/TP 的输出
        """
        return 2.0 * self.batch_size * self.seq_len * self.hidden_size * self.q_lora_rank_per_tp

    def get_vector_flops(self):
        return 0.0

    def get_mem_bytes(self):
        """访存量: 按TP切分"""
        # Input: [batch, seq, hidden] - 每个TP节点都需要读取完整输入
        read_input = self.batch_size * self.seq_len * self.hidden_size * self.act_transfer_bytes
        # Weight: [hidden, q_lora_rank/TP] - 每个TP节点只存储1/TP的权重
        read_weight = self.hidden_size * self.q_lora_rank_per_tp * self.weight_bytes
        # Output: [batch, seq, q_lora_rank/TP]
        write_output = self.batch_size * self.seq_len * self.q_lora_rank_per_tp * self.act_transfer_bytes

        return read_input + read_weight + write_output

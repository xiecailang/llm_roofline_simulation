"""MLA Q LoRA解压缩算子 - q_b_proj

q_lora_rank -> num_heads * qk_head_dim

TP影响：ColumnParallel，每个TP节点计算 (num_heads/TP) * qk_head_dim 的输出
"""

from .layer_base import LayerBase


class LayerMLAQBProj(LayerBase):
    """MLA Q LoRA解压缩: q_b_proj (ColumnParallel)"""

    def __init__(self, hardware_config, model_config, deploy_config, quant_config, seq_len):
        super().__init__(hardware_config, model_config, deploy_config, quant_config)
        self.seq_len = seq_len
        self.q_lora_rank = getattr(model_config, 'q_lora_rank', 1536)
        self.num_heads = model_config.num_attention_heads
        self.qk_nope_head_dim = getattr(model_config, 'qk_nope_head_dim', 128)
        self.qk_rope_head_dim = getattr(model_config, 'qk_rope_head_dim', 64)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        # ColumnParallel: num_heads按TP切分
        self.num_heads_per_tp = self.num_heads // self.tp

    def get_cube_flops(self):
        """计算量: 2 * batch * seq * q_lora_rank * (num_heads/TP * qk_head_dim)

        ColumnParallel: 每个TP节点只计算 num_heads/TP 的输出
        """
        output_dim = self.num_heads_per_tp * self.qk_head_dim
        return 2.0 * self.batch_size * self.seq_len * self.q_lora_rank * output_dim

    def get_vector_flops(self):
        return 0.0

    def get_mem_bytes(self):
        """访存量: 按TP切分"""
        output_dim = self.num_heads_per_tp * self.qk_head_dim

        # Input: [batch, seq, q_lora_rank] - 每个TP节点都需要读取完整输入
        read_input = self.batch_size * self.seq_len * self.q_lora_rank * self.act_transfer_bytes
        # Weight: [q_lora_rank, num_heads/TP * qk_head_dim]
        read_weight = self.q_lora_rank * output_dim * self.weight_bytes
        # Output: [batch, seq, num_heads/TP * qk_head_dim]
        write_output = self.batch_size * self.seq_len * output_dim * self.act_transfer_bytes

        return read_input + read_weight + write_output

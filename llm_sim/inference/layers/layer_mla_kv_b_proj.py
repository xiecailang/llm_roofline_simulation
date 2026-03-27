"""MLA KV LoRA解压缩算子 - kv_b_proj

kv_lora_rank -> num_heads * (qk_nope_head_dim + v_head_dim)

TP影响：ColumnParallel，每个TP节点计算 (num_heads/TP) * (qk_nope + v) 的输出
"""

from .layer_base import LayerBase


class LayerMLAKVBProj(LayerBase):
    """MLA KV LoRA解压缩: kv_b_proj (ColumnParallel)"""

    def __init__(self, hardware_config, model_config, deploy_config, quant_config, seq_len):
        super().__init__(hardware_config, model_config, deploy_config, quant_config)
        self.seq_len = seq_len
        self.kv_lora_rank = getattr(model_config, 'kv_lora_rank', 512)
        self.num_heads = model_config.num_attention_heads
        self.qk_nope_head_dim = getattr(model_config, 'qk_nope_head_dim', 128)
        self.v_head_dim = getattr(model_config, 'v_head_dim', 128)
        # ColumnParallel: num_heads按TP切分
        self.num_heads_per_tp = self.num_heads // self.tp

    def get_cube_flops(self):
        """计算量: 2 * batch * seq * kv_lora_rank * (num_heads/TP * (qk_nope + v))

        ColumnParallel: 每个TP节点只计算 num_heads/TP 的输出
        """
        output_dim = self.num_heads_per_tp * (self.qk_nope_head_dim + self.v_head_dim)
        return 2.0 * self.batch_size * self.seq_len * self.kv_lora_rank * output_dim

    def get_vector_flops(self):
        return 0.0

    def get_mem_bytes(self):
        """访存量: 按TP切分"""
        output_dim = self.num_heads_per_tp * (self.qk_nope_head_dim + self.v_head_dim)

        # Input: [batch, seq, kv_lora_rank]
        read_input = self.batch_size * self.seq_len * self.kv_lora_rank * self.act_transfer_bytes
        # Weight: [kv_lora_rank, num_heads/TP * (qk_nope + v)]
        read_weight = self.kv_lora_rank * output_dim * self.weight_bytes
        # Output: [batch, seq, num_heads/TP * (qk_nope + v)]
        write_output = self.batch_size * self.seq_len * output_dim * self.act_transfer_bytes

        return read_input + read_weight + write_output

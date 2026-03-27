"""MLA KV LoRA压缩算子 - kv_a_proj

hidden -> kv_lora_rank + qk_rope_head_dim

TP影响：ReplicatedLinear，每个TP节点计算相同的输出（不切分）
"""

from .layer_base import LayerBase


class LayerMLAKVAProj(LayerBase):
    """MLA KV LoRA压缩: kv_a_proj (ReplicatedLinear)"""

    def __init__(self, hardware_config, model_config, deploy_config, quant_config, seq_len):
        super().__init__(hardware_config, model_config, deploy_config, quant_config)
        self.seq_len = seq_len
        self.kv_lora_rank = getattr(model_config, 'kv_lora_rank', 512)
        self.qk_rope_head_dim = getattr(model_config, 'qk_rope_head_dim', 64)
        # ReplicatedLinear: 不按TP切分，每个节点计算相同的输出

    def get_cube_flops(self):
        """计算量: 2 * batch * seq * hidden * (kv_lora_rank + qk_rope_head_dim)

        ReplicatedLinear: 每个TP节点都计算完整的输出（不切分）
        """
        output_dim = self.kv_lora_rank + self.qk_rope_head_dim
        return 2.0 * self.batch_size * self.seq_len * self.hidden_size * output_dim

    def get_vector_flops(self):
        return 0.0

    def get_mem_bytes(self):
        """访存量: 不按TP切分"""
        output_dim = self.kv_lora_rank + self.qk_rope_head_dim

        # Input: [batch, seq, hidden]
        read_input = self.batch_size * self.seq_len * self.hidden_size * self.act_transfer_bytes
        # Weight: [hidden, kv_lora_rank + qk_rope_head_dim] - 每个TP节点都存储完整权重
        read_weight = self.hidden_size * output_dim * self.weight_bytes
        # Output: [batch, seq, kv_lora_rank + qk_rope_head_dim]
        write_output = self.batch_size * self.seq_len * output_dim * self.act_transfer_bytes

        return read_input + read_weight + write_output

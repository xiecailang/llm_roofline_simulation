"""Linear Attention (Gated DeltaNet) Q/K/V 投影算子

Qwen3.5 混合注意力模型中，线性注意力层使用独立的投影维度:
- Q: hidden_size → linear_num_value_heads × linear_value_head_dim
- K: hidden_size → linear_num_key_heads × linear_key_head_dim
- V: hidden_size → linear_num_value_heads × linear_value_head_dim
- O: linear_num_value_heads × linear_value_head_dim → hidden_size

与 Full Attention (GQA) 的差异:
- GQA: Q/K/V 使用 num_attention_heads/num_kv_heads × head_dim
- DeltaNet: Q/V 使用 linear_num_value_heads × linear_value_head_dim
           K 使用 linear_num_key_heads × linear_key_head_dim

TP 影响:
- Q/V: ColumnParallel, 按 TP 切分 linear_num_value_heads
- K: ColumnParallel, 按 TP 切分 linear_num_key_heads
- O: RowParallel, 按 TP 切分 linear_num_value_heads
"""

from .layer_base import LayerBase


class LayerLinearQProj(LayerBase):
    """Linear Attention Q 投影: hidden_size → linear_num_value_heads × linear_value_head_dim

    ColumnParallel: 输出按 TP 切分 (linear_num_value_heads/TP)
    """

    def __init__(self, hardware_config, model_config, deploy_config, quant_config, seq_len):
        super().__init__(hardware_config, model_config, deploy_config, quant_config)
        self.seq_len = seq_len
        self.num_value_heads = model_config.linear_num_value_heads
        self.value_head_dim = model_config.linear_value_head_dim

    def get_cube_flops(self):
        m = self.batch_size * self.seq_len
        k = self.hidden_size
        n = self.num_value_heads * self.value_head_dim
        return 2.0 * m * k * (n / self.tp)

    def get_mem_bytes(self):
        m = self.batch_size * self.seq_len
        k = self.hidden_size
        n = self.num_value_heads * self.value_head_dim
        read_hidden = m * k * self.act_transfer_bytes
        read_weight = k * (n / self.tp) * self.weight_bytes
        write_output = m * (n / self.tp) * self.act_transfer_bytes
        return read_hidden + read_weight + write_output


class LayerLinearKProj(LayerBase):
    """Linear Attention K 投影: hidden_size → linear_num_key_heads × linear_key_head_dim

    ColumnParallel: 输出按 TP 切分 (linear_num_key_heads/TP)
    """

    def __init__(self, hardware_config, model_config, deploy_config, quant_config, seq_len):
        super().__init__(hardware_config, model_config, deploy_config, quant_config)
        self.seq_len = seq_len
        self.num_key_heads = model_config.linear_num_key_heads
        self.key_head_dim = model_config.linear_key_head_dim

    def get_cube_flops(self):
        m = self.batch_size * self.seq_len
        k = self.hidden_size
        n = self.num_key_heads * self.key_head_dim
        return 2.0 * m * k * (n / self.tp)

    def get_mem_bytes(self):
        m = self.batch_size * self.seq_len
        k = self.hidden_size
        n = self.num_key_heads * self.key_head_dim
        read_hidden = m * k * self.act_transfer_bytes
        read_weight = k * (n / self.tp) * self.weight_bytes
        write_output = m * (n / self.tp) * self.act_transfer_bytes
        return read_hidden + read_weight + write_output


class LayerLinearVProj(LayerBase):
    """Linear Attention V 投影: hidden_size → linear_num_value_heads × linear_value_head_dim

    ColumnParallel: 输出按 TP 切分 (linear_num_value_heads/TP)
    """

    def __init__(self, hardware_config, model_config, deploy_config, quant_config, seq_len):
        super().__init__(hardware_config, model_config, deploy_config, quant_config)
        self.seq_len = seq_len
        self.num_value_heads = model_config.linear_num_value_heads
        self.value_head_dim = model_config.linear_value_head_dim

    def get_cube_flops(self):
        m = self.batch_size * self.seq_len
        k = self.hidden_size
        n = self.num_value_heads * self.value_head_dim
        return 2.0 * m * k * (n / self.tp)

    def get_mem_bytes(self):
        m = self.batch_size * self.seq_len
        k = self.hidden_size
        n = self.num_value_heads * self.value_head_dim
        read_hidden = m * k * self.act_transfer_bytes
        read_weight = k * (n / self.tp) * self.weight_bytes
        write_output = m * (n / self.tp) * self.act_transfer_bytes
        return read_hidden + read_weight + write_output


class LayerLinearOProj(LayerBase):
    """Linear Attention O 投影: linear_num_value_heads × linear_value_head_dim → hidden_size

    RowParallel: 输入按 TP 切分, 输出完整
    """

    def __init__(self, hardware_config, model_config, deploy_config, quant_config, seq_len):
        super().__init__(hardware_config, model_config, deploy_config, quant_config)
        self.seq_len = seq_len
        self.num_value_heads = model_config.linear_num_value_heads
        self.value_head_dim = model_config.linear_value_head_dim

    def get_cube_flops(self):
        m = self.batch_size * self.seq_len
        k = self.num_value_heads * self.value_head_dim
        n = self.hidden_size
        return 2.0 * m * (k / self.tp) * n

    def get_mem_bytes(self):
        m = self.batch_size * self.seq_len
        k = self.num_value_heads * self.value_head_dim
        n = self.hidden_size
        read_attn = m * (k / self.tp) * self.act_transfer_bytes
        read_weight = (k / self.tp) * n * self.weight_bytes
        write_output = m * n * self.act_transfer_bytes
        return read_attn + read_weight + write_output

"""GQA Q/K/V 投影算子

GQA (Grouped Query Attention) 投影层：
- Q projection: hidden_size → num_heads × head_dim (ColumnParallel)
- K projection: hidden_size → num_kv_heads × head_dim (ColumnParallel)
- V projection: hidden_size → num_kv_heads × head_dim (ColumnParallel)

与 MLA 的差异：
- MLA: Q 使用 LoRA 压缩 (q_a_proj → q_b_proj)，KV 使用 LoRA 压缩 (kv_a_proj → kv_b_proj)
- GQA: Q/K/V 直接投影，无 LoRA 压缩

TP 影响：
- Q: ColumnParallel, 输出按 TP 切分 (num_heads/TP × head_dim)
- K: ColumnParallel, 输出按 TP 切分 (num_kv_heads/TP × head_dim)
- V: ColumnParallel, 输出按 TP 切分 (num_kv_heads/TP × head_dim)

GQA ratio = num_heads / num_kv_heads (例如 Qwen 2.5 72B: 64/8 = 8)
"""

from .layer_base import LayerBase


class LayerGQAQProj(LayerBase):
    """GQA Q 投影: hidden_size → num_heads × head_dim

    ColumnParallel: 输出按 TP 切分
    """

    def __init__(self, hardware_config, model_config, deploy_config, quant_config, seq_len):
        super().__init__(hardware_config, model_config, deploy_config, quant_config)
        self.seq_len = seq_len

        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.head_dim

    def get_cube_flops(self):
        """Q projection: [B, S, hidden] @ [hidden, num_heads × head_dim]"""
        m = self.batch_size * self.seq_len
        k = self.hidden_size
        n = self.num_heads * self.head_dim
        # ColumnParallel: n 按 TP 切分
        return 2.0 * m * k * (n / self.tp)

    def get_mem_bytes(self):
        """读 hidden, 读 Q weight, 写 Q output"""
        m = self.batch_size * self.seq_len
        k = self.hidden_size
        n = self.num_heads * self.head_dim

        read_hidden = m * k * self.act_transfer_bytes
        read_weight = k * (n / self.tp) * self.weight_bytes
        write_output = m * (n / self.tp) * self.act_transfer_bytes

        return read_hidden + read_weight + write_output


class LayerGQAKProj(LayerBase):
    """GQA K 投影: hidden_size → num_kv_heads × head_dim

    ColumnParallel: 输出按 TP 切分
    """

    def __init__(self, hardware_config, model_config, deploy_config, quant_config, seq_len):
        super().__init__(hardware_config, model_config, deploy_config, quant_config)
        self.seq_len = seq_len

        self.num_kv_heads = model_config.num_key_value_heads
        self.head_dim = model_config.head_dim

    def get_cube_flops(self):
        """K projection: [B, S, hidden] @ [hidden, num_kv_heads × head_dim]"""
        m = self.batch_size * self.seq_len
        k = self.hidden_size
        n = self.num_kv_heads * self.head_dim
        # ColumnParallel: n 按 TP 切分
        return 2.0 * m * k * (n / self.tp)

    def get_mem_bytes(self):
        """读 hidden, 读 K weight, 写 K output"""
        m = self.batch_size * self.seq_len
        k = self.hidden_size
        n = self.num_kv_heads * self.head_dim

        read_hidden = m * k * self.act_transfer_bytes
        read_weight = k * (n / self.tp) * self.weight_bytes
        write_output = m * (n / self.tp) * self.act_transfer_bytes

        return read_hidden + read_weight + write_output


class LayerGQAVProj(LayerBase):
    """GQA V 投影: hidden_size → num_kv_heads × head_dim

    ColumnParallel: 输出按 TP 切分
    """

    def __init__(self, hardware_config, model_config, deploy_config, quant_config, seq_len):
        super().__init__(hardware_config, model_config, deploy_config, quant_config)
        self.seq_len = seq_len

        self.num_kv_heads = model_config.num_key_value_heads
        self.head_dim = model_config.head_dim

    def get_cube_flops(self):
        """V projection: [B, S, hidden] @ [hidden, num_kv_heads × head_dim]"""
        m = self.batch_size * self.seq_len
        k = self.hidden_size
        n = self.num_kv_heads * self.head_dim
        # ColumnParallel: n 按 TP 切分
        return 2.0 * m * k * (n / self.tp)

    def get_mem_bytes(self):
        """读 hidden, 读 V weight, 写 V output"""
        m = self.batch_size * self.seq_len
        k = self.hidden_size
        n = self.num_kv_heads * self.head_dim

        read_hidden = m * k * self.act_transfer_bytes
        read_weight = k * (n / self.tp) * self.weight_bytes
        write_output = m * (n / self.tp) * self.act_transfer_bytes

        return read_hidden + read_weight + write_output


class LayerGQAOProj(LayerBase):
    """GQA O 投影: num_heads × head_dim → hidden_size

    RowParallel: 输入按 TP 切分, 输出完整
    """

    def __init__(self, hardware_config, model_config, deploy_config, quant_config, seq_len):
        super().__init__(hardware_config, model_config, deploy_config, quant_config)
        self.seq_len = seq_len

        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.head_dim

    def get_cube_flops(self):
        """O projection: [B, S, num_heads × head_dim] @ [num_heads × head_dim, hidden]"""
        m = self.batch_size * self.seq_len
        k = self.num_heads * self.head_dim
        n = self.hidden_size
        # RowParallel: k 按 TP 切分
        return 2.0 * m * (k / self.tp) * n

    def get_mem_bytes(self):
        """读 attention output, 读 O weight, 写 final output"""
        m = self.batch_size * self.seq_len
        k = self.num_heads * self.head_dim
        n = self.hidden_size

        read_attn = m * (k / self.tp) * self.act_transfer_bytes
        read_weight = (k / self.tp) * n * self.weight_bytes
        write_output = m * n * self.act_transfer_bytes

        return read_attn + read_weight + write_output

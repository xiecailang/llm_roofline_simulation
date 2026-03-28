"""GQA (Grouped Query Attention) 计算算子

GQA 与 MLA 的关键差异：
- MLA: KV cache 存储压缩后的 latent (kv_lora_rank 维度)，需要 LoRA 解压
- GQA: KV cache 存储完整的 num_kv_heads × head_dim，无压缩

GQA 特性：
- 多个 Q head 共享同一组 KV head (group_size = num_heads / num_kv_heads)
- Prefill: 完整 attention (seq_len × kv_seq_len)
- Decode: 完整 attention (1 × kv_seq_len)，无 DSA 稀疏

TP 影响：
- 每个 TP 节点计算 num_heads/TP 个 Q head
- KV head 按 TP 切分: num_kv_heads/TP
- KV cache 每个 TP 节点读取 kv_seq_len × (num_kv_heads/TP) × head_dim
"""

from .layer_base import LayerBase


class LayerGQAAttention(LayerBase):
    """GQA (Grouped Query Attention) 计算

    Q @ K^T + softmax + score @ V

    KV head expansion:
    - Q: [B, num_heads/TP, S, head_dim]
    - K: [B, num_kv_heads/TP, KV_S, head_dim]
    - V: [B, num_kv_heads/TP, KV_S, head_dim]
    - 通过 repeat_kv 扩展 K, V 到 num_heads/TP 维度
    - group_size = (num_heads/TP) / (num_kv_heads/TP) = num_heads / num_kv_heads
    """

    def __init__(self, hardware_config, model_config, deploy_config, quant_config,
                 seq_len, kv_seq_len, is_prefill=True):
        super().__init__(hardware_config, model_config, deploy_config, quant_config)
        self.seq_len = seq_len
        self.kv_seq_len = kv_seq_len
        self.is_prefill = is_prefill

        # 基础参数
        self.num_heads = model_config.num_attention_heads
        self.num_kv_heads = model_config.num_key_value_heads
        self.head_dim = model_config.head_dim

        # TP 影响
        self.num_heads_per_tp = self.num_heads // self.tp
        self.num_kv_heads_per_tp = self.num_kv_heads // self.tp
        self.group_size = self.num_heads_per_tp // self.num_kv_heads_per_tp

    def get_cube_flops(self):
        """Q @ K^T + score @ V

        GQA 注意力计算量:
        - Q @ K^T: [B, H/TP, S, head_dim] @ [B, KV_H/TP, head_dim, KV_S]
          注意: 实际计算时 K/V 会 expand 到 H/TP 维度，但 FLOPs 不变
          因为每个 Q head 只对应一个 KV head (GQA)
        - 等效 FLOPs = 2 * B * (H/TP) * S * head_dim * KV_S
        """
        # Q @ K^T: 每个 Q head 与对应 KV head 做点积
        qk_flops = 2 * self.batch_size * self.num_heads_per_tp * self.seq_len * self.head_dim * self.kv_seq_len
        # score @ V
        sv_flops = 2 * self.batch_size * self.num_heads_per_tp * self.seq_len * self.kv_seq_len * self.head_dim
        return qk_flops + sv_flops

    def get_vector_flops(self):
        """Softmax + RoPE

        GQA RoPE: 只应用在 K 的 head_dim 维度 (不同于 MLA 的 qk_rope_head_dim)
        Qwen 2.5: RoPE 应用在完整的 head_dim (128) 上
        """
        # Softmax: exp + sum + div
        softmax_flops = 3 * self.batch_size * self.num_heads_per_tp * self.seq_len * self.kv_seq_len
        # RoPE: sin/cos + multiply (应用在 K 和 Q 的 head_dim 维度)
        rope_flops = 4 * self.batch_size * (self.num_heads_per_tp + self.num_kv_heads_per_tp) * self.seq_len * self.head_dim
        return softmax_flops + rope_flops

    def get_mem_bytes(self):
        """GQA 内存访问

        GQA KV cache: 每个 token 存储 num_kv_heads × head_dim (无压缩)
        MLA KV cache: 每个 token 存储 kv_lora_rank (压缩后)

        TP 影响: 每个 TP 节点读取 num_kv_heads/TP 个 KV head
        """
        # Q: [B, H/TP, S, head_dim]
        read_q = self.batch_size * self.num_heads_per_tp * self.seq_len * self.head_dim * self.act_transfer_bytes
        # KV cache 读取: [B, KV_H/TP, KV_S, head_dim]
        read_kv = self.batch_size * self.num_kv_heads_per_tp * self.kv_seq_len * self.head_dim * self.cache_read_bytes
        # Output: [B, H/TP, S, head_dim]
        write_out = self.batch_size * self.num_heads_per_tp * self.seq_len * self.head_dim * self.act_transfer_bytes

        return read_q + read_kv + write_out

"""MLA (Multi-head Latent Attention) 计算算子

Q @ K^T + softmax + score @ V

关键特性：
- MLA: KV cache存储压缩后的latent (kv_lora_rank维度)
- TP影响：每个TP节点只计算 num_heads/TP 的attention

注意：如果需要稀疏注意力 (DSA)，请使用 layer_dsa_attention.py
"""

from .layer_base import LayerBase


class LayerMLAAttention(LayerBase):
    """MLA (Multi-head Latent Attention) 计算 - 无稀疏"""

    def __init__(self, hardware_config, model_config, deploy_config, quant_config, seq_len, kv_seq_len, is_prefill=True):
        super().__init__(hardware_config, model_config, deploy_config, quant_config)
        self.seq_len = seq_len
        self.kv_seq_len = kv_seq_len
        self.is_prefill = is_prefill
        self.num_heads = model_config.num_attention_heads
        self.qk_nope_head_dim = getattr(model_config, 'qk_nope_head_dim', 128)
        self.qk_rope_head_dim = getattr(model_config, 'qk_rope_head_dim', 64)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = getattr(model_config, 'v_head_dim', 128)
        self.kv_lora_rank = getattr(model_config, 'kv_lora_rank', 512)
        # TP影响：每个TP节点只计算 num_heads/TP 的attention
        self.num_heads_per_tp = self.num_heads // self.tp

    def get_cube_flops(self):
        """Q @ K^T + score @ V

        MLA不使用稀疏注意力，计算完整的 seq_len × kv_seq_len
        TP影响：每个TP节点只计算 num_heads/TP 的attention
        """
        # Q @ K^T: [B, H/TP, S, qk_head_dim] @ [B, H/TP, qk_head_dim, KV_S]
        qk_flops = 2 * self.batch_size * self.num_heads_per_tp * self.seq_len * self.qk_head_dim * self.kv_seq_len
        # score @ V: [B, H/TP, S, KV_S] @ [B, H/TP, KV_S, v_head_dim]
        sv_flops = 2 * self.batch_size * self.num_heads_per_tp * self.seq_len * self.kv_seq_len * self.v_head_dim
        return qk_flops + sv_flops

    def get_vector_flops(self):
        """Softmax + RoPE

        TP影响：每个TP节点只计算 num_heads/TP 的softmax和rope
        """
        # Softmax: exp + sum + div
        softmax_flops = 3 * self.batch_size * self.num_heads_per_tp * self.seq_len * self.kv_seq_len
        # RoPE: sin/cos + multiply (只应用在qk_rope_head_dim维度)
        rope_flops = 4 * self.batch_size * self.num_heads_per_tp * self.seq_len * self.qk_rope_head_dim
        return softmax_flops + rope_flops

    def get_mem_bytes(self):
        """MLA的内存访问

        MLA优化：KV cache存储压缩后的latent
        - compressed_kv: kv_lora_rank 维度（latent）
        - k_pe: qk_rope_head_dim 维度（位置编码）
        - 总计: kv_lora_rank + qk_rope_head_dim per token per layer

        参考: DeepSeek FlashMLA deep-dive
        memory_accessed ≈ 2 * s_k * (kv_lora_rank + qk_rope_head_dim) bytes

        TP影响：每个TP节点读取完整的KV cache latent，但只处理 num_heads/TP 的Q
        """
        kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim  # MLA KV cache per token

        # Q: [B, H/TP, S, qk_head_dim]
        read_q = self.batch_size * self.num_heads_per_tp * self.seq_len * self.qk_head_dim * self.act_transfer_bytes
        # KV cache读取：latent + position encoding [B, KV_S, kv_lora_rank + qk_rope_head_dim]
        read_kv_latent = self.batch_size * self.kv_seq_len * kv_cache_dim * self.cache_read_bytes
        # Output: [B, H/TP, S, v_head_dim]
        write_out = self.batch_size * self.num_heads_per_tp * self.seq_len * self.v_head_dim * self.act_transfer_bytes

        return read_q + read_kv_latent + write_out

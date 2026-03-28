"""DSA (DeepSeek Sparse Attention) 计算算子

DSA 是 DeepSeek V3.2 引入的稀疏注意力机制，与 MLA 不同：
- MLA: Multi-head Latent Attention，使用 KV LoRA 压缩
- DSA: DeepSeek Sparse Attention，在 MLA 基础上增加稀疏注意力模式

关键特性：
- Prefill 阶段：完整 attention，计算所有 token
- Decode 阶段：稀疏 attention，只计算 topk 个重要 token
- 使用 index_topk 参数（绝对值），而不是比例
- TP影响：每个TP节点只计算 num_heads/TP 的attention
"""

from .layer_base import LayerBase


class LayerDSAAttention(LayerBase):
    """DSA (DeepSeek Sparse Attention) 计算"""

    def __init__(self, hardware_config, model_config, deploy_config, quant_config, seq_len, kv_seq_len, is_prefill=True):
        super().__init__(hardware_config, model_config, deploy_config, quant_config)
        self.seq_len = seq_len
        self.kv_seq_len = kv_seq_len
        self.is_prefill = is_prefill

        # 基础参数
        self.num_heads = model_config.num_attention_heads
        self.qk_nope_head_dim = getattr(model_config, 'qk_nope_head_dim', 128)
        self.qk_rope_head_dim = getattr(model_config, 'qk_rope_head_dim', 64)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = getattr(model_config, 'v_head_dim', 128)
        self.kv_lora_rank = getattr(model_config, 'kv_lora_rank', 512)

        # TP影响：每个TP节点只计算 num_heads/TP 的attention
        self.num_heads_per_tp = self.num_heads // self.tp

        # DSA 特有参数：使用 index_topk (绝对值) 而不是 sparse_ratio (比例)
        self.index_topk = getattr(model_config, 'index_topk', 256)  # 稀疏注意力选择的 token 数量
        self.index_n_heads = getattr(model_config, 'index_n_heads', 64)  # indexer 的 head 数量
        self.index_head_dim = getattr(model_config, 'index_head_dim', 128)  # indexer 的 head 维度

    def get_cube_flops(self):
        """Q @ K^T + score @ V

        DSA 的关键优化：
        - Prefill 阶段：完整 attention，计算量 = seq_len × kv_seq_len
        - Decode 阶段：稀疏 attention，计算量 = seq_len × index_topk

        注意：这里使用 index_topk (绝对值)，而不是 sparse_ratio × kv_seq_len
        """
        # 计算有效的 KV 序列长度
        if self.is_prefill:
            # Prefill 阶段：完整 attention
            effective_kv_len = self.kv_seq_len
        else:
            # Decode 阶段：稀疏 attention，只计算 topk 个 token
            effective_kv_len = min(self.index_topk, self.kv_seq_len)

        # Q @ K^T: [B, H/TP, S, qk_head_dim] @ [B, H/TP, qk_head_dim, effective_KV_S]
        qk_flops = 2 * self.batch_size * self.num_heads_per_tp * self.seq_len * self.qk_head_dim * effective_kv_len
        # score @ V: [B, H/TP, S, effective_KV_S] @ [B, H/TP, effective_KV_S, v_head_dim]
        sv_flops = 2 * self.batch_size * self.num_heads_per_tp * self.seq_len * effective_kv_len * self.v_head_dim

        return qk_flops + sv_flops

    def get_vector_flops(self):
        """Softmax + RoPE

        DSA 优化：Decode 阶段 softmax 只在 topk 个 token 上计算
        """
        if self.is_prefill:
            effective_kv_len = self.kv_seq_len
        else:
            effective_kv_len = min(self.index_topk, self.kv_seq_len)

        # Softmax: exp + sum + div (只在有效 token 上)
        softmax_flops = 3 * self.batch_size * self.num_heads_per_tp * self.seq_len * effective_kv_len
        # RoPE: sin/cos + multiply (只应用在 qk_rope_head_dim 维度)
        rope_flops = 4 * self.batch_size * self.num_heads_per_tp * self.seq_len * self.qk_rope_head_dim

        return softmax_flops + rope_flops

    def get_mem_bytes(self):
        """DSA + MLA 的内存访问

        DSA 优化：Decode 阶段只读取 topk 个 token 的 KV cache latent
        MLA 优化：KV cache 存储压缩后的 latent
        - compressed_kv: kv_lora_rank 维度（latent）
        - k_pe: qk_rope_head_dim 维度（位置编码）
        - 总计: kv_lora_rank + qk_rope_head_dim per token per layer

        参考: DeepSeek FlashMLA deep-dive
        Decode: memory_accessed ≈ 2 * s_k * (kv_lora_rank + qk_rope_head_dim) bytes
        """
        kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim  # MLA KV cache per token

        if self.is_prefill:
            effective_kv_len = self.kv_seq_len
        else:
            # DSA: decode 阶段只读取 topk 个 token 的 latent
            effective_kv_len = min(self.index_topk, self.kv_seq_len)

        # Q: [B, H/TP, S, qk_head_dim]
        read_q = self.batch_size * self.num_heads_per_tp * self.seq_len * self.qk_head_dim * self.act_transfer_bytes

        # KV cache 读取：latent + position encoding [B, effective_KV_S, kv_lora_rank + qk_rope_head_dim]
        read_kv_latent = self.batch_size * effective_kv_len * kv_cache_dim * self.cache_read_bytes

        # Output: [B, H/TP, S, v_head_dim]
        write_out = self.batch_size * self.num_heads_per_tp * self.seq_len * self.v_head_dim * self.act_transfer_bytes

        return read_q + read_kv_latent + write_out

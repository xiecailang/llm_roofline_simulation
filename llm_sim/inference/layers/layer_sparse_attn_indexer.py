"""Sparse Attention Indexer算子 - Lightning Indexer的核心操作

执行以下操作:
1. FP8量化indexer_k并写入K cache
2. 计算FP8 MQA attention (indexer_q @ indexer_k^T)
3. TopK选择 (prefill: 每个query选topk, decode: 选择全局topk)

关键特性：
- 使用FP8进行MQA计算 (高效率)
- 返回topk_indices供FlashMLA使用
- 包含K cache的写入

Vector FLOPs估算:
- FP8量化: batch * seq * index_head_dim
- MQA softmax: batch * seq * index_n_heads * kv_len
- TopK: batch * seq * index_n_heads * kv_len (比较操作)
"""

from .layer_base import LayerBase


class LayerSparseAttnIndexer(LayerBase):
    """Sparse Attention Indexer算子 (SparseAttnIndexer in vLLM)"""

    def __init__(self, hardware_config, model_config, deploy_config, quant_config,
                 seq_len, kv_seq_len, is_prefill=True):
        super().__init__(hardware_config, model_config, deploy_config, quant_config)

        self.seq_len = seq_len
        self.kv_seq_len = kv_seq_len
        self.is_prefill = is_prefill
        self.batch_size = deploy_config.micro_batch_size

        # Indexer参数
        self.index_n_heads = getattr(model_config, 'index_n_heads', 64)
        self.index_head_dim = getattr(model_config, 'index_head_dim', 128)
        self.index_topk = getattr(model_config, 'index_topk', 2048)

        # K cache参数 (FP8)
        self.cache_bytes = 1  # FP8 = 1 byte

    def get_cube_flops(self):
        """MQA attention计算的CUBE FLOPs

        MQA: Multi-Query Attention (所有heads共享同一个K)
        Q: [batch, seq, index_n_heads, index_head_dim]
        K: [batch, kv_seq, 1, index_head_dim] (共享)

        Q @ K^T: batch * seq * index_n_heads * kv_seq * index_head_dim * 2
        """
        # MQA: 每个query head与共享的K计算attention
        qk_flops = (
            self.batch_size * self.seq_len *
            self.index_n_heads * self.kv_seq_len *
            self.index_head_dim * 2
        )
        return qk_flops

    def get_vector_flops(self):
        """Vector计算量

        1. FP8量化K: ~4 FLOPs per element
        2. Softmax: ~5 FLOPs per element (exp, sum, div)
        3. TopK: ~log(k) * n comparisons per element
        """
        batch_seq = self.batch_size * self.seq_len

        # FP8量化
        quant_flops = batch_seq * self.index_head_dim * 4

        # Softmax (over kv_seq for each query head)
        softmax_flops = batch_seq * self.index_n_heads * self.kv_seq_len * 5

        # TopK (heap-based, ~log(k) comparisons)
        topk_flops = batch_seq * self.index_n_heads * self.kv_seq_len * 10

        return quant_flops + softmax_flops + topk_flops

    def get_mem_bytes(self):
        """访存量

        1. 读取indexer_q: batch * seq * index_n_heads * index_head_dim * cache_bytes
        2. 读取indexer_k: batch * kv_seq * index_head_dim * cache_bytes (FP8 from cache)
        3. 写入K cache: batch * kv_seq * index_head_dim * cache_bytes (FP8)
        4. 输出topk_indices: batch * seq * index_topk * 4 (int32 indices)
        """
        # Indexer Q (FP8量化后)
        q_mem = self.batch_size * self.seq_len * self.index_n_heads * self.index_head_dim * self.cache_bytes

        # Indexer K (FP8 from cache, 只读)
        k_mem = self.batch_size * self.kv_seq_len * self.index_head_dim * self.cache_bytes

        # K cache写入 (FP8)
        k_cache_write = self.batch_size * self.kv_seq_len * self.index_head_dim * self.cache_bytes

        # TopK indices输出 (int32)
        output_mem = self.batch_size * self.seq_len * self.index_topk * 4

        return q_mem + k_mem + k_cache_write + output_mem

    def get_comm_bytes(self):
        """通信量 (无跨设备通信)"""
        return 0

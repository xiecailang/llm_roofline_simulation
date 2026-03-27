"""Indexer Q投影层 - Lightning Indexer的Q投影

从q_lora_rank空间投影到index_n_heads * index_head_dim空间。

关键特性：
- NO tensor parallel (ReplicatedLinear)
- 输入: q_c (q_lora_rank维度，来自q_a_norm)
- 输出: indexer_q (index_n_heads * index_head_dim维度)

FLOPs: 2 * batch * seq * q_lora_rank * (index_n_heads * index_head_dim)
"""

from .layer_base import LayerBase


class LayerIndexerWQProj(LayerBase):
    """Indexer Q投影层 (wq_b in vLLM)"""

    def __init__(self, hardware_config, model_config, deploy_config, quant_config, seq_len):
        super().__init__(hardware_config, model_config, deploy_config, quant_config)

        self.seq_len = seq_len
        self.batch_size = deploy_config.micro_batch_size

        # Indexer参数
        self.q_lora_rank = getattr(model_config, 'q_lora_rank', 1536)
        self.index_n_heads = getattr(model_config, 'index_n_heads', 64)
        self.index_head_dim = getattr(model_config, 'index_head_dim', 128)

        # 量化 - 使用indexer专用精度或默认激活精度
        self.act_bits = getattr(quant_config, 'indexer_activation_bits',
                                 quant_config.default_activation_compute_bits)
        self.weight_bits = quant_config.default_weight_bits

    def get_cube_flops(self):
        """计算CUBE FLOPs

        FLOPs = 2 * M * K * N (矩阵乘法)
        M = batch * seq
        K = q_lora_rank
        N = index_n_heads * index_head_dim
        """
        m = self.batch_size * self.seq_len
        k = self.q_lora_rank
        n = self.index_n_heads * self.index_head_dim
        return 2 * m * k * n

    def get_vector_flops(self):
        """Vector计算量 (无)"""
        return 0

    def get_mem_bytes(self):
        """计算访存量

        权重: q_lora_rank * (index_n_heads * index_head_dim) * weight_bytes
        激活输入: batch * seq * q_lora_rank * act_bytes
        激活输出: batch * seq * (index_n_heads * index_head_dim) * act_bytes
        """
        weight_bytes = self.weight_bits / 8
        act_bytes = self.act_bits / 8

        # 权重访存
        weight_mem = self.q_lora_rank * self.index_n_heads * self.index_head_dim * weight_bytes

        # 激活访存
        input_mem = self.batch_size * self.seq_len * self.q_lora_rank * act_bytes
        output_mem = self.batch_size * self.seq_len * self.index_n_heads * self.index_head_dim * act_bytes

        return weight_mem + input_mem + output_mem

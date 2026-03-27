"""Indexer K投影层 - Lightning Indexer的K投影

从hidden_size空间投影到index_head_dim空间。

关键特性：
- NO tensor parallel (ReplicatedLinear)
- 输入: hidden_states (hidden_size维度)
- 输出: indexer_k (index_head_dim维度)

FLOPs: 2 * batch * seq * hidden_size * index_head_dim
"""

from .layer_base import LayerBase


class LayerIndexerWKProj(LayerBase):
    """Indexer K投影层 (wk in vLLM)"""

    def __init__(self, hardware_config, model_config, deploy_config, quant_config, seq_len):
        super().__init__(hardware_config, model_config, deploy_config, quant_config)

        self.seq_len = seq_len
        self.batch_size = deploy_config.micro_batch_size
        self.hidden_size = model_config.hidden_size

        # Indexer参数
        self.index_head_dim = getattr(model_config, 'index_head_dim', 128)

        # 量化 - 使用indexer专用精度或默认激活精度
        self.act_bits = getattr(quant_config, 'indexer_activation_bits',
                                 quant_config.default_activation_compute_bits)
        self.weight_bits = quant_config.default_weight_bits

    def get_cube_flops(self):
        """计算CUBE FLOPs

        FLOPs = 2 * M * K * N (矩阵乘法)
        M = batch * seq
        K = hidden_size
        N = index_head_dim
        """
        m = self.batch_size * self.seq_len
        k = self.hidden_size
        n = self.index_head_dim
        return 2 * m * k * n

    def get_vector_flops(self):
        """Vector计算量 (无)"""
        return 0

    def get_mem_bytes(self):
        """计算访存量

        权重: hidden_size * index_head_dim * weight_bytes
        激活输入: batch * seq * hidden_size * act_bytes
        激活输出: batch * seq * index_head_dim * act_bytes
        """
        weight_bytes = self.weight_bits / 8
        act_bytes = self.act_bits / 8

        # 权重访存
        weight_mem = self.hidden_size * self.index_head_dim * weight_bytes

        # 激活访存
        input_mem = self.batch_size * self.seq_len * self.hidden_size * act_bytes
        output_mem = self.batch_size * self.seq_len * self.index_head_dim * act_bytes

        return weight_mem + input_mem + output_mem

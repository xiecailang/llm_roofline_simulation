"""Indexer权重投影层 - Lightning Indexer的MQA权重计算

从hidden_size空间投影到index_n_heads空间，用于MQA的attention权重。

关键特性：
- NO tensor parallel (ReplicatedLinear)
- NO quantization (quant_config=None in vLLM)
- 输入: hidden_states (hidden_size维度)
- 输出: weights (index_n_heads维度)

FLOPs: 2 * batch * seq * hidden_size * index_n_heads
"""

from .layer_base import LayerBase


class LayerIndexerWeightsProj(LayerBase):
    """Indexer权重投影层 (weights_proj in vLLM)"""

    def __init__(self, hardware_config, model_config, deploy_config, quant_config, seq_len):
        super().__init__(hardware_config, model_config, deploy_config, quant_config)

        self.seq_len = seq_len
        self.batch_size = deploy_config.micro_batch_size
        self.hidden_size = model_config.hidden_size

        # Indexer参数
        self.index_n_heads = getattr(model_config, 'index_n_heads', 64)

        # 量化 - indexer weights_proj不量化 (quant_config=None in vLLM)
        # 使用FP32进行计算
        self.act_bits = 32  # FP32, no quantization
        self.weight_bits = 32  # FP32

    def get_cube_flops(self):
        """计算CUBE FLOPs

        FLOPs = 2 * M * K * N (矩阵乘法)
        M = batch * seq
        K = hidden_size
        N = index_n_heads
        """
        m = self.batch_size * self.seq_len
        k = self.hidden_size
        n = self.index_n_heads
        return 2 * m * k * n

    def get_vector_flops(self):
        """Vector计算量 (无)"""
        return 0

    def get_mem_bytes(self):
        """计算访存量

        权重: hidden_size * index_n_heads * weight_bytes (FP32)
        激活输入: batch * seq * hidden_size * act_bytes
        激活输出: batch * seq * index_n_heads * act_bytes
        """
        weight_bytes = self.weight_bits / 8  # 4 bytes (FP32)
        act_bytes = self.act_bits / 8

        # 权重访存
        weight_mem = self.hidden_size * self.index_n_heads * weight_bytes

        # 激活访存
        input_mem = self.batch_size * self.seq_len * self.hidden_size * act_bytes
        output_mem = self.batch_size * self.seq_len * self.index_n_heads * act_bytes

        return weight_mem + input_mem + output_mem

"""Indexer K Norm层 - Lightning Indexer的K归一化

对indexer_k进行LayerNorm归一化。

关键特性：
- LayerNorm (不是RMSNorm!)
- eps = 1e-6
- 输入/输出维度: index_head_dim

Vector FLOPs公式:
  每个元素约8 FLOPs:
  - sum(x): ~2 FLOPs (归约)
  - x - mean: 1 sub
  - (x-mean)²: 1 mul
  - sum((x-mean)²): ~2 FLOPs (归约)
  - sqrt(var+eps): ~4 FLOPs
  - (x-mean)/sqrt: 1 div
  - * gamma: 1 mul
  - + beta: 1 add
  简化为 8 FLOPs/element

Memory公式:
  batch * seq * hidden * 2 * dtype (读输入 + 写输出)
  (权重gamma/beta访问可忽略)
"""

from .layer_base import LayerBase


class LayerIndexerKNorm(LayerBase):
    """Indexer K Norm层 (k_norm in vLLM)"""

    def __init__(self, hardware_config, model_config, deploy_config, quant_config, seq_len):
        super().__init__(hardware_config, model_config, deploy_config, quant_config)

        self.seq_len = seq_len
        # 有效batch = micro_batch_size / attention_tp
        self.effective_batch = self.batch_size / self.tp

        # Indexer参数
        self.index_head_dim = getattr(model_config, 'index_head_dim', 128)

    def get_cube_flops(self):
        """LayerNorm无CUBE计算"""
        return 0

    def get_vector_flops(self):
        """计算Vector FLOPs

        LayerNorm操作:
        - sum(x): ~2 FLOPs/element
        - x - mean: 1 FLOP/element
        - (x-mean)²: 1 FLOP/element
        - sum((x-mean)²): ~2 FLOPs/element
        - sqrt(var+eps): ~4 FLOPs (scalar)
        - (x-mean)/sqrt: 1 FLOP/element
        - * gamma: 1 FLOP/element
        - + beta: 1 FLOP/element
        总计约 8 FLOPs/element
        """
        return 8 * self.effective_batch * self.seq_len * self.index_head_dim

    def get_mem_bytes(self):
        """计算访存量

        简化公式: 2 * batch * seq * hidden * dtype (读输入 + 写输出)
        (权重gamma/beta访问可忽略)
        """
        return 2 * self.effective_batch * self.seq_len * self.index_head_dim * self.act_transfer_bytes

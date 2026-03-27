"""RMSNorm算子

y = x * rsqrt(mean(x²) + eps) * weight

Vector FLOPs公式:
  每个元素约6 FLOPs:
  - x²: 1 mul
  - sum(x²): ~2 FLOPs (归约)
  - rsqrt(mean + eps): ~4 FLOPs
  - x * rsqrt: 1 mul
  - * weight: 1 mul (可融合)
  简化为 6 FLOPs/element

Memory公式:
  - 读输入: batch * seq * hidden * dtype
  - 读权重: hidden * dtype (可忽略)
  - 写输出: batch * seq * hidden * dtype
  总计约: batch * seq * hidden * 2 * dtype

batch_size计算:
  effective_batch = micro_batch_size / attention_tp
  (与其他MoE层保持一致)
"""

from .layer_base import LayerBase


class LayerRMSNorm(LayerBase):
    """RMSNorm算子"""

    def __init__(self, hardware_config, model_config, deploy_config, quant_config, seq_len, norm_size):
        super().__init__(hardware_config, model_config, deploy_config, quant_config)
        self.seq_len = seq_len
        self.norm_size = norm_size

        # 有效batch = micro_batch_size / attention_tp
        # Norm是replicated操作，但为了与其他层保持一致，使用相同的batch计算方式
        self.effective_batch = self.batch_size / self.tp

    def get_cube_flops(self):
        """RMSNorm不使用CUBE"""
        return 0.0

    def get_vector_flops(self):
        """RMSNorm: x², sum, rsqrt, mul

        每个元素约6 FLOPs:
        - x²: 1 mul
        - sum归约: ~2 FLOPs
        - rsqrt: ~4 FLOPs (近似)
        - x * rsqrt: 1 mul
        - * weight: 1 mul (可融合到上一步)

        简化为 6 FLOPs/element
        """
        return 6 * self.effective_batch * self.seq_len * self.norm_size

    def get_mem_bytes(self):
        """读输入和权重，写输出

        简化公式: batch * seq * hidden * 2 * dtype
        (权重访问可忽略)
        """
        # 简化公式: 2 * batch * seq * hidden * dtype (读输入 + 写输出)
        return 2 * self.effective_batch * self.seq_len * self.norm_size * self.act_transfer_bytes

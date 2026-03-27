"""Dense FFN模块 - 标准SwiGLU FFN

用于 DeepSeek V3 的前几层 (first_k_dense_replace)，而非 MoE。

算子序列:
1. gate_proj: hidden -> intermediate (ColumnParallel) + SiLU
2. up_proj: hidden -> intermediate (ColumnParallel)
3. elementwise_mul: silu(gate) * up
4. down_proj: intermediate -> hidden (RowParallel)
5. AllReduce (TP通信)

FLOPs公式 (总计):
  Gate: hidden * (intermediate / tp) * 1 * batch * seq * 2
  Up:   hidden * (intermediate / tp) * 1 * batch * seq * 2
  Down: hidden * (intermediate / tp) * 2 * batch * seq * 2

  Total: 4 * hidden * (intermediate / tp) * batch * seq

与 MoE 的区别:
  - Dense FFN: 所有 token 经过同一个 FFN
  - MoE: token 被路由到不同的 expert

Dense FFN 使用 attention_tp，不涉及 EP
"""

from .module_base import ModuleBase
from ..layers import (
    LayerDenseGateProj,
    LayerDenseUp,
    LayerDenseDown,
    LayerAllReduce,
)


class ModuleDenseFFN(ModuleBase):
    """Dense FFN模块 (标准SwiGLU)"""

    def __init__(self, hardware_config, model_config, deploy_config, quant_config, seq_len):
        super().__init__(hardware_config, model_config, deploy_config, quant_config)

        self.seq_len = seq_len
        self.hidden_size = model_config.hidden_size
        self.tp = deploy_config.attention_tp

        self._build_layers()

    def _build_layers(self):
        """构建Dense FFN的所有算子"""
        batch_size = self.deploy_config.micro_batch_size
        act_bytes = self.quant_config.default_activation_transfer_bits / 8

        # 1. Gate Projection (ColumnParallel + SiLU)
        self.add_layer(
            'gate_proj',
            LayerDenseGateProj(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len
            )
        )

        # 2. Up Projection (ColumnParallel)
        self.add_layer(
            'up_proj',
            LayerDenseUp(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len
            )
        )

        # 3. Down Projection (RowParallel)
        self.add_layer(
            'down_proj',
            LayerDenseDown(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len
            )
        )

        # 4. AllReduce (TP通信)
        if self.tp > 1:
            data_size = batch_size * self.seq_len * self.hidden_size * act_bytes
            self.add_layer(
                'allreduce',
                LayerAllReduce(
                    self.hardware_config, self.model_config,
                    self.deploy_config, self.quant_config,
                    data_size, self.tp
                )
            )
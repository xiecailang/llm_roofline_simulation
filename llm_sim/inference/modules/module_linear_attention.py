"""Linear Attention (Gated DeltaNet) 模块 - 用于 Qwen3.5 混合注意力模型

Gated DeltaNet Linear Attention 结构:
1. [可选] AllGather Input (上游 TP < attention_TP)
2. Input RMSNorm
3. Q Projection (ColumnParallel): hidden → linear_num_value_heads × linear_value_head_dim
4. K Projection (ColumnParallel): hidden → linear_num_key_heads × linear_key_head_dim
5. V Projection (ColumnParallel): hidden → linear_num_value_heads × linear_value_head_dim
6. Linear Attention (Gated DeltaNet): state update + query
7. O Projection (RowParallel): linear_num_value_heads × linear_value_head_dim → hidden
8. [可选] AllReduce Output (attention_TP > 1)
9. [可选] ReduceScatter Output (attention_TP > 下游 TP)

与 GQA Attention 的关键差异:
- 投影维度不同 (linear_num_value_heads/linear_num_key_heads vs num_attention_heads/num_kv_heads)
- 注意力计算: 线性注意力 vs softmax attention
- Decode: O(d²) 恒定计算量 vs O(T×d) 随 input_length 增长
- 无 growing KV cache (固定大小状态)
"""

from .module_base import ModuleBase
from .module_attention_comm import ModuleAttentionTPComm
from ..layers import (
    LayerRMSNorm,
    LayerLinearQProj,
    LayerLinearKProj,
    LayerLinearVProj,
    LayerLinearOProj,
    LayerLinearAttention,
)


class ModuleLinearAttention(ModuleBase):
    """Gated DeltaNet Linear Attention 模块"""

    def __init__(self, hardware_config, model_config, deploy_config, quant_config,
                 seq_len, is_prefill=True, upstream_tp=None, downstream_tp=None):
        """初始化 Linear Attention 模块

        Args:
            seq_len: 查询序列长度
            is_prefill: 是否为 Prefill 阶段
            upstream_tp: 上游 TP 级别
            downstream_tp: 下游 TP 级别
        """
        super().__init__(hardware_config, model_config, deploy_config, quant_config)

        self.seq_len = seq_len
        self.is_prefill = is_prefill
        self.hidden_size = model_config.hidden_size
        self.attention_tp = deploy_config.attention_tp
        self.upstream_tp = upstream_tp or self.attention_tp
        self.downstream_tp = downstream_tp or self.attention_tp

        self._build_layers()

    def _build_layers(self):
        """构建 Linear Attention 的所有算子"""
        batch_size = self.deploy_config.micro_batch_size

        # ========== 1. [可选] AllGather Input ==========
        if self.upstream_tp < self.attention_tp:
            comm_module = ModuleAttentionTPComm(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                batch_size, self.seq_len, self.hidden_size,
                self.attention_tp, self.upstream_tp, self.attention_tp
            )
            for name, layer in comm_module.layers.items():
                if 'allgather' in name:
                    self.add_layer('allgather_input', layer)

        # ========== 2. Input RMSNorm ==========
        self.add_layer(
            'input_norm',
            LayerRMSNorm(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len, self.hidden_size
            )
        )

        # ========== 3. Q Projection (ColumnParallel) ==========
        self.add_layer(
            'q_proj',
            LayerLinearQProj(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len
            )
        )

        # ========== 4. K Projection (ColumnParallel) ==========
        self.add_layer(
            'k_proj',
            LayerLinearKProj(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len
            )
        )

        # ========== 5. V Projection (ColumnParallel) ==========
        self.add_layer(
            'v_proj',
            LayerLinearVProj(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len
            )
        )

        # ========== 6. Linear Attention (Gated DeltaNet) ==========
        self.add_layer(
            'attention',
            LayerLinearAttention(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len, self.is_prefill
            )
        )

        # ========== 7. O Projection (RowParallel) ==========
        self.add_layer(
            'o_proj',
            LayerLinearOProj(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len
            )
        )

        # ========== 8. [可选] AllReduce Output ==========
        if self.attention_tp > 1:
            comm_module = ModuleAttentionTPComm(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                batch_size, self.seq_len, self.hidden_size,
                self.attention_tp, self.attention_tp, self.attention_tp
            )
            for name, layer in comm_module.layers.items():
                if 'allreduce' in name:
                    self.add_layer('allreduce_output', layer)

        # ========== 9. [可选] ReduceScatter Output ==========
        if self.attention_tp > self.downstream_tp:
            comm_module = ModuleAttentionTPComm(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                batch_size, self.seq_len, self.hidden_size,
                self.attention_tp, self.attention_tp, self.downstream_tp
            )
            for name, layer in comm_module.layers.items():
                if 'reduce_scatter' in name:
                    self.add_layer('reduce_scatter_output', layer)

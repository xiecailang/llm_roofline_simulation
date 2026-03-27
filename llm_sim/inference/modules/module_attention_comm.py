"""Attention通信模块 - 可复用的通信算子

设计原则：
1. 通信算子与具体 Attention 实现解耦
2. 不同 Attention 类型（MLA/DSA/GQA/MHA）可复用同一套通信模块
3. 根据 TP 配置自动选择通信模式

通信算子列表：
1. `allgather_tp` - AllGather (TP通信)
2. `reduce_scatter_tp` - ReduceScatter (TP通信)
3. `allreduce_tp` - AllReduce (TP通信)

使用场景：
- Attention 前后：根据上游/下游的 TP 配置决定
- o_proj 后：RowParallel 需要 AllReduce
"""

from .module_base import ModuleBase
from ..layers import LayerAllGather, LayerReduceScatter, LayerAllReduce


class ModuleAttentionCommBase(ModuleBase):
    """Attention通信模块基类"""

    def __init__(self, hardware_config, model_config, deploy_config, quant_config,
                 data_size, num_devices, comm_name='comm'):
        super().__init__(hardware_config, model_config, deploy_config, quant_config)
        self.data_size = data_size
        self.num_devices = num_devices
        self.comm_name = comm_name


class ModuleAttentionAllGatherTP(ModuleAttentionCommBase):
    """Attention AllGather TP通信

    使用场景：
    - 上游 TP < 下游 TP：需要 AllGather 恢复完整激活
    - 例如：Attention TP=1, MoE TP=8，MoE前需要 AllGather

    通信量公式：
    comm_bytes = (N-1) / N * data_size

    时延公式：
    latency = data_size / (bandwidth * N) + rtt_overhead
    """

    def __init__(self, hardware_config, model_config, deploy_config, quant_config,
                 batch_size, seq_len, hidden_size, num_devices):
        act_bytes = quant_config.default_activation_transfer_bits / 8
        data_size = batch_size * seq_len * hidden_size * act_bytes

        super().__init__(hardware_config, model_config, deploy_config, quant_config,
                         data_size, num_devices, 'allgather_tp')

        self.add_layer(
            'allgather_tp',
            LayerAllGather(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.data_size, self.num_devices
            )
        )


class ModuleAttentionReduceScatterTP(ModuleAttentionCommBase):
    """Attention ReduceScatter TP通信

    使用场景：
    - 上游 TP > 下游 TP：需要 ReduceScatter 切分激活
    - 例如：Attention TP=8, MoE TP=1，MoE后需要 ReduceScatter

    通信量公式：
    comm_bytes = (N-1) / N * data_size

    时延公式：
    latency = data_size / (bandwidth * N) + rtt_overhead
    """

    def __init__(self, hardware_config, model_config, deploy_config, quant_config,
                 batch_size, seq_len, hidden_size, num_devices):
        act_bytes = quant_config.default_activation_transfer_bits / 8
        data_size = batch_size * seq_len * hidden_size * act_bytes

        super().__init__(hardware_config, model_config, deploy_config, quant_config,
                         data_size, num_devices, 'reduce_scatter_tp')

        self.add_layer(
            'reduce_scatter_tp',
            LayerReduceScatter(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.data_size, self.num_devices
            )
        )


class ModuleAttentionAllReduceTP(ModuleAttentionCommBase):
    """Attention AllReduce TP通信

    使用场景：
    - o_proj 后 (RowParallel)：聚合各 head 的结果
    - TP > 1 时需要

    通信量公式：
    comm_bytes = 2 * (N-1) / N * data_size

    时延公式：
    latency = 2 * data_size / (bandwidth * N) + rtt_overhead
    """

    def __init__(self, hardware_config, model_config, deploy_config, quant_config,
                 batch_size, seq_len, hidden_size, num_devices):
        act_bytes = quant_config.default_activation_transfer_bits / 8
        data_size = batch_size * seq_len * hidden_size * act_bytes

        super().__init__(hardware_config, model_config, deploy_config, quant_config,
                         data_size, num_devices, 'allreduce_tp')

        self.add_layer(
            'allreduce_tp',
            LayerAllReduce(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.data_size, self.num_devices
            )
        )


class ModuleAttentionTPComm(ModuleBase):
    """Attention TP通信统一管理

    根据 upstream_tp 和 downstream_tp 自动选择通信模式：
    - upstream_tp < downstream_tp: AllGather (恢复完整激活)
    - upstream_tp > downstream_tp: ReduceScatter (切分激活)
    - upstream_tp = downstream_tp: 无需通信

    同时处理 o_proj 后的 AllReduce (当 attention_tp > 1)
    """

    def __init__(self, hardware_config, model_config, deploy_config, quant_config,
                 batch_size, seq_len, hidden_size,
                 attention_tp, upstream_tp=None, downstream_tp=None):
        super().__init__(hardware_config, model_config, deploy_config, quant_config)

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.attention_tp = attention_tp
        self.upstream_tp = upstream_tp or attention_tp
        self.downstream_tp = downstream_tp or attention_tp

        self._build_comm_layers()

    def _build_comm_layers(self):
        """根据TP配置构建通信层"""
        act_bytes = self.quant_config.default_activation_transfer_bits / 8
        data_size = self.batch_size * self.seq_len * self.hidden_size * act_bytes

        # 1. AllGather (上游TP < 当前TP)
        if self.upstream_tp < self.attention_tp:
            self.add_layer(
                'allgather_input',
                LayerAllGather(
                    self.hardware_config, self.model_config,
                    self.deploy_config, self.quant_config,
                    data_size, self.attention_tp
                )
            )

        # 2. AllReduce (o_proj后，TP > 1)
        if self.attention_tp > 1:
            self.add_layer(
                'allreduce_output',
                LayerAllReduce(
                    self.hardware_config, self.model_config,
                    self.deploy_config, self.quant_config,
                    data_size, self.attention_tp
                )
            )

        # 3. ReduceScatter (当前TP > 下游TP)
        if self.attention_tp > self.downstream_tp:
            self.add_layer(
                'reduce_scatter_output',
                LayerReduceScatter(
                    self.hardware_config, self.model_config,
                    self.deploy_config, self.quant_config,
                    data_size, self.downstream_tp
                )
            )

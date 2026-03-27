"""P2P通信算子 - Pipeline Parallel stage间通信

Pipeline Parallel (PP) 中，不同 stage 之间需要 P2P 通信传递激活值。

通信模式：
- Send: 当前 stage 发送激活到下一个 stage
- Recv: 当前 stage 从上一个 stage 接收激活
- 实际实现中，Send 和 Recv 是成对出现的

通信量：
- data_size = batch_size * seq_len * hidden_size * act_bytes
- 单向传输，不需要像 AllReduce 那样的多轮通信

时延公式：
- transfer_time = data_size / bandwidth
- total_latency = transfer_time + rtt_overhead + static_overhead

触发条件：
- PP > 1 时，每个 stage 边界需要 P2P 通信
- Embedding -> Stage 0: 只有 Recv（或内置在 Embedding 中）
- Stage i -> Stage i+1: Send + Recv
- Last Stage -> LM Head: 内置在层内

注意：
- P2P 带宽通常与 AllReduce 带宽不同，需要单独配置
- 跨节点 PP 时使用 inter_node 带宽
"""

from .layer_base import LayerBase


class LayerP2P(LayerBase):
    """P2P通信算子 (PP stage 间通信)"""

    def __init__(self, hardware_config, model_config, deploy_config, quant_config,
                 data_size_bytes, is_cross_node=False):
        """
        Args:
            data_size_bytes: 传输的数据量 (bytes)
            is_cross_node: 是否跨节点通信
        """
        super().__init__(hardware_config, model_config, deploy_config, quant_config)
        self.data_size_bytes = data_size_bytes
        self.is_cross_node = is_cross_node
        self.is_comm_op = True

    def get_cube_flops(self):
        return 0.0

    def get_vector_flops(self):
        return 0.0

    def get_mem_bytes(self):
        return 0.0

    def get_comm_bytes(self):
        """P2P 通信量"""
        return self.data_size_bytes

    def _get_bandwidth(self):
        """获取 P2P 通信带宽

        P2P 通信使用专门的 p2p_bw 带宽配置。
        如果跨节点，使用 inter_node 带宽。
        """
        if self.is_cross_node:
            bw_gbps = self.hardware_config.comm_bw_inter_node_gbps
            bw_util = self.hardware_config.comm_bw_inter_node_utilization
        else:
            # P2P 通常使用 NVLink/高速互联，使用 intra_node 带宽
            # 如果有专门的 p2p_bw 配置则使用，否则回退到 intra_node
            if hasattr(self.hardware_config, 'comm_p2p_bw_gbps'):
                bw_gbps = self.hardware_config.comm_p2p_bw_gbps
                bw_util = getattr(self.hardware_config, 'comm_p2p_bw_utilization', 0.8)
            else:
                bw_gbps = self.hardware_config.comm_bw_intra_node_gbps
                bw_util = self.hardware_config.comm_bw_intra_node_utilization

        return bw_gbps, bw_util

    def get_comm_time(self):
        """P2P 通信时延计算

        公式: transfer_time + rtt_overhead + static_overhead

        P2P 是单向传输，不需要像 AllReduce 那样的多轮通信。
        """
        bw_gbps, bw_util = self._get_bandwidth()

        # 传输时间
        transfer_time_ms = self.data_size_bytes / (bw_gbps * 1e9) * 1000 / bw_util

        # RTT 开销（单向通信只需一次）
        rtt_overhead_ms = self.hardware_config.comm_rtt_overhead_ms

        # 静态开销
        static_overhead_ms = self.hardware_config.comm_static_overhead_ms

        return transfer_time_ms + rtt_overhead_ms + static_overhead_ms


class LayerP2PSend(LayerP2P):
    """P2P Send 通信算子

    发送激活到下一个 PP stage。
    与 LayerP2P 功能相同，只是语义上更明确。
    """
    pass


class LayerP2PRecv(LayerP2P):
    """P2P Recv 通信算子

    从上一个 PP stage 接收激活。
    与 LayerP2P 功能相同，只是语义上更明确。
    """
    pass

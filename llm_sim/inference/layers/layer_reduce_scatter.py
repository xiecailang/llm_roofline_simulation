"""ReduceScatter通信算子

ReduceScatter (Ring Reduce-Scatter) 通信模式：
- 总传输量 = (N-1) / N * data_size
- 先 AllReduce 再切分结果给各 rank

与 AllGather 是逆操作：
- AllGather: [data/N] -> [data] (每个 rank 得到完整数据)
- ReduceScatter: [data] -> [data/N] (每个 rank 得到切分后的归约结果)

参考 SKILL.md 中的 Reduce-Scatter 时延公式
"""

from .layer_base import LayerBase


class LayerReduceScatter(LayerBase):
    """ReduceScatter通信算子"""

    def __init__(self, hardware_config, model_config, deploy_config, quant_config, data_size_bytes, num_devices):
        super().__init__(hardware_config, model_config, deploy_config, quant_config)
        self.data_size_bytes = data_size_bytes
        self.num_devices = num_devices
        self.is_comm_op = True

    def get_cube_flops(self):
        return 0.0

    def get_vector_flops(self):
        return 0.0

    def get_mem_bytes(self):
        return self.data_size_bytes

    def get_comm_bytes(self):
        """Ring Reduce-Scatter 总通信量: (N-1) / N * data_size"""
        return (self.num_devices - 1) / self.num_devices * self.data_size_bytes

    def _get_bandwidth(self):
        """获取通信带宽和利用率"""
        max_chips_per_node = self.hardware_config.max_chips_per_node

        if self.num_devices <= 4:
            bw_gbps = self.hardware_config.comm_bw_4gpu_gbps
            bw_util = self.hardware_config.comm_bw_4gpu_utilization
        elif self.num_devices <= 8:
            bw_gbps = self.hardware_config.comm_bw_8gpu_gbps
            bw_util = self.hardware_config.comm_bw_8gpu_utilization
        elif self.num_devices <= max_chips_per_node:
            bw_gbps = self.hardware_config.comm_bw_intra_node_gbps
            bw_util = self.hardware_config.comm_bw_intra_node_utilization
        else:
            bw_gbps = self.hardware_config.comm_bw_inter_node_gbps
            bw_util = self.hardware_config.comm_bw_inter_node_utilization

        return bw_gbps, bw_util

    def get_comm_time(self):
        """ReduceScatter时延计算

        Ring Reduce-Scatter 公式:
        reduce_scatter_latency = data_size / (bandwidth * num_ranks) + rtt_overhead

        Ring Reduce-Scatter 总传输量 = (N-1) / N * data_size
        时延 = data_size / (bandwidth * N) + rtt_overhead
        """
        bw_gbps, bw_util = self._get_bandwidth()

        # Ring Reduce-Scatter: 有效传输时间 = data_size / (bandwidth * N)
        transfer_time_ms = self.data_size_bytes / (bw_gbps * 1e9 * self.num_devices) * 1000 / bw_util

        # RTT开销
        rtt_overhead_ms = self.hardware_config.comm_rtt_overhead_ms

        # 静态开销
        static_overhead_ms = self.hardware_config.comm_static_overhead_ms

        return transfer_time_ms + rtt_overhead_ms + static_overhead_ms
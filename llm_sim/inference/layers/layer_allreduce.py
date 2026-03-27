"""AllReduce通信算子

AllReduce (Ring All-Reduce) 通信模式：
- 总传输量 = 2 * (N-1) / N * data_size
- 包含 scatter-reduce 和 all-gather 两个阶段

参考 SKILL.md 中的 All-Reduce 时延公式
"""

from .layer_base import LayerBase


class LayerAllReduce(LayerBase):
    """AllReduce通信算子"""

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
        """Ring All-Reduce 总通信量: 2 * (N-1) / N * data_size"""
        return 2 * (self.num_devices - 1) / self.num_devices * self.data_size_bytes

    def _get_bandwidth(self):
        """获取通信带宽和利用率

        根据通信设备数量选择合适的带宽：
        - 4 GPU域内: 最高带宽 (NVLink全连接)
        - 8 GPU域内: 高带宽
        - 框内: 中等带宽
        - 框间: 较低带宽
        """
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
        """AllReduce时延计算

        Ring All-Reduce 公式 (来自 SKILL.md):
        all_reduce_latency = 2 * data_size / (bandwidth * num_ranks) + rtt_overhead

        Ring All-Reduce 总传输量 = 2 * (N-1) / N * data_size
        有效带宽 = 带宽 * N / (N-1)，因此：
        时延 = 2 * data_size / (bandwidth * N) + rtt_overhead
        """
        bw_gbps, bw_util = self._get_bandwidth()

        # Ring All-Reduce: 有效传输时间 = 2 * data_size / (bandwidth * N)
        transfer_time_ms = 2 * self.data_size_bytes / (bw_gbps * 1e9 * self.num_devices) * 1000 / bw_util

        # RTT开销
        rtt_overhead_ms = self.hardware_config.comm_rtt_overhead_ms

        # 静态开销
        static_overhead_ms = self.hardware_config.comm_static_overhead_ms

        return transfer_time_ms + rtt_overhead_ms + static_overhead_ms

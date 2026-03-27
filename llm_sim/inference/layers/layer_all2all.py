"""All-to-All通信算子 - MoE专用 (DeepEP优化版)

All-to-All通信模式：
- 每个 rank 需要与其他 (num_devices-1) 个 rank 通信
- 通信量 = data_size / num_devices per pair
- RTT开销需要累积 (num_devices-1) 次（如果不使用并发传输）

DeepEP优化技术:
1. **双模式内核**:
   - high_throughput: 训练/Prefill场景，追求带宽利用率
   - low_latency: Decode场景，使用Pure RDMA最小化延迟

2. **Hook-based Compute-Communication Overlap**:
   - 零SM占用: 通信由hook接口在后台执行
   - 当 compute_time >= comm_time 时，通信延迟可被完全隐藏
   - Effective time = max(comm_time, overlapable_compute_time)

3. **NVLink + RDMA混合通信**:
   - Intranode: NVLink高带宽 (节点内)
   - Internode: RDMA (跨节点，使用NVSHMEM)

4. **FP8通信**:
   - 当使用FP8传输时，通信量减半
   - 对于跨节点通信特别有效

参考:
- DeepEP GitHub: https://github.com/deepseek-ai/DeepEP
- DeepSeek-V3 Technical Report: https://arxiv.org/pdf/2412.19437
- SKILL.md 中的 All-to-All 时延公式
"""

from .layer_base import LayerBase


class LayerAll2All(LayerBase):
    """All-to-All通信算子 (MoE dispatch/combine) - DeepEP优化版

    支持两种模式:
    - high_throughput: 训练和Prefill场景，追求高带宽利用率
    - low_latency: Decode场景，使用Pure RDMA最小化延迟
    """

    def __init__(self, hardware_config, model_config, deploy_config, quant_config,
                 data_size_bytes, num_devices, mode='high_throughput',
                 overlapable_compute_time_ms=0.0, is_cross_node=None):
        """初始化All-to-All通信算子

        Args:
            data_size_bytes: 总通信数据量 (bytes)
            num_devices: 参与通信的设备数 (通常是EP)
            mode: 'high_throughput' 或 'low_latency'
            overlapable_compute_time_ms: 可与通信重叠的计算时间 (ms)
                                         DeepEP hook机制允许通信与计算并行
            is_cross_node: 是否跨节点通信，None表示自动判断
        """
        super().__init__(hardware_config, model_config, deploy_config, quant_config)
        self.data_size_bytes = data_size_bytes
        self.num_devices = num_devices
        self.mode = mode
        self.overlapable_compute_time_ms = overlapable_compute_time_ms
        self.is_comm_op = True

        # 标记是否已在内部处理overlap（避免ModuleBase重复处理）
        self.has_internal_overlap = overlapable_compute_time_ms > 0

        # 自动判断是否跨节点
        if is_cross_node is None:
            max_chips_per_node = hardware_config.max_chips_per_node
            self.is_cross_node = num_devices > max_chips_per_node
        else:
            self.is_cross_node = is_cross_node

    def get_cube_flops(self):
        return 0.0

    def get_vector_flops(self):
        return 0.0

    def get_mem_bytes(self):
        return 0.0

    def get_comm_bytes(self):
        return self.data_size_bytes

    def _get_bandwidth(self):
        """获取通信带宽和利用率

        根据通信设备数量和DeepEP优化策略选择合适的带宽：
        - 节点内 (NVLink): 高带宽，低延迟
        - 节点间 (RDMA): 较低带宽，较高延迟

        DeepEP使用NVSHMEM实现zero-copy RDMA通信
        """
        max_chips_per_node = self.hardware_config.max_chips_per_node

        if self.is_cross_node:
            # 跨节点: RDMA通信
            bw_gbps = self.hardware_config.comm_bw_inter_node_gbps
            bw_util = self.hardware_config.comm_bw_inter_node_utilization
        elif self.num_devices <= 4:
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

    def _get_low_latency_params(self):
        """获取低延迟模式的参数

        DeepEP low_latency模式特点:
        - Pure RDMA: 直接使用RDMA绕过NCCL
        - 零SM占用: 通信由网络接口完成，不占用GPU SM
        - Hook机制: 允许通信与计算重叠

        实测数据 (H800 + CX7 400Gb/s, EP=64):
        - Dispatch latency: ~173 µs
        - Combine latency: ~314 µs
        - RDMA bandwidth: ~43 GB/s

        延迟模型:
        - 基础延迟 + 传输延迟 + 启动开销
        - 由于RDMA的zero-copy特性，传输效率更高
        """
        # 使用HardwareConfig中的DeepEP配置
        rdma_efficiency = getattr(self.hardware_config, 'comm_rdma_efficiency', 0.85)
        base_latency_us = getattr(self.hardware_config, 'deepep_base_latency_us', 50.0)

        # 每个rank对的额外延迟
        # DeepEP使用并发传输，延迟增长为 O(log(N)) 而非 O(N)
        import math
        scaling_factor = math.log2(max(self.num_devices, 2))

        return rdma_efficiency, base_latency_us, scaling_factor

    def get_comm_time(self):
        """All-to-All时延计算 (DeepEP优化版)

        DeepEP优化策略:

        1. **High-Throughput模式** (训练/Prefill):
           all_to_all_latency = data_size / bandwidth + rtt_overhead * sqrt(N-1)
           - 追求高带宽利用率
           - RTT开销累积 sqrt(N-1) 次（部分并发）

        2. **Low-Latency模式** (Decode):
           latency = base_latency + transfer_time * scaling_factor
           - Pure RDMA绕过NCCL
           - 零SM占用的Hook机制
           - 延迟增长为 O(log N)

        3. **Compute-Communication Overlap**:
           effective_time = max(comm_time, overlapable_compute_time)
           - 当 overlapable_compute_time >= comm_time 时，通信完全隐藏
           - 典型场景: Shared Expert计算与Dispatch通信重叠
        """
        # 数据量为0时直接返回
        if self.data_size_bytes <= 0:
            return 0.0

        if self.mode == 'low_latency':
            # DeepEP Low-Latency模式
            rdma_eff, base_latency_us, scaling_factor = self._get_low_latency_params()

            # 使用RDMA带宽（跨节点通信）
            rdma_bw_gbps = getattr(self.hardware_config, 'comm_rdma_bw_gbps', 50.0)
            effective_bw = rdma_bw_gbps * 1e9 * rdma_eff

            # 传输时间 (ms)
            transfer_time_ms = self.data_size_bytes / effective_bw * 1000

            # 总延迟 = 基础延迟 + 传输延迟 * 缩放因子
            base_latency_ms = base_latency_us / 1000.0
            comm_time = base_latency_ms + transfer_time_ms * scaling_factor
        else:
            # High-Throughput模式
            bw_gbps, bw_util = self._get_bandwidth()
            transfer_time_ms = self.data_size_bytes / (bw_gbps * 1e9) * 1000 / bw_util

            # RTT开销: 部分并发传输
            import math
            rtt_accumulated = self.hardware_config.comm_rtt_overhead_ms * math.sqrt(self.num_devices - 1)
            static_overhead_ms = self.hardware_config.comm_static_overhead_ms

            comm_time = transfer_time_ms + rtt_accumulated + static_overhead_ms

        # DeepEP Compute-Communication Overlap
        # 当 overlapable_compute_time >= comm_time 时，通信延迟可被隐藏
        if self.overlapable_compute_time_ms > 0:
            effective_time = max(comm_time, self.overlapable_compute_time_ms)
        else:
            effective_time = comm_time

        return effective_time

    def get_overlap_efficiency(self):
        """计算通信-计算重叠效率

        Returns:
            float: 重叠效率 (0.0 - 1.0)
                   1.0 表示通信完全被隐藏
                   0.0 表示无重叠
        """
        if self.overlapable_compute_time_ms <= 0 or self.data_size_bytes <= 0:
            return 0.0

        comm_time = self.get_comm_time()
        # 计算被隐藏的通信时间比例
        if comm_time <= 0:
            return 1.0

        hidden_ratio = min(self.overlapable_compute_time_ms / comm_time, 1.0)
        return hidden_ratio

    def get_profiling(self):
        """获取算子性能分析数据 (DeepEP扩展版)"""
        return {
            'op_name': self.__class__.__name__,
            'cube_flops': self.get_cube_flops(),
            'cube_time_ms': self.get_cube_time(),
            'vector_flops': self.get_vector_flops(),
            'vector_time_ms': self.get_vector_time(),
            'mem_bytes': self.get_mem_bytes(),
            'mem_time_ms': self.get_mem_time(),
            'comm_bytes': self.get_comm_bytes(),
            'comm_time_ms': self.get_comm_time(),
            'total_time_ms': self.get_cost_time(),
            # DeepEP扩展信息
            'deepep_mode': self.mode,
            'deepep_is_cross_node': self.is_cross_node,
            'deepep_overlapable_compute_ms': self.overlapable_compute_time_ms,
            'deepep_overlap_efficiency': self.get_overlap_efficiency(),
        }

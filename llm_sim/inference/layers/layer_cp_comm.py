"""CP通信算子 - Context Parallel Ring Attention 通信

Context Parallelism (CP) 使用 Ring Attention 模式：
- 序列按 token 切分到 CP ranks
- 每个 CP rank 处理 seq_len / CP 个 token
- Ring Attention 通过 (CP-1) 轮 KV cache 交换实现全局注意力

通信模式 (Ring Attention):
  Round 0: 本地 Q @ 本地 K (无需通信)
  Round 1: 本地 Q @ 远端 K_1 (P2P send/recv)
  Round 2: 本地 Q @ 远端 K_2 (P2P send/recv)
  ...
  Round CP-1: 本地 Q @ 远端 K_{CP-1} (P2P send/recv)

每轮通信量:
  - MLA 压缩后: batch × seq_per_cp × kv_lora_rank × dtype_bytes
  - 未压缩: batch × seq_per_cp × num_heads × qk_head_dim × dtype_bytes

时延公式:
  cp_comm_time = (CP-1) × per_round_latency
  per_round_latency = kv_bytes / bandwidth + rtt_overhead

带宽选择:
  - CP 通信通常在节点内 (NVLink)
  - 当 CP > max_chips_per_node 时使用跨节点带宽
"""

from .layer_base import LayerBase


class LayerCPComm(LayerBase):
    """Context Parallel Ring Attention 通信

    Ring Attention 中，每个 CP rank 需要与 (CP-1) 个其他 rank 交换 KV cache。
    使用 Ring 模式，每轮与一个 neighbor 交换 KV block。

    MLA 优化:
    - KV cache 存储压缩后的 latent (kv_lora_rank 维度)
    - 通信量 = batch × seq_per_cp × kv_lora_rank × dtype (而非完整 head 维度)
    """

    def __init__(self, hardware_config, model_config, deploy_config, quant_config,
                 batch_size, seq_per_cp, num_cp, kv_lora_rank=512):
        """初始化 CP 通信算子

        Args:
            batch_size: 批大小
            seq_per_cp: 每个 CP rank 处理的序列长度
            num_cp: Context Parallelism 的并行度
            kv_lora_rank: MLA KV cache 压缩后的维度 (默认512)
        """
        super().__init__(hardware_config, model_config, deploy_config, quant_config)
        self.batch_size = batch_size
        self.seq_per_cp = seq_per_cp
        self.num_cp = num_cp
        self.kv_lora_rank = kv_lora_rank
        self.is_comm_op = True

        # KV cache 使用 cache_write_bits 量化
        self.cache_bytes = quant_config.default_cache_write_bits / 8

    def get_cube_flops(self):
        return 0.0

    def get_vector_flops(self):
        return 0.0

    def get_mem_bytes(self):
        return 0.0

    def get_comm_bytes(self):
        """Ring Attention 总通信量

        (CP-1) 轮 KV cache 交换，每轮:
        - Send: batch × seq_per_cp × kv_lora_rank × dtype
        - Recv: batch × seq_per_cp × kv_lora_rank × dtype

        总通信量 = 2 × (CP-1) × per_round_bytes
        """
        per_round_bytes = (
            self.batch_size * self.seq_per_cp * self.kv_lora_rank * self.cache_bytes
        )
        return 2 * (self.num_cp - 1) * per_round_bytes

    def _get_bandwidth(self):
        """获取 CP 通信带宽

        CP 通常在节点内通信 (NVLink)，使用 intra-node 带宽。
        当 CP > max_chips_per_node 时使用跨节点带宽。
        """
        max_chips = self.hardware_config.max_chips_per_node

        if self.num_cp <= 4:
            bw_gbps = self.hardware_config.comm_bw_4gpu_gbps
            bw_util = self.hardware_config.comm_bw_4gpu_utilization
        elif self.num_cp <= 8:
            bw_gbps = self.hardware_config.comm_bw_8gpu_gbps
            bw_util = self.hardware_config.comm_bw_8gpu_utilization
        elif self.num_cp <= max_chips:
            bw_gbps = self.hardware_config.comm_bw_intra_node_gbps
            bw_util = self.hardware_config.comm_bw_intra_node_utilization
        else:
            bw_gbps = self.hardware_config.comm_bw_inter_node_gbps
            bw_util = self.hardware_config.comm_bw_inter_node_utilization

        return bw_gbps, bw_util

    def get_comm_time(self):
        """CP Ring Attention 通信时延

        时延 = (CP-1) × per_round_latency
        per_round_latency = kv_bytes / bandwidth + rtt_overhead + static_overhead

        注意: Ring Attention 的通信-计算可以重叠
        - 当一轮通信完成时，下一轮的 attention 计算已经开始
        - 简化建模: 不考虑 overlap (保守估计)
        """
        if self.num_cp <= 1:
            return 0.0

        per_round_bytes = (
            self.batch_size * self.seq_per_cp * self.kv_lora_rank * self.cache_bytes
        )

        bw_gbps, bw_util = self._get_bandwidth()

        # 每轮传输时间 (send + recv 串行)
        # Ring 模式下 send 和 recv 可以流水线化，简化为单次传输
        transfer_time_ms = per_round_bytes / (bw_gbps * 1e9) * 1000 / bw_util

        # RTT 开销
        rtt_overhead_ms = self.hardware_config.comm_rtt_overhead_ms

        # 静态开销 (每轮)
        static_overhead_ms = self.hardware_config.comm_static_overhead_ms

        # (CP-1) 轮通信
        per_round_latency = transfer_time_ms + rtt_overhead_ms + static_overhead_ms
        return (self.num_cp - 1) * per_round_latency

    def get_profiling(self):
        """获取 CP 通信性能分析数据"""
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
            # CP 扩展信息
            'cp_num_ranks': self.num_cp,
            'cp_seq_per_rank': self.seq_per_cp,
            'cp_kv_lora_rank': self.kv_lora_rank,
            'cp_rounds': self.num_cp - 1,
        }

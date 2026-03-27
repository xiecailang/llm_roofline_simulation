"""Embedding算子 - 查表操作

并行策略影响：
- Vocab Parallel (lm_head_tp): Embedding表按vocab维度切分
  每个 TP rank 存储 vocab_size / lm_head_tp 的 embedding
  **重要**: VocabParallelEmbedding内部使用AllReduce，输出是replicated状态
- 无 TP 切分: 每个 rank 存储完整 embedding 表

vLLM的VocabParallelEmbedding forward流程:
1. 根据input tokens获取对应embedding (每个rank只查自己那部分vocab)
2. 对不在本rank vocab范围内的token，将embedding置零
3. 调用 tensor_model_parallel_all_reduce() 汇总所有rank的结果
4. 输出是 **replicated** 状态（每个rank都有完整的hidden states）

因此，Embedding之后的hidden states是replicated的，下游的ColumnParallelLinear
（如QKV投影）可以直接使用，不需要额外的AllGather。
"""

from .layer_base import LayerBase


class LayerEmbedding(LayerBase):
    """Embedding查表算子"""

    def __init__(self, hardware_config, model_config, deploy_config, quant_config, seq_len):
        super().__init__(hardware_config, model_config, deploy_config, quant_config)
        self.seq_len = seq_len
        self.vocab_size = model_config.vocab_size
        # Vocab Parallel: lm_head_tp > 1 时 embedding 表按 vocab 切分
        self.lm_head_tp = getattr(deploy_config, 'lm_head_tp', 1)
        self.vocab_size_per_tp = self.vocab_size // self.lm_head_tp

        # AllReduce通信数据量 (当使用VocabParallel时)
        if self.lm_head_tp > 1:
            # 每个rank输出 batch * seq * hidden，AllReduce汇总
            self.allreduce_data_size = self.batch_size * self.seq_len * self.hidden_size * self.act_transfer_bytes
        else:
            self.allreduce_data_size = 0

    def get_cube_flops(self):
        """Embedding是查表操作，无CUBE计算"""
        return 0.0

    def get_vector_flops(self):
        """Embedding是查表操作，无Vector计算"""
        return 0.0

    def get_mem_bytes(self):
        """读取embedding table和写入输出

        Vocab Parallel: 每个 TP rank 只存储 vocab_size/lm_head_tp 的 embedding
        """
        # 读取: batch * seq_len 个embedding向量
        # Vocab Parallel: 每个 rank 只查自己那部分 vocab，输入需要经过 ReduceScatter
        read_bytes = self.batch_size * self.seq_len * self.hidden_size * self.act_transfer_bytes
        # 写入: [batch, seq, hidden]
        write_bytes = self.batch_size * self.seq_len * self.hidden_size * self.act_transfer_bytes

        return read_bytes + write_bytes

    def get_comm_bytes(self):
        """VocabParallelEmbedding内部的AllReduce通信量

        Ring All-Reduce 总通信量: 2 * (N-1) / N * data_size
        """
        if self.lm_head_tp <= 1:
            return 0.0
        return 2 * (self.lm_head_tp - 1) / self.lm_head_tp * self.allreduce_data_size

    def get_comm_time(self):
        """AllReduce通信时延

        VocabParallelEmbedding在forward最后调用tensor_model_parallel_all_reduce
        """
        if self.lm_head_tp <= 1:
            return 0.0

        # 根据通信设备数量选择带宽
        num_devices = self.lm_head_tp
        max_chips_per_node = self.hardware_config.max_chips_per_node

        if num_devices <= 4:
            bw_gbps = self.hardware_config.comm_bw_4gpu_gbps
            bw_util = self.hardware_config.comm_bw_4gpu_utilization
        elif num_devices <= 8:
            bw_gbps = self.hardware_config.comm_bw_8gpu_gbps
            bw_util = self.hardware_config.comm_bw_8gpu_utilization
        elif num_devices <= max_chips_per_node:
            bw_gbps = self.hardware_config.comm_bw_intra_node_gbps
            bw_util = self.hardware_config.comm_bw_intra_node_utilization
        else:
            bw_gbps = self.hardware_config.comm_bw_inter_node_gbps
            bw_util = self.hardware_config.comm_bw_inter_node_utilization

        # Ring All-Reduce: 有效传输时间 = 2 * data_size / (bandwidth * N)
        transfer_time_ms = 2 * self.allreduce_data_size / (bw_gbps * 1e9 * num_devices) * 1000 / bw_util
        rtt_overhead_ms = self.hardware_config.comm_rtt_overhead_ms
        static_overhead_ms = self.hardware_config.comm_static_overhead_ms

        return transfer_time_ms + rtt_overhead_ms + static_overhead_ms

    def get_cost_time(self):
        """Embedding总耗时 = 查表时间 + AllReduce通信时间"""
        mem_time = self.get_mem_time()
        comm_time = self.get_comm_time()
        return self.hardware_config.op_overhead_ms + mem_time + comm_time

"""Layer基类 - 算子层基类

并行策略对计算量的影响：
- TP (Tensor Parallel):
  * ColumnParallel: 权重按列切分，每个TP节点计算量 = 2*m*k*(n/TP)，访存量同比减少
  * RowParallel: 权重按行切分，每个TP节点计算量 = 2*m*(k/TP)*n，访存量同比减少
- EP (Expert Parallel):
  * 每个EP节点只存储并计算 n_experts/EP 个专家
  * All-to-All通信量 = batch * seq * hidden（每个token路由到对应EP节点）
- PP (Pipeline Parallel):
  * 每个PP节点只计算 num_layers/PP 层
  * 引入pipeline bubble开销
- CP (Context Parallel):
  * 每个CP节点处理 seq_len/CP 的序列
  * 影响attention计算量和KV cache访存量
"""


class LayerBase:
    """算子基类"""

    def __init__(self, hardware_config, model_config, deploy_config, quant_config):
        self.hardware_config = hardware_config
        self.model_config = model_config
        self.deploy_config = deploy_config
        self.quant_config = quant_config

        # 提取常用配置
        self.hidden_size = model_config.hidden_size
        self.batch_size = deploy_config.micro_batch_size

        # 并行策略
        self.tp = deploy_config.attention_tp
        self.ep = deploy_config.expert_parallel
        self.pp = deploy_config.pipeline_parallel
        self.cp = deploy_config.context_parallel
        self.moe_tp = deploy_config.moe_tp

        # 量化配置
        self.weight_bits = quant_config.default_weight_bits
        self.act_compute_bits = quant_config.default_activation_compute_bits
        self.act_transfer_bits = quant_config.default_activation_transfer_bits
        self.cache_read_bits = quant_config.default_cache_read_bits
        self.cache_write_bits = quant_config.default_cache_write_bits

        # 字节数
        self.weight_bytes = self.weight_bits / 8
        self.act_compute_bytes = self.act_compute_bits / 8
        self.act_transfer_bytes = self.act_transfer_bits / 8
        self.cache_read_bytes = self.cache_read_bits / 8
        self.cache_write_bytes = self.cache_write_bits / 8

        # 标记是否为通信算子
        self.is_comm_op = False
        # 标记是否已在内部处理overlap（避免ModuleBase重复处理）
        self.has_internal_overlap = False

    def get_cube_flops(self):
        """计算CUBE计算量 (FLOPs) - 子类实现"""
        return 0.0

    def get_vector_flops(self):
        """计算Vector计算量 (FLOPs) - 子类实现"""
        return 0.0

    def get_mem_bytes(self):
        """计算访存量 (Bytes) - 子类实现"""
        return 0.0

    def get_comm_bytes(self):
        """计算通信量 (Bytes) - 子类实现"""
        return 0.0

    def get_cube_time(self):
        """计算CUBE时延 (ms)"""
        flops = self.get_cube_flops()
        if flops == 0:
            return 0.0
        # 根据activation计算精度选择算力
        bits = self.act_compute_bits
        if bits <= 4:
            tflops = self.hardware_config.cube_tflops_fp4
        elif bits <= 8:
            tflops = self.hardware_config.cube_tflops_fp8
        elif bits <= 16:
            tflops = self.hardware_config.cube_tflops_fp16
        else:
            tflops = self.hardware_config.cube_tflops_fp32
        return flops / (tflops * 1e12) * 1000 / self.hardware_config.cube_utilization

    def get_vector_time(self):
        """计算Vector时延 (ms)

        **重要**: Vector单元统一使用FP16算力，与activation计算精度无关。
        原因：
        1. 现代AI加速器的Vector单元主要优化FP16/BF16操作
        2. 激活函数(SiLU, Softmax, LayerNorm等)在FP16下执行足够精确
        3. FP32 vector操作会显著降低性能，实际实现中很少使用
        """
        flops = self.get_vector_flops()
        if flops == 0:
            return 0.0
        # Vector单元统一使用FP16算力
        tflops = self.hardware_config.vector_tflops_fp16
        return flops / (tflops * 1e12) * 1000 / self.hardware_config.vector_utilization

    def get_mem_time(self):
        """计算访存时延 (ms)"""
        mem_bytes = self.get_mem_bytes()
        if mem_bytes == 0:
            return 0.0
        bw_gbps = self.hardware_config.hbm_read_bw_gbps
        bw_util = self.hardware_config.hbm_read_bw_utilization
        return mem_bytes / (bw_gbps * 1e9) * 1000 / bw_util

    def get_comm_time(self):
        """计算通信时延 (ms) - 子类实现"""
        return 0.0

    def get_cost_time(self):
        """计算总耗时 (ms)

        通信算子：直接返回通信时延
        计算算子：max(cube + vector, mem) + 算子头开销
        """
        if self.is_comm_op:
            return self.get_comm_time()
        compute_time = self.get_cube_time() + self.get_vector_time()
        mem_time = self.get_mem_time()
        return self.hardware_config.op_overhead_ms + max(compute_time, mem_time)

    def get_profiling(self):
        """获取算子性能分析数据"""
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
        }

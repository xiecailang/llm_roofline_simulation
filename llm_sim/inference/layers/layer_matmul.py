"""MatMul算子 - GEMM矩阵乘法

C = A @ B
A: [m, k], B: [k, n], C: [m, n]

TP影响：需要在外部指定是ColumnParallel还是RowParallel
- ColumnParallel: n按TP切分，每个节点计算 n/TP
- RowParallel: k按TP切分，每个节点计算 k/TP
"""

from .layer_base import LayerBase


class LayerMatMul(LayerBase):
    """矩阵乘法算子"""

    def __init__(self, hardware_config, model_config, deploy_config, quant_config, m, k, n,
                 is_column_parallel=True):
        """
        Args:
            is_column_parallel: True表示ColumnParallel(n切分), False表示RowParallel(k切分)
        """
        super().__init__(hardware_config, model_config, deploy_config, quant_config)
        self.m = m
        self.k = k
        self.n = n
        self.is_column_parallel = is_column_parallel

    def get_cube_flops(self):
        """GEMM计算量: 2 * M * K * N

        TP影响：
        - ColumnParallel: 2 * M * K * (N/TP)
        - RowParallel: 2 * M * (K/TP) * N
        """
        if self.is_column_parallel:
            # ColumnParallel: n按TP切分
            return 2.0 * self.m * self.k * (self.n / self.tp)
        else:
            # RowParallel: k按TP切分
            return 2.0 * self.m * (self.k / self.tp) * self.n

    def get_vector_flops(self):
        """GEMM主要使用CUBE"""
        return 0.0

    def get_mem_bytes(self):
        """访存量: 读A、B，写C

        TP影响：
        - ColumnParallel: A完整读取, B按列切分, C按列切分
        - RowParallel: A按列切分, B按行切分, C完整写入
        """
        if self.is_column_parallel:
            # ColumnParallel: A: [m, k], B: [k, n/TP], C: [m, n/TP]
            read_a = self.m * self.k * self.act_transfer_bytes
            read_b = self.k * (self.n / self.tp) * self.weight_bytes
            write_c = self.m * (self.n / self.tp) * self.act_transfer_bytes
        else:
            # RowParallel: A: [m, k/TP], B: [k/TP, n], C: [m, n]
            read_a = self.m * (self.k / self.tp) * self.act_transfer_bytes
            read_b = (self.k / self.tp) * self.n * self.weight_bytes
            write_c = self.m * self.n * self.act_transfer_bytes

        return read_a + read_b + write_c

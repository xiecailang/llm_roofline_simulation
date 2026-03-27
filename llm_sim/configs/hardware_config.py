"""硬件配置数据类"""

from dataclasses import dataclass
from typing import Dict
import json


@dataclass
class HardwareConfig:
    """硬件配置"""

    # CUBE算力 (TFLOPS)
    cube_tflops_fp4: float
    cube_tflops_fp8: float
    cube_tflops_fp16: float
    cube_tflops_fp32: float
    cube_utilization: float  # 算力利用率

    # Vector算力 (TFLOPS)
    vector_tflops_fp16: float
    vector_tflops_fp32: float
    vector_utilization: float  # 算力利用率

    # 算子头开销 (ms)
    op_overhead_ms: float

    # HBM (G1)
    hbm_size_gb: float
    hbm_utilization: float
    hbm_read_bw_gbps: float
    hbm_read_bw_utilization: float
    hbm_write_bw_gbps: float
    hbm_write_bw_utilization: float

    # Host Memory (G2)
    host_mem_size_gb: float
    host_mem_utilization: float
    host_to_device_bw_gbps: float
    host_to_device_bw_utilization: float
    device_to_host_bw_gbps: float
    device_to_host_bw_utilization: float

    # SSD in rack (G3)
    ssd_rack_size_gb: float
    ssd_rack_read_bw_gbps: float
    ssd_rack_read_bw_utilization: float
    ssd_rack_write_bw_gbps: float
    ssd_rack_write_bw_utilization: float

    # SSD outside (G4)
    ssd_outside_size_gb: float
    ssd_outside_read_bw_gbps: float
    ssd_outside_read_bw_utilization: float
    ssd_outside_write_bw_gbps: float
    ssd_outside_write_bw_utilization: float

    # 通信带宽 (GB/s, 单向)
    comm_bw_4gpu_gbps: float
    comm_bw_4gpu_utilization: float
    comm_bw_8gpu_gbps: float
    comm_bw_8gpu_utilization: float
    comm_bw_intra_node_gbps: float
    comm_bw_intra_node_utilization: float
    comm_bw_inter_node_gbps: float
    comm_bw_inter_node_utilization: float
    comm_p2p_bw_gbps: float
    comm_rtt_overhead_ms: float
    comm_static_overhead_ms: float

    # 芯片配置
    chips_per_card: int
    max_chips_per_node: int
    total_chips: int
    chip_name: str

    # DeepEP通信优化配置
    # 参考: https://github.com/deepseek-ai/DeepEP
    # RDMA带宽 (跨节点All-to-All通信)
    comm_rdma_bw_gbps: float = 50.0  # CX7 400Gb/s = 50 GB/s
    comm_rdma_efficiency: float = 0.85  # DeepEP实测约85%效率
    # DeepEP low-latency模式基础延迟 (us)
    deepep_base_latency_us: float = 50.0  # RDMA基础延迟
    # DeepEP compute-communication overlap效率
    # 当compute_time >= comm_time时，overlap效率=1.0（完全隐藏）
    # 当compute_time < comm_time时，overlap效率=compute_time/comm_time
    deepep_overlap_efficiency: float = 0.9  # 典型场景overlap效率

    @classmethod
    def from_json(cls, json_path: str) -> "HardwareConfig":
        """从JSON文件加载配置"""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, json_path: str):
        """保存配置到JSON文件"""
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2, ensure_ascii=False)

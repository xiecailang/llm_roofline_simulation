"""结果输出模块"""

import csv
import json
from pathlib import Path
from typing import List, Dict
from ..ops.op_base import OpMetrics


class Reporter:
    """性能结果输出"""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.op_metrics: List[Dict] = []

    def add_op_metric(self, op_name: str, metrics: OpMetrics):
        """添加算子性能指标"""
        self.op_metrics.append({
            "op_name": op_name,
            "cube_flops": metrics.cube_flops,
            "cube_latency_ms": metrics.cube_latency_ms,
            "vector_flops": metrics.vector_flops,
            "vector_latency_ms": metrics.vector_latency_ms,
            "memory_bytes": metrics.memory_bytes,
            "memory_latency_ms": metrics.memory_latency_ms,
            "comm_bytes": metrics.comm_bytes,
            "comm_latency_ms": metrics.comm_latency_ms,
            "total_latency_ms": metrics.total_latency_ms,
        })

    def save_op_details_csv(self, filename: str = "op_details.csv"):
        """保存算子级详细信息到CSV"""
        csv_path = self.output_dir / filename
        if not self.op_metrics:
            return

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.op_metrics[0].keys())
            writer.writeheader()
            writer.writerows(self.op_metrics)

        print(f"算子详细信息已保存到: {csv_path}")

    def save_single_card_perf(
        self,
        total_latency_ms: float,
        memory_usage_gb: float,
        ttft_ms: float,
        tpot_ms: float,
        filename: str = "single_card_perf.json",
    ):
        """保存单卡性能指标到JSON"""
        # 计算各维度占比
        total_cube_lat = sum(m["cube_latency_ms"] for m in self.op_metrics)
        total_vector_lat = sum(m["vector_latency_ms"] for m in self.op_metrics)
        total_mem_lat = sum(m["memory_latency_ms"] for m in self.op_metrics)
        total_comm_lat = sum(m["comm_latency_ms"] for m in self.op_metrics)

        perf_data = {
            "total_latency_ms": total_latency_ms,
            "memory_usage_gb": memory_usage_gb,
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "qps": 1000.0 / total_latency_ms if total_latency_ms > 0 else 0,
            "tps": 1000.0 / tpot_ms if tpot_ms > 0 else 0,
            "latency_breakdown": {
                "cube_latency_ms": total_cube_lat,
                "vector_latency_ms": total_vector_lat,
                "memory_latency_ms": total_mem_lat,
                "comm_latency_ms": total_comm_lat,
            },
            "latency_ratio": {
                "cube_ratio": total_cube_lat / total_latency_ms if total_latency_ms > 0 else 0,
                "vector_ratio": total_vector_lat / total_latency_ms if total_latency_ms > 0 else 0,
                "memory_ratio": total_mem_lat / total_latency_ms if total_latency_ms > 0 else 0,
                "comm_ratio": total_comm_lat / total_latency_ms if total_latency_ms > 0 else 0,
            },
        }

        json_path = self.output_dir / filename
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(perf_data, f, indent=2, ensure_ascii=False)

        print(f"单卡性能指标已保存到: {json_path}")

    def save_system_perf(
        self,
        system_throughput: float,
        system_ttft_ms: float,
        system_tpot_ms: float,
        max_concurrency: int,
        filename: str = "system_perf.json",
    ):
        """保存系统级性能指标到JSON"""
        system_data = {
            "system_throughput": system_throughput,
            "system_ttft_ms": system_ttft_ms,
            "system_tpot_ms": system_tpot_ms,
            "max_concurrency": max_concurrency,
        }

        json_path = self.output_dir / filename
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(system_data, f, indent=2, ensure_ascii=False)

        print(f"系统级性能指标已保存到: {json_path}")

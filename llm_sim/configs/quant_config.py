"""量化配置数据类"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import json


@dataclass
class QuantConfig:
    """量化配置，支持per-op量化精度设置"""

    # 默认量化精度 (bits)
    default_weight_bits: int = 8
    default_activation_compute_bits: int = 8
    default_activation_transfer_bits: int = 8
    default_cache_read_bits: int = 8
    default_cache_write_bits: int = 8

    # per-op量化配置覆盖 (op_name -> precision_dict)
    op_overrides: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def get_op_quant(self, op_name: str) -> Dict[str, int]:
        """获取指定算子的量化配置"""
        if op_name in self.op_overrides:
            return self.op_overrides[op_name]
        return {
            "weight_bits": self.default_weight_bits,
            "activation_compute_bits": self.default_activation_compute_bits,
            "activation_transfer_bits": self.default_activation_transfer_bits,
            "cache_read_bits": self.default_cache_read_bits,
            "cache_write_bits": self.default_cache_write_bits,
        }

    @classmethod
    def from_json(cls, json_path: str) -> "QuantConfig":
        """从JSON文件加载配置"""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, json_path: str):
        """保存配置到JSON文件"""
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2, ensure_ascii=False)

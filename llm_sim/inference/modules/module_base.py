"""Module基类 - 模块层基类

每个具体模块继承此类，包含多个Layer（算子）
模块负责聚合算子，处理计算-通信掩盖

关键优化:
1. **DeepEP Compute-Communication Overlap**:
   - 累积计算时间，遇到通信时尝试重叠
   - effective_time = max(comm_time, accumulated_compute * efficiency)
   - 避免shared expert被double-counting

2. **EP负载不均衡**:
   - MoE专家计算时间乘以 ep_load_imbalance_factor
   - 不影响通信时间（通信是同步点）
"""

from collections import OrderedDict


class ModuleBase:
    """模块基类"""

    def __init__(self, hardware_config, model_config, deploy_config, quant_config):
        # 保存配置
        self.hardware_config = hardware_config
        self.model_config = model_config
        self.deploy_config = deploy_config
        self.quant_config = quant_config

        # 使用OrderedDict保证层顺序
        self.layers = OrderedDict()

    def add_layer(self, name, layer):
        """添加算子"""
        self.layers[name] = layer

    def get_cube_time(self):
        """计算模块总CUBE时延"""
        return sum(layer.get_cube_time() for layer in self.layers.values())

    def get_vector_time(self):
        """计算模块总Vector时延"""
        return sum(layer.get_vector_time() for layer in self.layers.values())

    def get_mem_time(self):
        """计算模块总访存时延"""
        return sum(layer.get_mem_time() for layer in self.layers.values())

    def get_comm_time(self):
        """计算模块总通信时延"""
        return sum(layer.get_comm_time() for layer in self.layers.values())

    def get_cost_time(self):
        """计算模块总耗时，处理计算-通信掩盖和EP负载不均衡

        Compute-Communication Overlap逻辑:
        1. 累积连续计算算子时间到 accumulated_compute
        2. 遇到通信算子时：
           - 从total中减去accumulated_compute（避免double-counting）
           - 有效时间 = max(comm_time, accumulated_compute * efficiency)
           - 这样shared expert可以与dispatch完全重叠
        3. 如果overlap未启用，则串行累加

        EP负载不均衡:
        - 专家计算时间乘以 ep_load_imbalance_factor
        - 不影响通信时间
        """
        overlap_enabled = getattr(self.deploy_config, 'enable_compute_comm_overlap', False)
        overlap_efficiency = getattr(self.deploy_config, 'overlap_efficiency', 0.9)
        ep_imbalance = getattr(self.deploy_config, 'ep_load_imbalance_factor', 1.0)

        total_time = 0.0
        accumulated_compute = 0.0

        for name, layer in self.layers.items():
            layer_time = layer.get_cost_time()

            if layer.is_comm_op:
                # 通信算子: 尝试与之前的计算重叠
                if overlap_enabled and accumulated_compute > 0:
                    # 减去已添加的accumulated_compute，避免double-counting
                    total_time -= accumulated_compute
                    # 有效时间 = max(comm, compute * efficiency)
                    effective_time = max(
                        layer_time,
                        accumulated_compute * overlap_efficiency
                    )
                    total_time += effective_time
                else:
                    total_time += layer_time
                # 通信后重置累积
                accumulated_compute = 0.0
            else:
                # 计算算子: 应用EP负载不均衡（仅专家层）
                if self._is_expert_compute_layer(name):
                    layer_time *= ep_imbalance
                total_time += layer_time
                accumulated_compute += layer_time

        # 处理剩余的计算时间（如果有未遇到通信的尾部计算）
        # accumulated_compute 已在循环中添加，这里不需要再加
        return total_time

    def _is_expert_compute_layer(self, layer_name: str) -> bool:
        """判断是否为专家计算算子（受EP负载不均衡影响）

        专家计算算子:
        - share_up, share_gate, share_down (Shared Expert)
        - moe_up, moe_gate, moe_down (Routed Expert)

        注意: e_topk_weight (Gate Routing) 不受影响，因为它不是专家计算
        """
        expert_keywords = ['share_up', 'share_gate', 'share_down',
                          'moe_up', 'moe_gate', 'moe_down']
        return any(kw in layer_name for kw in expert_keywords)

    def get_profiling(self):
        """获取模块性能分析数据"""
        return {
            'module_name': self.__class__.__name__,
            'cube_time_ms': self.get_cube_time(),
            'vector_time_ms': self.get_vector_time(),
            'mem_time_ms': self.get_mem_time(),
            'comm_time_ms': self.get_comm_time(),
            'total_time_ms': self.get_cost_time(),
            'layers': {name: layer.get_profiling() for name, layer in self.layers.items()},
        }

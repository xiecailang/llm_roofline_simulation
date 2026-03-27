"""MoE Gate算子 - 专家路由

hidden -> n_routed_experts (计算路由分数)

并行策略影响：
- EP: Gate在每个EP节点独立执行，输入是完整batch
- MoE TP: 如果使用MoE TP（而非EP），Gate权重可以按expert维度切分
- 通常Gate很小，计算量不大，一般不切分（ReplicatedLinear）

注意：Gate计算后需要执行All-to-All dispatch将token发送到对应专家

FLOPs公式:
  cube_flops = (micro_bs/attn_tp) * seq * hidden * num_experts * 2

公式解析:
  - micro_bs/attn_tp: 每个TP rank处理的有效batch size
  - seq: 序列长度
  - hidden: 隐藏层维度
  - num_experts: 路由专家总数
  - 2: FLOPs系数 (每次乘加算2 FLOPs)

关键点:
  - Gate是replicated linear，每个TP rank独立计算
  - 有效batch = micro_batch_size / attention_tp (与MoE层一致)
"""

from .layer_base import LayerBase


class LayerMoEGate(LayerBase):
    """MoE Gate: hidden -> n_routed_experts"""

    def __init__(self, hardware_config, model_config, deploy_config, quant_config, seq_len):
        super().__init__(hardware_config, model_config, deploy_config, quant_config)
        self.seq_len = seq_len
        self.n_routed_experts = model_config.num_experts

        # MoE TP影响：如果使用MoE TP，Gate输出可以按expert切分
        # 但通常Gate不切分，每个rank计算完整路由分数
        # 这里保持保守估计，不切分
        self.gate_tp = 1  # Gate通常不切分

        # 计算有效batch size: 与MoE层一致
        # gate_batch_size = micro_batch_size / attention_tp
        self.gate_batch_size = self.batch_size / self.tp

    def get_cube_flops(self):
        """计算量: 2 * (micro_bs/attn_tp) * seq * hidden * n_routed_experts

        Gate是一个小矩阵乘法，通常不切分
        有效batch = micro_batch_size / attention_tp
        """
        return 2.0 * self.gate_batch_size * self.seq_len * self.hidden_size * self.n_routed_experts

    def get_vector_flops(self):
        """Softmax计算"""
        return 3 * self.gate_batch_size * self.seq_len * self.n_routed_experts

    def get_mem_bytes(self):
        """访存量"""
        # Input: [gate_batch_size, seq, hidden]
        read_input = self.gate_batch_size * self.seq_len * self.hidden_size * self.act_transfer_bytes
        # Weight: [hidden, n_routed_experts]
        read_weight = self.hidden_size * self.n_routed_experts * self.weight_bytes
        # Output: [gate_batch_size, seq, n_routed_experts]
        write_output = self.gate_batch_size * self.seq_len * self.n_routed_experts * self.act_transfer_bytes

        return read_input + read_weight + write_output

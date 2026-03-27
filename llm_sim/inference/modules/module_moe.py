"""MoE模块 - DeepSeek-V3的Mixture of Experts (DeepEP优化版)

完整算子序列（参考 vLLM DeepSeekV2MoE 实现）：
1. e_topk_weight     - Gate routing (hidden -> n_experts, sigmoid + topk)
2. allgather_moe_tp  - AllGather (当 attention_tp < moe_tp 时恢复完整激活)
3. share_up          - Shared expert up projection (ColumnParallel)
4. share_gate_proj   - Shared expert gate projection + SiLU (ColumnParallel)
5. share_down        - Shared expert down projection (RowParallel)
6. dispatch          - All-to-All dispatch (EP通信) [与3-5可重叠]
7. moe_up            - Routed expert up projection (ColumnParallel)
8. moe_gate_proj     - Routed expert gate projection + SiLU (ColumnParallel)
9. moe_down          - Routed expert down projection (RowParallel)
10. combine          - All-to-All combine (EP通信)
11. reduce_scatter_moe_tp - ReduceScatter (当 attention_tp > moe_tp 时切分激活)
12. allgather_restore  - AllGather (当 attention_tp < moe_tp 时恢复到attention_tp级别)

关键原则:
  - EP只影响通信(All-to-All)和权重存储，不影响FLOPs
  - 每个token仍然需要经过top_k个专家的计算
  - EP只是把专家分布到不同节点，不减少总计算量
  - moe_batch_size = micro_batch_size / attention_tp * moe_tp

DeepEP优化技术:
  1. **Dispatch + Shared Expert 重叠**:
     - DeepEP的hook机制允许dispatch通信在后台执行
     - Shared expert计算可与dispatch通信并行
     - Effective time = max(dispatch_time, shared_expert_time)

  2. **双模式内核**:
     - high_throughput: Prefill场景，追求带宽利用率
     - low_latency: Decode场景，使用Pure RDMA最小化延迟

  3. **零SM占用的通信**:
     - Hook机制下，通信不占用GPU SM资源
     - 通信由RDMA网络接口完成

  4. **NVLink + RDMA混合**:
     - 节点内通信: NVLink (高带宽)
     - 跨节点通信: RDMA (NVSHMEM)

通信算子触发条件:
  - allgather_moe_tp:   attention_tp < moe_tp 时需要（MoE需要完整激活）
  - reduce_scatter_moe_tp: attention_tp > moe_tp 时需要（MoE内部ReduceScatter）
  - allgather_restore:    attention_tp < moe_tp 时需要（MoE输出恢复到attn_tp级别）
  - dispatch/combine:   EP > 1 时需要（EP通信）

TP转换逻辑:
  输入级别 → MoE处理级别 → 输出级别

  attention_tp = moe_tp:  输入=attn_tp → MoE=attn_tp → 输出=attn_tp (无通信)
  attention_tp > moe_tp:  输入=attn_tp → RS(attn_tp→moe_tp) → MoE=moe_tp
                        → RS(moe_tp) → AG(moe_tp→attn_tp) → 输出=attn_tp
  attention_tp < moe_tp:  输入=attn_tp → AG(attn_tp→moe_tp) → MoE=moe_tp
                        → RS(moe_tp→attn_tp) → 输出=attn_tp

参考:
  - DeepEP GitHub: https://github.com/deepseek-ai/DeepEP
  - DeepSeek-V3 Technical Report: https://arxiv.org/pdf/2412.19437
"""

from .module_base import ModuleBase
from ..layers import (
    LayerMoEGate,
    LayerExpertGateProj,
    LayerExpertUp,
    LayerExpertDown,
    LayerAll2All,
    LayerAllGather,
    LayerReduceScatter,
)


class ModuleMoE(ModuleBase):
    """MoE模块 (DeepEP优化版)

    支持DeepEP的compute-communication overlap优化:
    - Dispatch通信与Shared Expert计算并行
    - 根据Prefill/Decode选择high_throughput/low_latency模式
    """

    def __init__(self, hardware_config, model_config, deploy_config, quant_config, seq_len,
                 is_prefill=False, enable_overlap=True):
        """初始化MoE模块

        Args:
            seq_len: 序列长度
            is_prefill: 是否为Prefill阶段（影响DeepEP模式选择）
            enable_overlap: 是否启用DeepEP compute-communication overlap
        """
        super().__init__(hardware_config, model_config, deploy_config, quant_config)

        self.seq_len = seq_len
        self.hidden_size = model_config.hidden_size
        self.ep = deploy_config.expert_parallel
        self.moe_tp = deploy_config.moe_tp
        self.attention_tp = deploy_config.attention_tp
        self.n_routed_experts = model_config.num_experts
        self.top_k = model_config.num_experts_per_tok
        self.n_shared = getattr(model_config, 'num_shared_experts', 0)
        self.is_prefill = is_prefill
        self.enable_overlap = enable_overlap

        # DeepEP模式: Prefill用high_throughput, Decode用low_latency
        self.deepep_mode = 'high_throughput' if is_prefill else 'low_latency'

        self._build_layers()

    def _build_layers(self):
        """构建MoE的所有算子（按执行顺序，考虑DeepEP优化）"""
        batch_size = self.deploy_config.micro_batch_size
        act_bytes = self.quant_config.default_activation_transfer_bits / 8

        # ========== 1. e_topk_weight: Gate Routing ==========
        # hidden -> n_routed_experts, sigmoid + topk 选择
        self.add_layer(
            'e_topk_weight',
            LayerMoEGate(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len
            )
        )

        # ========== 2. allgather_moe_tp: TP 通信（attention_tp < moe_tp）==========
        if self.attention_tp < self.moe_tp:
            data_size = batch_size * self.seq_len * self.hidden_size * act_bytes
            self.add_layer(
                'allgather_moe_tp',
                LayerAllGather(
                    self.hardware_config, self.model_config,
                    self.deploy_config, self.quant_config,
                    data_size, self.attention_tp
                )
            )

        # ========== DeepEP优化: 计算Shared Expert时间用于overlap ==========
        shared_expert_time_ms = 0.0
        if self.n_shared > 0 and self.enable_overlap:
            # 计算Shared Expert的计算时间（用于与dispatch overlap）
            shared_up = LayerExpertUp(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len, top_k=1
            )
            shared_gate = LayerExpertGateProj(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len, top_k=1
            )
            shared_down = LayerExpertDown(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len, top_k=1
            )
            shared_expert_time_ms = (
                shared_up.get_cost_time() +
                shared_gate.get_cost_time() +
                shared_down.get_cost_time()
            )

        # ========== 3-5. Shared Expert (可与Dispatch重叠) ==========
        if self.n_shared > 0:
            self.add_layer(
                'share_up',
                LayerExpertUp(
                    self.hardware_config, self.model_config,
                    self.deploy_config, self.quant_config,
                    self.seq_len, top_k=1
                )
            )

            self.add_layer(
                'share_gate_proj',
                LayerExpertGateProj(
                    self.hardware_config, self.model_config,
                    self.deploy_config, self.quant_config,
                    self.seq_len, top_k=1
                )
            )

            self.add_layer(
                'share_down',
                LayerExpertDown(
                    self.hardware_config, self.model_config,
                    self.deploy_config, self.quant_config,
                    self.seq_len, top_k=1
                )
            )

        # ========== 6. dispatch: All-to-All (EP通信, DeepEP优化) ==========
        # DeepEP优化: dispatch可与Shared Expert计算重叠
        # 当 enable_overlap=True 且 n_shared > 0 时，
        # 通信时间 = max(dispatch_time, shared_expert_time)
        data_size_dispatch = 0.0
        is_cross_node = False

        if self.ep > 1:
            max_chips = self.hardware_config.max_chips_per_node
            if self.moe_tp >= max_chips:
                # EP通信在节点内，延迟可忽略
                data_size_dispatch = 0.0
            else:
                tokens = max(batch_size * self.seq_len / self.attention_tp, 1)
                data_size_dispatch = tokens * self.top_k * (self.ep - 1) / self.ep * self.hidden_size * act_bytes
                # 判断是否跨节点
                is_cross_node = (self.ep > max_chips / self.moe_tp)

            # DeepEP优化: 设置overlapable_compute_time
            overlap_time = shared_expert_time_ms if (self.enable_overlap and self.n_shared > 0) else 0.0

            self.add_layer(
                'dispatch',
                LayerAll2All(
                    self.hardware_config, self.model_config,
                    self.deploy_config, self.quant_config,
                    data_size_dispatch, self.ep,
                    mode=self.deepep_mode,
                    overlapable_compute_time_ms=overlap_time,
                    is_cross_node=is_cross_node
                )
            )

        # ========== 7-9. Routed Expert Compute ==========
        self.add_layer(
            'moe_up',
            LayerExpertUp(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len, top_k=self.top_k
            )
        )

        self.add_layer(
            'moe_gate_proj',
            LayerExpertGateProj(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len, top_k=self.top_k
            )
        )

        self.add_layer(
            'moe_down',
            LayerExpertDown(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len, top_k=self.top_k
            )
        )

        # ========== 10. combine: All-to-All (EP通信) ==========
        if self.ep > 1:
            self.add_layer(
                'combine',
                LayerAll2All(
                    self.hardware_config, self.model_config,
                    self.deploy_config, self.quant_config,
                    data_size_dispatch, self.ep,
                    mode=self.deepep_mode,
                    is_cross_node=is_cross_node
                )
            )

        # ========== 11. reduce_scatter_moe_tp: TP 通信（attention_tp > moe_tp）==========
        if self.attention_tp > self.moe_tp:
            data_size = batch_size * self.seq_len * self.hidden_size * act_bytes
            self.add_layer(
                'reduce_scatter_moe_tp',
                LayerReduceScatter(
                    self.hardware_config, self.model_config,
                    self.deploy_config, self.quant_config,
                    data_size, self.moe_tp
                )
            )
            self.add_layer(
                'allgather_restore',
                LayerAllGather(
                    self.hardware_config, self.model_config,
                    self.deploy_config, self.quant_config,
                    data_size, self.moe_tp
                )
            )

        # ========== 12. reduce_scatter_restore: TP 通信（attention_tp < moe_tp）==========
        if self.attention_tp < self.moe_tp:
            data_size = batch_size * self.seq_len * self.hidden_size * act_bytes
            self.add_layer(
                'reduce_scatter_restore',
                LayerReduceScatter(
                    self.hardware_config, self.model_config,
                    self.deploy_config, self.quant_config,
                    data_size, self.moe_tp
                )
            )
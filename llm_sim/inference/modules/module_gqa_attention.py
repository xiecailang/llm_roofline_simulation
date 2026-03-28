"""GQA Attention 模块 - 用于 Qwen 2.5 等使用 Grouped Query Attention 的模型

GQA (Grouped Query Attention) 结构:
1. [可选] AllGather Input (上游 TP < attention_TP)
2. Input RMSNorm
3. Q Projection (ColumnParallel)
4. K Projection (ColumnParallel)
5. V Projection (ColumnParallel)
6. GQA Attention (Flash Attention)
7. O Projection (RowParallel)
8. [可选] AllReduce Output (attention_TP > 1)
9. [可选] ReduceScatter Output (attention_TP > 下游 TP)

与 DSA/MLA 的差异:
- 无 LoRA 压缩 (Q/K/V 直接投影)
- 无 Lightning Indexer (非稀疏注意力)
- KV cache 存储 num_kv_heads × head_dim (非压缩 latent)

TP 影响:
- Q/K/V projection: ColumnParallel, 输出按 TP 切分
- O projection: RowParallel, 输入按 TP 切分
- Attention: 每个 TP 节点计算 num_heads/TP 个 Q head
"""

from .module_base import ModuleBase
from .module_attention_comm import ModuleAttentionTPComm
from ..layers import (
    LayerRMSNorm,
    LayerGQAQProj,
    LayerGQAKProj,
    LayerGQAVProj,
    LayerGQAOProj,
    LayerGQAAttention,
)


class ModuleGQAAttention(ModuleBase):
    """GQA Attention 模块"""

    def __init__(self, hardware_config, model_config, deploy_config, quant_config,
                 seq_len, is_prefill=True, upstream_tp=None, downstream_tp=None,
                 kv_seq_len=None):
        """初始化 GQA Attention 模块

        Args:
            seq_len: 查询序列长度 (本地 CP rank 的序列)
            is_prefill: 是否为 Prefill 阶段
            upstream_tp: 上游 TP 级别
            downstream_tp: 下游 TP 级别
            kv_seq_len: KV 序列长度覆盖 (用于 CP 场景)
                       Prefill + CP: kv_seq_len = effective_seq_len (完整序列)
                       Decode: kv_seq_len = input_length + 1 (缓存序列)
                       None: 自动推导
        """
        super().__init__(hardware_config, model_config, deploy_config, quant_config)

        self.seq_len = seq_len
        self.is_prefill = is_prefill
        self.hidden_size = model_config.hidden_size
        self.attention_tp = deploy_config.attention_tp
        self.upstream_tp = upstream_tp or self.attention_tp
        self.downstream_tp = downstream_tp or self.attention_tp
        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.head_dim
        self.num_heads_per_tp = self.num_heads // self.attention_tp

        # KV cache 长度
        if kv_seq_len is not None:
            self.kv_seq_len = kv_seq_len
        elif is_prefill:
            self.kv_seq_len = seq_len
        else:
            self.kv_seq_len = deploy_config.input_length + 1

        self._build_layers()

    def _build_layers(self):
        """构建 GQA Attention 的所有算子"""
        batch_size = self.deploy_config.micro_batch_size

        # ========== 1. [可选] AllGather Input ==========
        if self.upstream_tp < self.attention_tp:
            comm_module = ModuleAttentionTPComm(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                batch_size, self.seq_len, self.hidden_size,
                self.attention_tp, self.upstream_tp, self.attention_tp
            )
            for name, layer in comm_module.layers.items():
                if 'allgather' in name:
                    self.add_layer('allgather_input', layer)

        # ========== 2. Input RMSNorm ==========
        self.add_layer(
            'input_norm',
            LayerRMSNorm(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len, self.hidden_size
            )
        )

        # ========== 3. Q Projection (ColumnParallel) ==========
        self.add_layer(
            'q_proj',
            LayerGQAQProj(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len
            )
        )

        # ========== 4. K Projection (ColumnParallel) ==========
        self.add_layer(
            'k_proj',
            LayerGQAKProj(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len
            )
        )

        # ========== 5. V Projection (ColumnParallel) ==========
        self.add_layer(
            'v_proj',
            LayerGQAVProj(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len
            )
        )

        # ========== 6. GQA Attention (Flash Attention) ==========
        self.add_layer(
            'attention',
            LayerGQAAttention(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len, self.kv_seq_len, self.is_prefill
            )
        )

        # ========== 7. O Projection (RowParallel) ==========
        self.add_layer(
            'o_proj',
            LayerGQAOProj(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len
            )
        )

        # ========== 8. [可选] AllReduce Output ==========
        if self.attention_tp > 1:
            comm_module = ModuleAttentionTPComm(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                batch_size, self.seq_len, self.hidden_size,
                self.attention_tp, self.attention_tp, self.attention_tp
            )
            for name, layer in comm_module.layers.items():
                if 'allreduce' in name:
                    self.add_layer('allreduce_output', layer)

        # ========== 9. [可选] ReduceScatter Output ==========
        if self.attention_tp > self.downstream_tp:
            comm_module = ModuleAttentionTPComm(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                batch_size, self.seq_len, self.hidden_size,
                self.attention_tp, self.attention_tp, self.downstream_tp
            )
            for name, layer in comm_module.layers.items():
                if 'reduce_scatter' in name:
                    self.add_layer('reduce_scatter_output', layer)

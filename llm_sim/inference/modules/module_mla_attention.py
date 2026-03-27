"""MLA Attention模块 - DeepSeek-V3的Multi-head Latent Attention

算子序列（按执行顺序）：
1. [可选] allgather_input - AllGather (上游TP < attention_TP)
2. Input RMSNorm
3. Q LoRA: q_a_proj + q_a_norm + q_b_proj
4. KV LoRA: kv_a_proj + kv_a_norm + kv_b_proj
5. MLA Attention
6. O Projection (RowParallel)
7. [可选] allreduce_output - AllReduce (attention_TP > 1)
8. [可选] reduce_scatter_output - ReduceScatter (attention_TP > 下游TP)

TP影响：
- q_a_proj: ColumnParallel, 输出按TP切分
- q_b_proj: ColumnParallel, 输出按TP切分
- kv_a_proj: ReplicatedLinear, 不切分
- kv_b_proj: ColumnParallel, 输出按TP切分
- mla_attention: 每个TP节点计算 num_heads/TP
- o_proj: RowParallel, 输入按TP切分
"""

from .module_base import ModuleBase
from .module_attention_comm import ModuleAttentionTPComm
from ..layers import (
    LayerRMSNorm,
    LayerMLAQAProj,
    LayerMLAQBProj,
    LayerMLAKVAProj,
    LayerMLAKVBProj,
    LayerMLAAttention,
    LayerMatMul,
)


class ModuleMLAAttention(ModuleBase):
    """MLA Attention模块"""

    def __init__(self, hardware_config, model_config, deploy_config, quant_config,
                 seq_len, is_prefill=True, upstream_tp=None, downstream_tp=None):
        super().__init__(hardware_config, model_config, deploy_config, quant_config)

        self.seq_len = seq_len
        self.is_prefill = is_prefill
        self.hidden_size = model_config.hidden_size
        self.attention_tp = deploy_config.attention_tp
        self.upstream_tp = upstream_tp or self.attention_tp
        self.downstream_tp = downstream_tp or self.attention_tp
        self.num_heads = model_config.num_attention_heads
        self.v_head_dim = getattr(model_config, 'v_head_dim', 128)
        self.num_heads_per_tp = self.num_heads // self.attention_tp

        # KV cache长度
        if is_prefill:
            self.kv_seq_len = seq_len
        else:
            self.kv_seq_len = deploy_config.input_length + 1

        self._build_layers()

    def _build_layers(self):
        """构建MLA Attention的所有算子"""
        batch_size = self.deploy_config.micro_batch_size

        # ========== 1. [可选] AllGather Input (上游TP < attention_TP) ==========
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

        # ========== 3-4. Q LoRA压缩 (ColumnParallel) ==========
        self.add_layer(
            'q_a_proj',
            LayerMLAQAProj(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len
            )
        )

        q_lora_rank = getattr(self.model_config, 'q_lora_rank', 1536)
        q_lora_rank_per_tp = q_lora_rank // self.attention_tp
        self.add_layer(
            'q_a_norm',
            LayerRMSNorm(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len, q_lora_rank_per_tp
            )
        )

        self.add_layer(
            'q_b_proj',
            LayerMLAQBProj(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len
            )
        )

        # ========== 5-7. KV LoRA压缩 ==========
        self.add_layer(
            'kv_a_proj',
            LayerMLAKVAProj(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len
            )
        )

        kv_lora_rank = getattr(self.model_config, 'kv_lora_rank', 512)
        self.add_layer(
            'kv_a_norm',
            LayerRMSNorm(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len, kv_lora_rank
            )
        )

        self.add_layer(
            'kv_b_proj',
            LayerMLAKVBProj(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len
            )
        )

        # ========== 8. MLA Attention计算 ==========
        self.add_layer(
            'mla_attention',
            LayerMLAAttention(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len, self.kv_seq_len, self.is_prefill
            )
        )

        # ========== 9. O Projection (RowParallel) ==========
        m = batch_size * self.seq_len
        k = self.num_heads_per_tp * self.v_head_dim
        n = self.hidden_size
        self.add_layer(
            'o_proj',
            LayerMatMul(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                m, k, n,
                is_column_parallel=False  # RowParallel
            )
        )

        # ========== 10. [可选] AllReduce Output (attention_TP > 1) ==========
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

        # ========== 11. [可选] ReduceScatter Output (attention_TP > 下游TP) ==========
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
"""DSA Attention模块 - DeepSeek-V3.2的DeepSeek Sparse Attention

DSA 是 DeepSeek V3.2 引入的稀疏注意力机制，在 MLA 基础上增加 Lightning Indexer。

算子序列（按执行顺序）：
1. [可选] allgather_input - AllGather (上游TP < attention_TP)
2. Input RMSNorm
3. Q LoRA: q_a_proj + q_a_norm + q_b_proj
4. KV LoRA: kv_a_proj + kv_a_norm
5. Lightning Indexer (DSA特有):
   - indexer_wq_b: 从q_lora空间投影到indexer Q
   - indexer_wk: 从hidden空间投影到indexer K
   - indexer_k_norm: LayerNorm归一化
   - indexer_weights_proj: 计算MQA权重
   - sparse_attn_indexer: FP8 MQA + TopK选择
6. KV LoRA: kv_b_proj
7. DSA Attention (FlashMLA稀疏注意力)
8. O Projection (RowParallel)
9. [可选] allreduce_output - AllReduce (attention_TP > 1)
10. [可选] reduce_scatter_output - ReduceScatter (attention_TP > 下游TP)

关键特性：
- Lightning Indexer没有TP (ReplicatedLinear)
- Indexer使用FP8进行MQA计算
- Decode阶段只计算topk个重要token

TP影响：
- q_a_proj: ColumnParallel, 输出按TP切分
- q_b_proj: ColumnParallel, 输出按TP切分
- kv_a_proj: ReplicatedLinear, 不切分
- kv_b_proj: ColumnParallel, 输出按TP切分
- dsa_attention: 每个TP节点计算 num_heads/TP
- o_proj: RowParallel, 输入按TP切分
- indexer_*: 无TP (ReplicatedLinear)
"""

from .module_base import ModuleBase
from .module_attention_comm import ModuleAttentionTPComm
from ..layers import (
    LayerRMSNorm,
    LayerMLAQAProj,
    LayerMLAQBProj,
    LayerMLAKVAProj,
    LayerMLAKVBProj,
    LayerDSAAttention,
    LayerMatMul,
    LayerIndexerWQProj,
    LayerIndexerWKProj,
    LayerIndexerKNorm,
    LayerIndexerWeightsProj,
    LayerSparseAttnIndexer,
)


class ModuleDSAAttention(ModuleBase):
    """DSA Attention模块"""

    def __init__(self, hardware_config, model_config, deploy_config, quant_config,
                 seq_len, is_prefill=True, upstream_tp=None, downstream_tp=None,
                 kv_seq_len=None):
        """初始化 DSA Attention 模块

        Args:
            seq_len: 查询序列长度 (本地 CP rank 的序列)
            is_prefill: 是否为 Prefill 阶段
            upstream_tp: 上游 TP 级别
            downstream_tp: 下游 TP 级别
            kv_seq_len: KV 序列长度覆盖 (用于 CP 场景)
                       Prefill + CP: kv_seq_len = effective_seq_len (完整序列)
                       Decode: kv_seq_len = input_length + 1 (缓存序列)
                       None: 自动推导 (prefill=seq_len, decode=input_length+1)
        """
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
        if kv_seq_len is not None:
            # 显式指定 (CP 场景)
            self.kv_seq_len = kv_seq_len
        elif is_prefill:
            self.kv_seq_len = seq_len
        else:
            self.kv_seq_len = deploy_config.input_length + 1

        self._build_layers()

    def _build_layers(self):
        """构建DSA Attention的所有算子"""
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

        # ========== 3. Q LoRA压缩 (ColumnParallel) ==========
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

        # ========== 4. KV LoRA压缩 (前半部分) ==========
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

        # ========== 5. Lightning Indexer (DSA特有，Decode阶段启用) ==========
        # Indexer只在Decode阶段工作，Prefill阶段使用完整attention
        if not self.is_prefill:
            # 5.1 indexer_wq_b: q_lora_rank -> index_n_heads * index_head_dim
            # NO TP (ReplicatedLinear)
            self.add_layer(
                'indexer_wq_b',
                LayerIndexerWQProj(
                    self.hardware_config, self.model_config,
                    self.deploy_config, self.quant_config,
                    self.seq_len
                )
            )

            # 5.2 indexer_wk: hidden_size -> index_head_dim
            # NO TP (ReplicatedLinear)
            self.add_layer(
                'indexer_wk',
                LayerIndexerWKProj(
                    self.hardware_config, self.model_config,
                    self.deploy_config, self.quant_config,
                    self.seq_len
                )
            )

            # 5.3 indexer_k_norm: LayerNorm(index_head_dim)
            self.add_layer(
                'indexer_k_norm',
                LayerIndexerKNorm(
                    self.hardware_config, self.model_config,
                    self.deploy_config, self.quant_config,
                    self.seq_len
                )
            )

            # 5.4 indexer_weights_proj: hidden_size -> index_n_heads
            # NO TP, NO quantization
            self.add_layer(
                'indexer_weights_proj',
                LayerIndexerWeightsProj(
                    self.hardware_config, self.model_config,
                    self.deploy_config, self.quant_config,
                    self.seq_len
                )
            )

            # 5.5 SparseAttnIndexer: FP8 MQA + TopK
            self.add_layer(
                'sparse_attn_indexer',
                LayerSparseAttnIndexer(
                    self.hardware_config, self.model_config,
                    self.deploy_config, self.quant_config,
                    self.seq_len, self.kv_seq_len, self.is_prefill
                )
            )

        # ========== 6. KV LoRA压缩 (后半部分: kv_b_proj) ==========
        self.add_layer(
            'kv_b_proj',
            LayerMLAKVBProj(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len
            )
        )

        # ========== 7. DSA Attention计算 (FlashMLA稀疏注意力) ==========
        self.add_layer(
            'dsa_attention',
            LayerDSAAttention(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len, self.kv_seq_len, self.is_prefill
            )
        )

        # ========== 8. O Projection (RowParallel) ==========
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

        # ========== 9. [可选] AllReduce Output (attention_TP > 1) ==========
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

        # ========== 10. [可选] ReduceScatter Output (attention_TP > 下游TP) ==========
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

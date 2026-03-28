"""MiniMax M2.5 Prefill 模型

模型结构:
- Embedding
- N * TransformerBlock:
  - RMSNorm
  - GQA Attention (48 Q heads, 8 KV heads, head_dim=128)
  - RMSNorm
  - MoE FFN (256 experts, 8 per token, NO shared expert)
- RMSNorm
- LM Head
- No MTP (投机解码仅用于 Decode 阶段)

关键差异（vs Decode）:
1. seq_len = input_length × (1 - prefix_cache_hit_rate) / CP
2. Full attention (非 sparse DSA)
3. DeepEP 'high_throughput' mode (非 'low_latency')
4. CP communication (Ring Attention)
5. Prefix cache hit rate 减少有效计算量

CP (Context Parallelism) 支持:
- 序列按 token 切分到 CP ranks
- 每个 CP rank 处理 effective_seq_len / CP tokens
- Ring Attention 需要在每层 attention 后进行 CP 通信
- CP 通信量 = batch × (seq/CP) × num_kv_heads × head_dim × dtype × (CP-1) rounds

Prefix Cache 支持:
- 缓存公共前缀的 KV cache (如 system prompt)
- 命中的 token 不需要重新计算
- effective_seq_len = input_length × (1 - prefix_cache_hit_rate)
"""

from .inference_base import InferenceBase
from ..modules import ModuleGQAAttention, ModuleMoE, ModuleBase
from ..layers import (
    LayerEmbedding,
    LayerMatMul,
    LayerRMSNorm,
    LayerAllGather,
    LayerP2P,
    LayerCPComm,
)


class PrefillMiniMaxM25(InferenceBase):
    """MiniMax M2.5 Prefill 模型

    Prefill 阶段特点:
    - 完整序列处理 (seq_len >> 1)
    - Full causal attention
    - 支持 Context Parallelism (CP) 处理长序列
    - 支持 Prefix Cache 命中优化
    """

    def __init__(self, hardware_config, model_config, deploy_config, quant_config, pp_stage=0):
        """初始化 Prefill 模型

        Args:
            pp_stage: 当前建模的 PP stage（0-indexed），默认为 0
        """
        super().__init__(hardware_config, model_config, deploy_config, quant_config)

        self.num_layers = model_config.num_hidden_layers
        self.hidden_size = model_config.hidden_size
        self.vocab_size = model_config.vocab_size
        self.lm_head_tp = deploy_config.lm_head_tp

        # GQA 参数
        self.num_kv_heads = model_config.num_key_value_heads
        self.head_dim = model_config.head_dim

        # CP 配置
        self.cp = deploy_config.context_parallel

        # Prefix Cache 配置
        self.prefix_cache_hit_rate = getattr(deploy_config, 'prefix_cache_hit_rate', 0.0)

        # 计算有效 prefill 长度
        # effective_seq_len: 需要计算的 token 总数
        # seq_per_cp: 每个 CP rank 处理的 token 数
        self.effective_seq_len = int(deploy_config.input_length * (1 - self.prefix_cache_hit_rate))
        self.seq_len = self.effective_seq_len // self.cp if self.cp > 1 else self.effective_seq_len

        # PP 配置
        self.pp = deploy_config.pipeline_parallel
        self.pp_stage = pp_stage

        if self.pp > 1:
            self.layers_per_stage = self.num_layers // self.pp
            self.start_layer = self.pp_stage * self.layers_per_stage
            self.end_layer = self.start_layer + self.layers_per_stage
            if self.pp_stage == self.pp - 1:
                self.end_layer = self.num_layers
        else:
            self.start_layer = 0
            self.end_layer = self.num_layers

        self._build_modules()

    def _build_modules(self):
        """构建 Prefill 模型的所有模块

        与 Decode 的关键差异:
        1. seq_len = effective_seq_len / CP
        2. is_prefill=True (启用 full attention)
        3. kv_seq_len = effective_seq_len (CP 场景下完整序列)
        4. CP 通信层 (当 CP > 1)
        5. MoE 使用 high_throughput 模式
        6. 无 MTP
        """
        batch_size = self.deploy_config.micro_batch_size
        is_first_stage = (self.pp_stage == 0)
        is_last_stage = (self.pp_stage == self.pp - 1)

        # 1. Embedding (只在第一个 PP stage)
        if is_first_stage:
            embedding_layer = LayerEmbedding(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.effective_seq_len  # 使用完整序列长度
            )
            embedding_module = ModuleBase(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config
            )
            embedding_module.add_layer('embedding', embedding_layer)
            self.add_module('embedding', embedding_module)

        # 2. Transformer Blocks
        attention_tp = self.deploy_config.attention_tp
        moe_tp = self.deploy_config.moe_tp
        current_upstream_tp = attention_tp

        for layer_idx in range(self.start_layer, self.end_layer):
            downstream_tp = moe_tp

            # GQA Attention (Prefill: full attention)
            # 关键: kv_seq_len = effective_seq_len (CP 场景需要完整序列的 KV)
            attn_module = ModuleGQAAttention(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                seq_len=self.seq_len,
                is_prefill=True,  # Full attention
                upstream_tp=current_upstream_tp,
                downstream_tp=downstream_tp,
                kv_seq_len=self.effective_seq_len if self.cp > 1 else None  # CP: 完整序列 KV
            )
            self.add_module(f'layer_{layer_idx}_attention', attn_module)

            # CP Communication (Ring Attention)
            # GQA: KV cache 维度 = num_kv_heads × head_dim (无压缩)
            if self.cp > 1:
                cp_comm_module = ModuleBase(
                    self.hardware_config, self.model_config,
                    self.deploy_config, self.quant_config
                )
                cp_comm_module.add_layer(
                    'cp_comm',
                    LayerCPComm(
                        self.hardware_config, self.model_config,
                        self.deploy_config, self.quant_config,
                        batch_size=batch_size,
                        seq_per_cp=self.seq_len,
                        num_cp=self.cp,
                        kv_cache_size=self.num_kv_heads * self.head_dim  # GQA KV size
                    )
                )
                self.add_module(f'layer_{layer_idx}_cp_comm', cp_comm_module)

            # MoE FFN (Prefill: high_throughput 模式)
            ffn_module = ModuleMoE(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                seq_len=self.seq_len,
                is_prefill=True,  # high_throughput mode
                enable_overlap=True
            )
            current_upstream_tp = attention_tp
            self.add_module(f'layer_{layer_idx}_ffn', ffn_module)

        # 3. PP 通信 (非最后一个 stage)
        if self.pp > 1 and not is_last_stage:
            act_bytes = self.quant_config.default_activation_transfer_bits / 8
            data_size = batch_size * self.seq_len * self.hidden_size * act_bytes
            p2p_module = ModuleBase(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config
            )
            p2p_module.add_layer(
                'p2p_send',
                LayerP2P(
                    self.hardware_config, self.model_config,
                    self.deploy_config, self.quant_config,
                    data_size, is_cross_node=True
                )
            )
            self.add_module('p2p_send', p2p_module)

        # 4. LM Head (只在最后一个 PP stage)
        if is_last_stage:
            lm_head_module = ModuleBase(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config
            )

            # Final RMSNorm
            lm_head_module.add_layer(
                'norm',
                LayerRMSNorm(
                    self.hardware_config, self.model_config,
                    self.deploy_config, self.quant_config,
                    self.seq_len, self.hidden_size
                )
            )

            # LM Head Projection
            m = batch_size * self.seq_len
            k = self.hidden_size
            n = self.vocab_size // self.lm_head_tp if self.lm_head_tp > 1 else self.vocab_size
            lm_head_module.add_layer(
                'lm_head',
                LayerMatMul(
                    self.hardware_config, self.model_config,
                    self.deploy_config, self.quant_config,
                    m, k, n
                )
            )

            # AllGather (TP 通信)
            if self.lm_head_tp > 1:
                act_bytes = self.quant_config.default_activation_transfer_bits / 8
                data_size = batch_size * self.seq_len * n * act_bytes
                lm_head_module.add_layer(
                    'allgather',
                    LayerAllGather(
                        self.hardware_config, self.model_config,
                        self.deploy_config, self.quant_config,
                        data_size, self.lm_head_tp
                    )
                )

            self.add_module('lm_head', lm_head_module)

    def get_memory_usage(self):
        """计算 Prefill 阶段内存占用 (GB)

        Prefill 阶段需要存储完整 KV cache (或根据 prefix cache 减少)
        """
        w = self.quant_config.default_weight_bits / 8
        cache_bytes = self.quant_config.default_cache_write_bits / 8

        h = self.model_config.hidden_size
        v = self.model_config.vocab_size
        n_layers = self.model_config.num_hidden_layers
        n_heads = self.model_config.num_attention_heads
        n_kv_heads = self.model_config.num_key_value_heads
        head_dim = self.model_config.head_dim

        # 每层 Attention 参数 (GQA)
        attn_params = (
            h * n_heads * head_dim      # q_proj
            + h * n_kv_heads * head_dim  # k_proj
            + h * n_kv_heads * head_dim  # v_proj
            + n_heads * head_dim * h     # o_proj
            + 2 * h                      # norms
        )

        # MoE 参数 (无 shared expert)
        n_experts = self.model_config.num_experts or 1
        n_shared = getattr(self.model_config, 'num_shared_experts', 0) or 0
        moe_inter = getattr(self.model_config, 'moe_intermediate_size', h)
        moe_params = (
            h * n_experts
            + n_experts * moe_inter * h * 3
            + n_shared * moe_inter * h * 3
        )

        # Embedding + LM Head
        emb_params = v * h
        lm_head_params = v * h

        # 总参数
        total_params = emb_params + lm_head_params + n_layers * (attn_params + moe_params)

        # KV cache (GQA: num_kv_heads × head_dim per token)
        # Prefill: 只存储 effective_seq_len 个 token
        batch = self.deploy_config.micro_batch_size
        kv_cache = 2 * batch * self.effective_seq_len * n_layers * n_kv_heads * head_dim * cache_bytes

        return (total_params * w + kv_cache) / 1e9

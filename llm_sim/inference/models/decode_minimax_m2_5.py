"""MiniMax M2.5 Decode 模型

模型结构:
- Embedding
- N * TransformerBlock:
  - RMSNorm
  - GQA Attention (48 Q heads, 8 KV heads, head_dim=128)
  - RMSNorm
  - MoE FFN (256 experts, 8 per token, NO shared expert)
- RMSNorm
- LM Head
- MTP Layers (投机解码, 3 modules)

关键特性:
- MoE (Mixture of Experts): 256 experts, 8 per token
- NO Shared Expert (与 DeepSeek V3/Kimi K2.5 不同)
- GQA (Grouped Query Attention): 多个 Q head 共享 KV head
- RoPE: 标准 RoPE (head_dim=128, rotary_dim=64)
- MTP (Multi-Token Prediction): 3 个投机解码模块
- KV cache: num_kv_heads × head_dim per token (无压缩)

与 DeepSeek V3.2 的差异:
- GQA Attention (非 DSA/MLA)
- No Shared Expert (DeepSeek V3 有 1 个 shared expert)
- 标准 RoPE (非 MLA 的 qk_rope_head_dim 拆分)
"""

from .inference_base import InferenceBase
from ..modules import ModuleGQAAttention, ModuleMoE, ModuleMTPLayer, ModuleBase
from ..layers import LayerEmbedding, LayerMatMul, LayerRMSNorm, LayerAllGather, LayerP2P


class DecodeMiniMaxM25(InferenceBase):
    """MiniMax M2.5 Decode 模型

    GQA + MoE (No Shared Expert) + MTP
    """

    def __init__(self, hardware_config, model_config, deploy_config, quant_config, pp_stage=0):
        """初始化模型

        Args:
            pp_stage: 当前建模的 PP stage（0-indexed），默认为 0
                      当 PP > 1 时，每个 stage 只构建部分层
        """
        super().__init__(hardware_config, model_config, deploy_config, quant_config)

        # Decode 阶段: seq_len = 1
        self.seq_len = 1
        self.num_layers = model_config.num_hidden_layers
        self.hidden_size = model_config.hidden_size
        self.vocab_size = model_config.vocab_size
        self.lm_head_tp = deploy_config.lm_head_tp

        # GQA 参数
        self.num_kv_heads = model_config.num_key_value_heads
        self.head_dim = model_config.head_dim

        # MoE 参数
        self.num_experts = model_config.num_experts
        self.top_k = model_config.num_experts_per_tok
        self.num_shared_experts = getattr(model_config, 'num_shared_experts', 0)

        # MTP 参数
        self.mtp_length = deploy_config.mtp_length

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
        """构建 MiniMax M2.5 Decode 的所有模块

        PP 支持:
        - Stage 0: 构建 Embedding + 当前 stage 的 Transformer 层
        - Stage i (0 < i < PP-1): 只构建当前 stage 的 Transformer 层 + P2P send
        - Stage PP-1: 构建 Transformer 层 + LM Head + MTP
        """
        batch_size = self.deploy_config.micro_batch_size
        is_first_stage = (self.pp_stage == 0)
        is_last_stage = (self.pp_stage == self.pp - 1)

        # 1. Embedding (只在第一个 PP stage)
        if is_first_stage:
            embedding_layer = LayerEmbedding(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len
            )
            embedding_module = ModuleBase(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config
            )
            embedding_module.add_layer('embedding', embedding_layer)
            self.add_module('embedding', embedding_module)

        # 2. Transformer Blocks (只构建当前 PP stage 的层)
        attention_tp = self.deploy_config.attention_tp
        moe_tp = self.deploy_config.moe_tp
        current_upstream_tp = attention_tp

        for layer_idx in range(self.start_layer, self.end_layer):
            # 下游 TP: MoE 层用 moe_tp
            downstream_tp = moe_tp

            # GQA Attention
            attn_module = ModuleGQAAttention(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len, is_prefill=False,
                upstream_tp=current_upstream_tp,
                downstream_tp=downstream_tp
            )
            self.add_module(f'layer_{layer_idx}_attention', attn_module)

            # MoE FFN (MiniMax M2.5: 无 shared expert)
            ffn_module = ModuleMoE(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len,
                is_prefill=False,  # Decode 阶段使用 low_latency 模式
                enable_overlap=True  # DeepEP compute-communication overlap
            )
            # MoE 输出会转换回 attention_tp 级别
            current_upstream_tp = attention_tp
            self.add_module(f'layer_{layer_idx}_ffn', ffn_module)

        # 3. PP 通信 (非最后一个 stage 需要发送激活到下一个 stage)
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
                    data_size, is_cross_node=self.pp > 1
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

        # 5. MTP 验证层（投机解码，只在最后一个 stage）
        if is_last_stage and self.mtp_length > 0:
            for mtp_idx in range(self.mtp_length):
                mtp_module = ModuleMTPLayer(
                    self.hardware_config, self.model_config,
                    self.deploy_config, self.quant_config,
                    self.seq_len
                )
                self.add_module(f'mtp_{mtp_idx}', mtp_module)

    def get_memory_usage(self):
        """计算内存占用 (GB)

        MiniMax M2.5 参数:
        - Embedding: vocab_size * hidden_size
        - 每层 Attention (GQA):
          - q_proj: hidden * num_heads * head_dim
          - k_proj: hidden * num_kv_heads * head_dim
          - v_proj: hidden * num_kv_heads * head_dim
          - o_proj: num_heads * head_dim * hidden
          - input_norm + post_attn_norm: 2 * hidden
        - 每层 MoE:
          - gate: hidden * n_experts
          - routed experts: n_experts * (hidden * moe_intermediate * 3)
          - NO shared expert
        - LM Head: hidden * vocab_size
        - KV cache: 2 * batch * max_seq * layers * num_kv_heads * head_dim
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
            + 2 * h                      # norms (input + post_attn)
        )

        # MoE 参数 (无 shared expert)
        n_experts = self.model_config.num_experts or 1
        n_shared = getattr(self.model_config, 'num_shared_experts', 0) or 0
        moe_inter = getattr(self.model_config, 'moe_intermediate_size', h)
        moe_params = (
            h * n_experts                                       # gate
            + n_experts * moe_inter * h * 3                    # routed experts (gate+up+down)
            + n_shared * moe_inter * h * 3                     # shared experts (0 for MiniMax)
        )

        # Embedding + LM Head
        emb_params = v * h
        lm_head_params = v * h

        # 总参数
        total_params = emb_params + lm_head_params + n_layers * (attn_params + moe_params)

        # KV cache (GQA: 存储 num_kv_heads × head_dim per token)
        max_seq = self.deploy_config.input_length + self.deploy_config.output_length
        batch = self.deploy_config.micro_batch_size
        kv_cache = 2 * batch * max_seq * n_layers * n_kv_heads * head_dim * cache_bytes

        return (total_params * w + kv_cache) / 1e9

"""Qwen3.5 397B Decode 模型

模型结构:
- Embedding
- N * TransformerBlock (60 layers, 3:1 混合注意力):
  - RMSNorm
  - Linear Attention (Gated DeltaNet) 或 Full Attention (GQA)
  - RMSNorm
  - MoE FFN (512 experts, 10 per token, 1 shared expert)
- RMSNorm
- LM Head
- MTP Layer (投机解码, 1 module)

混合注意力:
- 每4层: 3层 Gated DeltaNet (线性注意力) + 1层 GQA (Full Attention)
- DeltaNet Decode: O(d²) 恒定计算量，不受 input_length 影响
- GQA Decode: O(T×d) 标准注意力，KV cache 随 input_length 增长
- 15个 Full Attention 层 (layer 3,7,11,...,59)
- 45个 Linear Attention 层

与 MiniMax M2.5 的差异:
- 混合注意力 vs 纯 GQA
- 1 shared expert vs 0 shared expert
- Linear attention 固定状态 vs Growing KV cache
"""

from .inference_base import InferenceBase
from ..modules import ModuleGQAAttention, ModuleLinearAttention, ModuleMoE, ModuleMTPLayer, ModuleBase
from ..layers import LayerEmbedding, LayerMatMul, LayerRMSNorm, LayerAllGather, LayerP2P


class DecodeQwen35(InferenceBase):
    """Qwen3.5 397B Decode 模型

    混合注意力 (Gated DeltaNet + GQA) + MoE FFN + MTP
    """

    def __init__(self, hardware_config, model_config, deploy_config, quant_config, pp_stage=0):
        """初始化模型

        Args:
            pp_stage: 当前建模的 PP stage（0-indexed），默认为 0
        """
        super().__init__(hardware_config, model_config, deploy_config, quant_config)

        # Decode 阶段: seq_len = 1
        self.seq_len = 1
        self.num_layers = model_config.num_hidden_layers
        self.hidden_size = model_config.hidden_size
        self.vocab_size = model_config.vocab_size
        self.lm_head_tp = deploy_config.lm_head_tp

        # 混合注意力配置
        self.layer_types = self._get_layer_types(model_config)
        self.num_full_layers = sum(1 for t in self.layer_types if t == "full_attention")
        self.num_linear_layers = self.num_layers - self.num_full_layers

        # GQA 参数 (Full Attention layers)
        self.num_kv_heads = model_config.num_key_value_heads
        self.head_dim = model_config.head_dim

        # Linear Attention 参数
        self.linear_num_key_heads = model_config.linear_num_key_heads
        self.linear_key_head_dim = model_config.linear_key_head_dim
        self.linear_num_value_heads = model_config.linear_num_value_heads
        self.linear_value_head_dim = model_config.linear_value_head_dim

        # MoE 参数
        self.num_experts = model_config.num_experts
        self.top_k = model_config.num_experts_per_tok
        self.num_shared_experts = getattr(model_config, 'num_shared_experts', 1)

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

    def _get_layer_types(self, model_config):
        """获取每层的注意力类型列表"""
        if model_config.layer_types is not None:
            return list(model_config.layer_types)
        # 根据 full_attention_interval 生成默认列表
        interval = model_config.full_attention_interval or 4
        layer_types = []
        for i in range(model_config.num_hidden_layers):
            if (i + 1) % interval == 0:
                layer_types.append("full_attention")
            else:
                layer_types.append("linear_attention")
        return layer_types

    def _build_modules(self):
        """构建 Qwen3.5 Decode 的所有模块"""
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

        # 2. Transformer Blocks
        attention_tp = self.deploy_config.attention_tp
        moe_tp = self.deploy_config.moe_tp
        current_upstream_tp = attention_tp

        for layer_idx in range(self.start_layer, self.end_layer):
            downstream_tp = moe_tp
            layer_type = self.layer_types[layer_idx]

            if layer_type == "full_attention":
                # Full Attention (GQA): 标准 softmax attention
                attn_module = ModuleGQAAttention(
                    self.hardware_config, self.model_config,
                    self.deploy_config, self.quant_config,
                    self.seq_len, is_prefill=False,
                    upstream_tp=current_upstream_tp,
                    downstream_tp=downstream_tp
                )
            else:
                # Linear Attention (Gated DeltaNet): O(d²) 恒定计算量
                attn_module = ModuleLinearAttention(
                    self.hardware_config, self.model_config,
                    self.deploy_config, self.quant_config,
                    self.seq_len, is_prefill=False,
                    upstream_tp=current_upstream_tp,
                    downstream_tp=downstream_tp
                )

            self.add_module(f'layer_{layer_idx}_attention', attn_module)

            # MoE FFN (Qwen3.5: 1 shared expert)
            ffn_module = ModuleMoE(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len,
                is_prefill=False,
                enable_overlap=True
            )
            current_upstream_tp = attention_tp
            self.add_module(f'layer_{layer_idx}_ffn', ffn_module)

        # 3. PP 通信
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

            lm_head_module.add_layer(
                'norm',
                LayerRMSNorm(
                    self.hardware_config, self.model_config,
                    self.deploy_config, self.quant_config,
                    self.seq_len, self.hidden_size
                )
            )

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
        # Qwen3.5 MTP 使用 GQA attention (非 linear attention)
        if is_last_stage and self.mtp_length > 0:
            for mtp_idx in range(self.mtp_length):
                mtp_module = ModuleMTPLayer(
                    self.hardware_config, self.model_config,
                    self.deploy_config, self.quant_config,
                    self.seq_len,
                    attention_type='gqa'
                )
                self.add_module(f'mtp_{mtp_idx}', mtp_module)

    def get_memory_usage(self):
        """计算内存占用 (GB)

        Qwen3.5 内存组成:
        - 模型参数: Embedding + LM Head + 每层 (Attn + MoE)
        - KV cache: 仅 Full Attention 层需要 (15层 × 2 KV heads × 256 dim)
        - 线性注意力固定状态: 45层 × 16 K heads × 128 × 128 (不随序列增长)
        """
        w = self.quant_config.default_weight_bits / 8
        cache_bytes = self.quant_config.default_cache_write_bits / 8

        h = self.model_config.hidden_size
        v = self.model_config.vocab_size
        n_layers = self.model_config.num_hidden_layers

        # Full Attention (GQA) 参数
        n_heads = self.model_config.num_attention_heads
        n_kv_heads = self.model_config.num_key_value_heads
        head_dim = self.model_config.head_dim

        # Linear Attention 参数
        lin_k_heads = self.linear_num_key_heads
        lin_k_dim = self.linear_key_head_dim
        lin_v_heads = self.linear_num_value_heads
        lin_v_dim = self.linear_value_head_dim

        # 每层 Linear Attention 参数
        linear_attn_params = (
            h * lin_v_heads * lin_v_dim      # q_proj
            + h * lin_k_heads * lin_k_dim    # k_proj
            + h * lin_v_heads * lin_v_dim    # v_proj
            + lin_v_heads * lin_v_dim * h    # o_proj
            + 2 * h                          # norms
        )

        # 每层 Full Attention (GQA) 参数
        full_attn_params = (
            h * n_heads * head_dim           # q_proj
            + h * n_kv_heads * head_dim      # k_proj
            + h * n_kv_heads * head_dim      # v_proj
            + n_heads * head_dim * h         # o_proj
            + 2 * h                          # norms
        )

        # MoE 参数 (512 experts + 1 shared)
        n_experts = self.model_config.num_experts or 1
        n_shared = self.num_shared_experts
        moe_inter = getattr(self.model_config, 'moe_intermediate_size', h)
        shared_inter = getattr(self.model_config, 'shared_expert_intermediate_size', moe_inter)
        moe_params = (
            h * n_experts                           # gate
            + n_experts * moe_inter * h * 3          # routed experts
            + n_shared * shared_inter * h * 3        # shared expert
        )

        # 总参数
        emb_params = v * h
        lm_head_params = v * h
        total_params = emb_params + lm_head_params + n_layers * moe_params
        total_params += self.num_linear_layers * linear_attn_params
        total_params += self.num_full_layers * full_attn_params

        # KV cache (仅 Full Attention 层)
        max_seq = self.deploy_config.input_length + self.deploy_config.output_length
        batch = self.deploy_config.micro_batch_size
        kv_cache = 2 * batch * max_seq * self.num_full_layers * n_kv_heads * head_dim * cache_bytes

        # 线性注意力固定状态 (不随序列长度增长)
        # 每层: num_key_heads × key_head_dim × value_head_dim
        linear_state = (self.num_linear_layers * lin_k_heads
                        * lin_k_dim * lin_v_dim * cache_bytes)

        return (total_params * w + kv_cache + linear_state) / 1e9

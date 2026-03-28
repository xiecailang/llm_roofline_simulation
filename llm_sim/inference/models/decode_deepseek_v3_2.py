"""DeepSeek-V3.2 Decode模型 - PD分离部署的Decode阶段

模型结构：
- Embedding
- N * TransformerBlock:
  - Layer 0 ~ (first_k_dense_replace-1): Dense FFN
  - Layer first_k_dense_replace ~ N-1: MoE
- LM Head
- MTP Layers (投机解码)

关键配置:
  - first_k_dense_replace: 前N层使用Dense FFN而非MoE
  - moe_layer_freq: MoE层间隔 (默认1，即每层都是MoE)

PP (Pipeline Parallel) 支持:
  - 模型按层切分到不同 PP stage
  - 每个 stage 只构建 num_layers / PP 层
  - Stage 0: Embedding + 前 N/PP 层
  - Stage i: 第 i*N/PP ~ (i+1)*N/PP 层
  - Stage PP-1: 后 N/PP 层 + LM Head + MTP
  - Stage 间通过 P2P 通信传递激活
"""

from .inference_base import InferenceBase
from ..modules import ModuleMLAAttention, ModuleDSAAttention, ModuleMoE, ModuleDenseFFN, ModuleMTPLayer
from ..modules.module_base import ModuleBase
from ..layers import LayerEmbedding, LayerMatMul, LayerRMSNorm, LayerAllGather, LayerP2P


class DecodeDeepSeekV32(InferenceBase):
    """DeepSeek-V3.2 Decode模型"""

    def __init__(self, hardware_config, model_config, deploy_config, quant_config, pp_stage=0):
        """初始化模型

        Args:
            pp_stage: 当前建模的 PP stage（0-indexed），默认为 0
                      当 PP > 1 时，每个 stage 只构建部分层
        """
        super().__init__(hardware_config, model_config, deploy_config, quant_config)

        # Decode阶段：seq_len=1
        self.seq_len = 1
        self.num_layers = model_config.num_hidden_layers
        self.hidden_size = model_config.hidden_size
        self.vocab_size = model_config.vocab_size
        self.lm_head_tp = deploy_config.lm_head_tp

        # Dense FFN 层数配置
        self.first_k_dense_replace = getattr(model_config, 'first_k_dense_replace', None) or 0
        self.moe_layer_freq = getattr(model_config, 'moe_layer_freq', None) or 1

        # Attention 类型: "mla" (Kimi K2.5, DeepSeek V3) 或 "dsa" (DeepSeek V3.2)
        self.attention_type = getattr(model_config, 'attention_type', 'mla')

        # PP 配置
        self.pp = deploy_config.pipeline_parallel
        self.pp_stage = pp_stage
        self.num_stages = self.pp

        # 计算当前 stage 的层范围
        if self.pp > 1:
            self.layers_per_stage = self.num_layers // self.pp
            self.start_layer = self.pp_stage * self.layers_per_stage
            self.end_layer = self.start_layer + self.layers_per_stage
            # 最后一个 stage 可能需要处理余数层
            if self.pp_stage == self.pp - 1:
                self.end_layer = self.num_layers
        else:
            self.start_layer = 0
            self.end_layer = self.num_layers

        self._build_modules()

    def _is_moe_layer(self, layer_idx: int) -> bool:
        """判断该层是否使用MoE

        逻辑（来自 vLLM）:
        - layer_idx >= first_k_dense_replace: 使用MoE
        - layer_idx % moe_layer_freq == 0: MoE层

        对于 DeepSeek V3.2:
        - first_k_dense_replace = 3
        - moe_layer_freq = 1
        - Layer 0, 1, 2: Dense FFN
        - Layer 3, 4, ..., 60: MoE
        """
        if self.model_config.num_experts is None:
            return False

        return (
            layer_idx >= self.first_k_dense_replace
            and layer_idx % self.moe_layer_freq == 0
        )

    def _build_modules(self):
        """构建DeepSeek-V3.2 Decode的所有模块

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
            # 将Layer包装成Module
            embedding_module = ModuleBase(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config
            )
            embedding_module.add_layer('embedding', embedding_layer)
            self.add_module('embedding', embedding_module)

        # 2. Transformer Blocks (只构建当前 PP stage 的层)
        attention_tp = self.deploy_config.attention_tp
        moe_tp = self.deploy_config.moe_tp

        # 追踪实际的 upstream_tp (每层FFN的输出TP级别)
        current_upstream_tp = attention_tp

        for layer_idx in range(self.start_layer, self.end_layer):
            # 判断该层FFN类型
            is_moe = self._is_moe_layer(layer_idx)

            # 下游TP：MoE层用moe_tp，Dense层用attention_tp
            downstream_tp = moe_tp if is_moe else attention_tp

            # Attention: MLA (Kimi K2.5, DeepSeek V3) 或 DSA (DeepSeek V3.2)
            if self.attention_type == 'dsa':
                attn_module = ModuleDSAAttention(
                    self.hardware_config, self.model_config,
                    self.deploy_config, self.quant_config,
                    self.seq_len, is_prefill=False,
                    upstream_tp=current_upstream_tp,
                    downstream_tp=downstream_tp
                )
            else:
                attn_module = ModuleMLAAttention(
                    self.hardware_config, self.model_config,
                    self.deploy_config, self.quant_config,
                    self.seq_len, is_prefill=False,
                    upstream_tp=current_upstream_tp,
                    downstream_tp=downstream_tp
                )
            self.add_module(f'layer_{layer_idx}_attention', attn_module)

            # FFN: Dense FFN 或 MoE
            if is_moe:
                # MoE层 (Decode阶段使用DeepEP low_latency模式)
                ffn_module = ModuleMoE(
                    self.hardware_config, self.model_config,
                    self.deploy_config, self.quant_config,
                    self.seq_len,
                    is_prefill=False,  # Decode阶段使用low_latency模式
                    enable_overlap=True  # 启用DeepEP compute-communication overlap
                )
                # MoE 输出会转换回 attention_tp 级别 (在 module_moe.py 中处理)
                current_upstream_tp = attention_tp
            else:
                # Dense FFN层 (前 first_k_dense_replace 层)
                ffn_module = ModuleDenseFFN(
                    self.hardware_config, self.model_config,
                    self.deploy_config, self.quant_config,
                    self.seq_len
                )
                # Dense FFN 有 AllReduce，输出 fully replicated
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

            # AllGather (TP通信)
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

        # 5. MTP验证层（投机解码，只在最后一个 stage）
        if is_last_stage:
            mtp_length = self.deploy_config.mtp_length
            for mtp_idx in range(mtp_length):
                mtp_module = ModuleMTPLayer(
                    self.hardware_config, self.model_config,
                    self.deploy_config, self.quant_config,
                    self.seq_len
                )
                self.add_module(f'mtp_{mtp_idx}', mtp_module)

    def get_memory_usage(self):
        """计算内存占用 (GB)

        参数内存精确计算（基于DeepSeek-V3.2架构）：
        - Embedding: vocab_size * hidden_size
        - 每层Attention (MLA):
          - q_a_proj: hidden * q_lora_rank
          - q_b_proj: q_lora_rank * (num_heads * qk_head_dim)
          - kv_a_proj: hidden * (kv_lora_rank + qk_rope_head_dim)
          - kv_b_proj: kv_lora_rank * (num_heads * (qk_nope + v_head_dim))
          - o_proj: num_heads * v_head_dim * hidden
          - norms: hidden * 2 + kv_lora_rank + q_lora_rank
        - 每层Dense FFN:
          - gate + up + down: 3 * hidden * intermediate
        - 每层MoE:
          - gate: hidden * n_routed_experts
          - routed experts: n_routed_experts * (hidden * moe_intermediate * 3)
          - shared experts: n_shared * (hidden * moe_intermediate * 3)
        - LM Head: hidden * vocab_size
        - KV cache (MLA压缩): 2 * batch * max_seq * layers * kv_lora_rank
        """
        w = self.quant_config.default_weight_bits / 8
        cache_bytes = self.quant_config.default_cache_write_bits / 8

        h = self.model_config.hidden_size
        v = self.model_config.vocab_size
        n_layers = self.model_config.num_hidden_layers
        n_heads = self.model_config.num_attention_heads
        q_lora = getattr(self.model_config, 'q_lora_rank', 1536)
        kv_lora = getattr(self.model_config, 'kv_lora_rank', 512)
        qk_nope = getattr(self.model_config, 'qk_nope_head_dim', 128)
        qk_rope = getattr(self.model_config, 'qk_rope_head_dim', 64)
        v_dim = getattr(self.model_config, 'v_head_dim', 128)

        # Dense FFN 参数
        intermediate = self.model_config.intermediate_size
        dense_ffn_params = 3 * h * intermediate  # gate + up + down

        # MoE 参数
        n_experts = self.model_config.num_experts or 1
        n_shared = getattr(self.model_config, 'num_shared_experts', 0) or 0
        moe_inter = getattr(self.model_config, 'moe_intermediate_size', intermediate)
        moe_params = (
            h * n_experts                                       # gate
            + n_experts * moe_inter * h * 3                    # routed experts (gate+up+down)
            + n_shared * moe_inter * h * 3                     # shared experts
        )

        # Embedding + LM Head (tie_word_embeddings=False时各自独立)
        emb_params = v * h
        lm_head_params = v * h

        # 每层Attention参数
        attn_params = (
            h * q_lora                                          # q_a_proj
            + q_lora * (n_heads * (qk_nope + qk_rope))         # q_b_proj
            + h * (kv_lora + qk_rope)                          # kv_a_proj
            + kv_lora * (n_heads * (qk_nope + v_dim))          # kv_b_proj
            + n_heads * v_dim * h                               # o_proj
            + h + kv_lora + q_lora + h                         # norms (input, kv_a, q_a, post_attn)
        )

        # 计算总参数（考虑Dense FFN和MoE的混合）
        total_params = emb_params + lm_head_params
        for layer_idx in range(n_layers):
            total_params += attn_params
            if self._is_moe_layer(layer_idx):
                total_params += moe_params
            else:
                total_params += dense_ffn_params

        # KV cache (MLA压缩：只存kv_lora_rank维度)
        max_seq = self.deploy_config.input_length + self.deploy_config.output_length
        batch = self.deploy_config.micro_batch_size
        kv_cache = 2 * batch * max_seq * n_layers * kv_lora * cache_bytes  # 2 for K and V latent

        return (total_params * w + kv_cache) / 1e9
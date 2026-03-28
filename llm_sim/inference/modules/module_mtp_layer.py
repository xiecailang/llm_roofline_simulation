"""MTP (Multi-Token Prediction) 验证层模块

DeepSeek-V3的MTP结构（基于vllm deepseek_mtp.py）：
- 每个MTP层包含：enorm + hnorm + eh_proj + 1个TransformerBlock
- eh_proj: [hidden*2, hidden] 融合embedding和hidden state
- 复用主模型的embedding和lm_head
- mtp_length个验证层串行执行

支持不同 Attention 类型:
- MLA (DeepSeek V3/Kimi K2.5)
- GQA (MiniMax M2.5)
"""

from .module_base import ModuleBase
from ..layers import LayerMatMul, LayerRMSNorm, LayerAllReduce, LayerAll2All
from ..layers import LayerMoEGate, LayerExpertGateUp, LayerExpertDown
from ..layers import LayerMLAQAProj, LayerMLAQBProj, LayerMLAKVAProj, LayerMLAKVBProj
from ..layers import LayerMLAAttention, LayerRMSNorm as LayerRMSNormAlias


class ModuleMTPLayer(ModuleBase):
    """MTP验证层模块

    结构：
    1. enorm + hnorm (对embedding和hidden state分别norm)
    2. eh_proj: [hidden*2 -> hidden] (融合)
    3. 1个TransformerBlock (Attention + MoE)

    支持 MLA 和 GQA 两种 Attention 类型
    """

    def __init__(self, hardware_config, model_config, deploy_config, quant_config, seq_len=1,
                 attention_type=None):
        super().__init__(hardware_config, model_config, deploy_config, quant_config)
        self.seq_len = seq_len
        self.hidden_size = model_config.hidden_size
        self.tp = deploy_config.attention_tp
        self.kv_seq_len = deploy_config.input_length + 1

        # Attention 类型: 从 model_config 或参数获取
        self.attention_type = attention_type or getattr(model_config, 'attention_type', 'mla')

        self._build_layers()

    def _build_layers(self):
        batch = self.deploy_config.micro_batch_size
        h = self.hidden_size
        tp = self.tp

        # 1. enorm: RMSNorm on embedding
        self.add_layer('enorm', LayerRMSNorm(
            self.hardware_config, self.model_config,
            self.deploy_config, self.quant_config,
            self.seq_len, h
        ))

        # 2. hnorm: RMSNorm on hidden state
        self.add_layer('hnorm', LayerRMSNorm(
            self.hardware_config, self.model_config,
            self.deploy_config, self.quant_config,
            self.seq_len, h
        ))

        # 3. eh_proj: [hidden*2 -> hidden] (ColumnParallel)
        self.add_layer('eh_proj', LayerMatMul(
            self.hardware_config, self.model_config,
            self.deploy_config, self.quant_config,
            m=batch * self.seq_len, k=h * 2, n=h,
            is_column_parallel=True
        ))

        # 4. Attention (根据 attention_type 选择)
        if self.attention_type in ('mla', 'dsa'):
            from .module_mla_attention import ModuleMLAAttention
            attn = ModuleMLAAttention(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len, is_prefill=False
            )
        elif self.attention_type == 'gqa':
            from .module_gqa_attention import ModuleGQAAttention
            attn = ModuleGQAAttention(
                self.hardware_config, self.model_config,
                self.deploy_config, self.quant_config,
                self.seq_len, is_prefill=False
            )
        else:
            raise ValueError(f"不支持的 attention_type: {self.attention_type}")

        for name, layer in attn.layers.items():
            self.add_layer(f'attn_{name}', layer)

        # 5. MoE (复用主模型的MoE结构, Decode阶段使用DeepEP low_latency模式)
        from .module_moe import ModuleMoE
        moe = ModuleMoE(
            self.hardware_config, self.model_config,
            self.deploy_config, self.quant_config,
            self.seq_len,
            is_prefill=False,  # MTP在Decode阶段，使用low_latency模式
            enable_overlap=True  # 启用DeepEP compute-communication overlap
        )
        for name, layer in moe.layers.items():
            self.add_layer(f'moe_{name}', layer)

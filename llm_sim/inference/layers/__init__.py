"""算子层 - 所有具体算子"""

from .layer_base import LayerBase
from .layer_matmul import LayerMatMul
from .layer_mla_q_a_proj import LayerMLAQAProj
from .layer_mla_q_b_proj import LayerMLAQBProj
from .layer_mla_kv_a_proj import LayerMLAKVAProj
from .layer_mla_kv_b_proj import LayerMLAKVBProj
from .layer_mla_attention import LayerMLAAttention
from .layer_dsa_attention import LayerDSAAttention
from .layer_rmsnorm import LayerRMSNorm
from .layer_embedding import LayerEmbedding
from .layer_dense_gate_proj import LayerDenseGateProj
from .layer_dense_up import LayerDenseUp
from .layer_dense_down import LayerDenseDown
from .layer_moe_gate import LayerMoEGate  # Router Gate (路由)
from .layer_expert_gate_proj import LayerExpertGateProj  # Expert Gate (SwiGLU gate分支)
from .layer_expert_up import LayerExpertUp  # Expert Up (SwiGLU up分支)
from .layer_expert_gate_up import LayerExpertGateUp  # Expert Gate+Up融合
from .layer_expert_down import LayerExpertDown  # Expert Down (SwiGLU down投影)
from .layer_allreduce import LayerAllReduce
from .layer_allgather import LayerAllGather
from .layer_reduce_scatter import LayerReduceScatter
from .layer_all2all import LayerAll2All
from .layer_p2p import LayerP2P, LayerP2PSend, LayerP2PRecv
from .layer_cp_comm import LayerCPComm  # Context Parallel communication
from .layer_indexer_wq_proj import LayerIndexerWQProj
from .layer_indexer_wk_proj import LayerIndexerWKProj
from .layer_indexer_k_norm import LayerIndexerKNorm
from .layer_indexer_weights_proj import LayerIndexerWeightsProj
from .layer_sparse_attn_indexer import LayerSparseAttnIndexer

# 别名，保持向后兼容
LayerMoEGateProj = LayerExpertGateProj
LayerMoEUp = LayerExpertUp
LayerMoEGateUp = LayerExpertGateUp
LayerMoEDown = LayerExpertDown

__all__ = [
    'LayerBase',
    'LayerMatMul',
    'LayerMLAQAProj',
    'LayerMLAQBProj',
    'LayerMLAKVAProj',
    'LayerMLAKVBProj',
    'LayerMLAAttention',
    'LayerDSAAttention',
    'LayerRMSNorm',
    'LayerEmbedding',
    'LayerDenseGateProj',
    'LayerDenseUp',
    'LayerDenseDown',
    'LayerMoEGate',
    'LayerExpertGateProj',
    'LayerExpertUp',
    'LayerExpertGateUp',
    'LayerExpertDown',
    # 别名
    'LayerMoEGateProj',
    'LayerMoEUp',
    'LayerMoEGateUp',
    'LayerMoEDown',
    # 通信
    'LayerAllReduce',
    'LayerAllGather',
    'LayerReduceScatter',
    'LayerAll2All',
    'LayerP2P',
    'LayerP2PSend',
    'LayerP2PRecv',
    'LayerCPComm',
    # Indexer
    'LayerIndexerWQProj',
    'LayerIndexerWKProj',
    'LayerIndexerKNorm',
    'LayerIndexerWeightsProj',
    'LayerSparseAttnIndexer',
]

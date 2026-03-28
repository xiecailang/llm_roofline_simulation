"""Gated DeltaNet 线性注意力算子

Gated DeltaNet (Qwen3.5) 核心特性:
- 线性注意力: 替代 O(T×d) 的 softmax attention，O(d²) per token
- 固定大小状态: S_t ∈ R^{d_k × d_v} per head，不随序列长度增长
- Delta rule 更新: S_t = β_t · S_{t-1} + (1 - β_t) · v_t ⊗ k_t^T

Decode vs Prefill:
- Decode: O(d_k × d_v) per token (固定计算量，与历史长度无关)
- Prefill: O(T × d_k × d_v) (T=seq_len，需遍历全序列构建状态)

与传统 Attention 的对比:
- Full Attention (GQA/MLA): Decode O(T × d) per token, T 随 input_length 增长
- Linear Attention (DeltaNet): Decode O(d²) per token, 常数级别

TP 影响:
- Q/V head 按 TP 切分: linear_num_value_heads / TP
- K head 按 TP 切分: linear_num_key_heads / TP
- 每个状态 S: [linear_key_head_dim, linear_value_head_dim]

性能建模要点:
- Decode: 计算量恒定，不受 input_length 影响
- KV cache: 固定大小 = num_key_heads × key_head_dim × value_head_dim per token (不含 T)
- 内存: 比 Full Attention 显著更低 (无 growing KV cache)
"""

from .layer_base import LayerBase


class LayerLinearAttention(LayerBase):
    """Gated DeltaNet 线性注意力计算

    计算:
    1. State Update: S_t = β_t · S_{t-1} + (1 - β_t) · v_t ⊗ k_t^T
       - FLOPs: 2 × num_key_heads × key_head_dim × value_head_dim
    2. Query: o_t = q_t @ S_t
       - FLOPs: 2 × num_value_heads × value_head_dim × key_head_dim
    """

    def __init__(self, hardware_config, model_config, deploy_config, quant_config,
                 seq_len, is_prefill=True):
        super().__init__(hardware_config, model_config, deploy_config, quant_config)
        self.seq_len = seq_len
        self.is_prefill = is_prefill

        # DeltaNet 线性注意力参数
        self.num_key_heads = model_config.linear_num_key_heads
        self.num_value_heads = model_config.linear_num_value_heads
        self.key_head_dim = model_config.linear_key_head_dim
        self.value_head_dim = model_config.linear_value_head_dim
        self.conv_kernel_dim = model_config.linear_conv_kernel_dim

        # TP 影响
        self.num_key_heads_per_tp = self.num_key_heads // self.tp
        self.num_value_heads_per_tp = self.num_value_heads // self.tp

    def get_cube_flops(self):
        """线性注意力 FLOPs

        1. State Update (每个 token):
           S_t = β_t · S_{t-1} + (1 - β_t) · v_t ⊗ k_t^T
           等效矩阵乘: 外积 v ⊗ k^T = [value_head_dim] × [key_head_dim]
           FLOPs = 2 × num_key_heads × key_head_dim × value_head_dim

        2. Query (每个 token):
           o_t = q_t @ S_t
           q_t: [num_value_heads, value_head_dim]
           S_t: [num_key_heads, key_head_dim, value_head_dim]
           每个价值头对所有键头状态做查询
           FLOPs = 2 × num_value_heads × value_head_dim × key_head_dim

        Prefill: seq_len × per_token_flops (逐 token 构建状态并查询)
        Decode: 1 × per_token_flops
        """
        # State update per token
        state_update_flops = (2 * self.num_key_heads_per_tp
                              * self.key_head_dim * self.value_head_dim)
        # Query per token
        query_flops = (2 * self.num_value_heads_per_tp
                       * self.value_head_dim * self.key_head_dim)
        # Per token total
        per_token_flops = state_update_flops + query_flops
        return self.batch_size * self.seq_len * per_token_flops

    def get_vector_flops(self):
        """Vector 操作: 门控 + Beta 计算 + Conv

        Gated DeltaNet 的非矩阵操作:
        - Beta gate (遗忘门): 每个键头的标量
        - Gating (SiLU/Sigmoid): Q, K, V 的门控
        - 1D Conv (linear_conv_kernel_dim=4): 局部卷积
        """
        # Beta gate 计算: sigmoid 每个键头
        beta_flops = self.batch_size * self.num_key_heads_per_tp * self.seq_len * 3  # exp+sum+div
        # 门控函数 (SiLU/Sigmoid on Q, K, V)
        gate_flops = (3 * self.batch_size * self.seq_len
                      * (self.num_value_heads_per_tp * self.value_head_dim  # Q gating
                         + self.num_key_heads_per_tp * self.key_head_dim     # K gating
                         + self.num_value_heads_per_tp * self.value_head_dim))  # V gating
        # 1D Conv: kernel_size × channels per token
        conv_flops = (2 * self.conv_kernel_dim
                      * self.batch_size * self.seq_len
                      * (self.num_key_heads_per_tp * self.key_head_dim))
        return beta_flops + gate_flops + conv_flops

    def get_mem_bytes(self):
        """内存访问

        DeltaNet 线性注意力的内存特点:
        - 无 growing KV cache (不存储历史 K, V)
        - 固定大小状态 S: [num_key_heads, key_head_dim, value_head_dim] per batch
        - 只需读取当前 token 的 Q, K, V + 状态 S

        Prefill: 从空状态开始构建，不需要读取历史状态
        Decode: 读取固定大小状态 S，更新后写回
        """
        # Q: [B, num_value_heads/TP, S, value_head_dim]
        read_q = (self.batch_size * self.num_value_heads_per_tp
                  * self.seq_len * self.value_head_dim * self.act_transfer_bytes)
        # K: [B, num_key_heads/TP, S, key_head_dim]
        read_k = (self.batch_size * self.num_key_heads_per_tp
                  * self.seq_len * self.key_head_dim * self.act_transfer_bytes)
        # V: [B, num_value_heads/TP, S, value_head_dim]
        read_v = (self.batch_size * self.num_value_heads_per_tp
                  * self.seq_len * self.value_head_dim * self.act_transfer_bytes)

        # State S: [num_key_heads/TP, key_head_dim, value_head_dim] per batch
        # Prefill: 从空状态开始，不需要读取
        # Decode: 读取上一 token 的状态
        if self.is_prefill:
            read_state = 0
        else:
            read_state = (self.batch_size * self.num_key_heads_per_tp
                          * self.key_head_dim * self.value_head_dim * self.cache_read_bytes)

        # Output: [B, num_value_heads/TP, S, value_head_dim]
        write_out = (self.batch_size * self.num_value_heads_per_tp
                     * self.seq_len * self.value_head_dim * self.act_transfer_bytes)

        return read_q + read_k + read_v + read_state + write_out

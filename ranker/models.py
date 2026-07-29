"""
models.py — 下游精排模型族
============================
5 种主流 CTR/CVR 精排模型:

| 模型        | 核心机制                                    |
|------------|--------------------------------------------|
| DNN        | Embedding + MLP                            |
| DeepFM     | FM 二阶交叉 + DNN                           |
| DCN        | Cross Network 显式高阶交叉 + Deep           |
| AutoInt    | Multi-Head Self-Attention 自动特征交互       |
| TokenMixer | Token Mixing (RMSNorm + PertokenSwiGLU)     |

所有模型统一实现 extract_embedding() 供 FALCON FM 优化。
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_ranker import BaseRanker


# ════════════════════════════════════════════════════════════
# DNN
# ════════════════════════════════════════════════════════════

class DNN(BaseRanker):
    """Embedding + MLP 基础模型"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mlp = self.build_mlp(self.concat_embed_dim, out_dim=1)
        # 用于 extract_embedding: 复用 MLP 前 N-1 层
        self.trunk = nn.Sequential(*list(self.mlp.children())[:-1])

    def forward(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.mlp(self.embed_input(sparse_values)).squeeze(-1)

    def extract_embedding(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.trunk(self.embed_input(sparse_values))


# ════════════════════════════════════════════════════════════
# DeepFM
# ════════════════════════════════════════════════════════════

class DeepFM(BaseRanker):
    """
    DeepFM = 一阶线性项 + FM 二阶交叉 + Deep MLP

    FM 二阶交叉高效算法:
        Σ_i Σ_j <v_i, v_j> = 0.5 * [ (Σ v_i)² - Σ v_i² ]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # 一阶线性项: 每个特征一个标量权重
        self.first_order = nn.ModuleDict({
            f: nn.Embedding(self.feature_voc_sizes[f], 1, padding_idx=0)
            for f in self.sparse_feature_names
        })

        self.mlp = self.build_mlp(self.concat_embed_dim, out_dim=1)
        self.trunk = nn.Sequential(*list(self.mlp.children())[:-1])
        # FM 二阶输出 (标量) 也参与 extract_embedding 的拼接
        self._fm_to_embed = nn.Linear(self.hidden_dims[-1] + 1, self.hidden_dims[-1])

    def _fm_second_order(self, stacked: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stacked: (B, N, embed_dim)
        Returns:
            (B,) 二阶交叉标量
        """
        sum_sq = stacked.sum(dim=1).pow(2)      # (B, D)
        sq_sum = stacked.pow(2).sum(dim=1)      # (B, D)
        return 0.5 * (sum_sq - sq_sum).sum(dim=1)

    def forward(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 一阶
        first = sum(self.first_order[f](sparse_values[f]).squeeze(-1)
                    for f in self.sparse_feature_names)

        # 二阶 (需要统一维度, 分组维度时降级为跳过)
        stacked = self.embed_input_stacked(sparse_values)
        second = self._fm_second_order(stacked)

        # Deep
        deep = self.mlp(self.embed_input(sparse_values)).squeeze(-1)

        return first + second + deep

    def extract_embedding(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        deep_repr = self.trunk(self.embed_input(sparse_values))           # (B, H)
        stacked = self.embed_input_stacked(sparse_values)
        fm = self._fm_second_order(stacked).unsqueeze(-1)                 # (B, 1)
        return self._fm_to_embed(torch.cat([deep_repr, fm], dim=-1))


# ════════════════════════════════════════════════════════════
# DCN
# ════════════════════════════════════════════════════════════

class CrossLayer(nn.Module):
    """
    Cross Layer: x_{l+1} = x_0 ⊙ (W x_l + b) + x_l

    显式建模高阶特征交叉，层数 = 交叉阶数。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        return x0 * self.linear(xl) + xl


class DCN(BaseRanker):
    """DCN = Cross Network (显式交叉) + Deep Network, 并行结构"""

    def __init__(self, num_cross_layers: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.num_cross_layers = num_cross_layers

        self.cross_layers = nn.ModuleList([
            CrossLayer(self.concat_embed_dim) for _ in range(num_cross_layers)
        ])

        # Deep 分支
        self.deep = self.build_mlp(self.concat_embed_dim, out_dim=self.hidden_dims[-1])
        # 融合 Cross 输出 + Deep 输出
        self.output = nn.Linear(self.concat_embed_dim + self.hidden_dims[-1], 1)
        self._embed_proj = nn.Linear(
            self.concat_embed_dim + self.hidden_dims[-1], self.hidden_dims[-1]
        )

    def _shared_repr(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        x0 = self.embed_input(sparse_values)
        xl = x0
        for layer in self.cross_layers:
            xl = layer(x0, xl)
        deep_out = self.deep(x0)
        return torch.cat([xl, deep_out], dim=-1)

    def forward(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.output(self._shared_repr(sparse_values)).squeeze(-1)

    def extract_embedding(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._embed_proj(self._shared_repr(sparse_values))


# ════════════════════════════════════════════════════════════
# AutoInt
# ════════════════════════════════════════════════════════════

class InteractingLayer(nn.Module):
    """
    AutoInt 交互层: Multi-Head Self-Attention + 残差。

    把每个特征 embedding 当作 token，用注意力自动学习特征交互。
    """

    def __init__(self, embed_dim: int, attention_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        inner = attention_dim * num_heads

        self.w_q = nn.Linear(embed_dim, inner, bias=False)
        self.w_k = nn.Linear(embed_dim, inner, bias=False)
        self.w_v = nn.Linear(embed_dim, inner, bias=False)
        self.w_res = nn.Linear(embed_dim, inner, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, embed_dim)
        B, N, _ = x.shape
        H, A = self.num_heads, self.attention_dim

        q = self.w_q(x).view(B, N, H, A).transpose(1, 2)   # (B, H, N, A)
        k = self.w_k(x).view(B, N, H, A).transpose(1, 2)
        v = self.w_v(x).view(B, N, H, A).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, N, H * A)

        out = self.norm(F.relu(attn + self.w_res(x)))
        return self.dropout(out)


class AutoInt(BaseRanker):
    """AutoInt = 多层 Multi-Head Self-Attention 特征交互 + 输出层"""

    def __init__(
        self,
        num_attention_layers: int = 3,
        num_attention_heads: int = 4,
        attention_dim: int = 64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_fields = len(self.sparse_feature_names)

        layers = []
        in_dim = self.embed_dim
        for _ in range(num_attention_layers):
            layers.append(InteractingLayer(
                in_dim, attention_dim, num_attention_heads, self.dropout_rate
            ))
            in_dim = attention_dim * num_attention_heads
        self.interacting_layers = nn.ModuleList(layers)

        self.flat_dim = self.num_fields * in_dim
        self.output = nn.Linear(self.flat_dim, 1)
        self._embed_proj = nn.Linear(self.flat_dim, self.hidden_dims[-1])

    def _shared_repr(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.embed_input_stacked(sparse_values)     # (B, N, embed_dim)
        for layer in self.interacting_layers:
            x = layer(x)
        return x.flatten(start_dim=1)                    # (B, N * inner)

    def forward(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.output(self._shared_repr(sparse_values)).squeeze(-1)

    def extract_embedding(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._embed_proj(self._shared_repr(sparse_values))


# ════════════════════════════════════════════════════════════
# TokenMixer
# ════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm (无 bias, 不算 mean)"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class PertokenSwiGLU(nn.Module):
    """pSwiGLU(x) = FC_down(Swish(FC_gate(x)) ⊙ FC_up(x))"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: Optional[int] = None,
                 dropout: float = 0.1):
        super().__init__()
        out_dim = out_dim or in_dim
        self.fc_gate = nn.Linear(in_dim, hidden_dim, bias=False)
        self.fc_up = nn.Linear(in_dim, hidden_dim, bias=False)
        self.fc_down = nn.Linear(hidden_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc_down(F.silu(self.fc_gate(x)) * self.fc_up(x)))


class TokenMixingReverting(nn.Module):
    """
    Token Mixing & Reverting:
      (B, T, D) → 沿 D 拆 H 组 → 沿 T 拼接 (B, T·H, D/H)
                → pSwiGLU → 还原 (B, T, D)

    用 MLP 沿 token 维做混合，替代 Self-Attention，计算更省。
    """

    def __init__(self, num_tokens: int, embed_dim: int, num_groups: int = 4,
                 ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_groups == 0, \
            f"embed_dim({embed_dim}) 必须能被 num_groups({num_groups}) 整除"
        self.num_tokens = num_tokens
        self.num_groups = num_groups
        self.group_dim = embed_dim // num_groups
        self.mixer = PertokenSwiGLU(
            self.group_dim, self.group_dim * ff_mult, self.group_dim, dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, G = self.num_groups, self.group_dim
        x = x.view(B, T * H, G)
        x = self.mixer(x)
        return x.view(B, T, D)


class TokenMixerBlock(nn.Module):
    """Pre-Norm Token Mixing + Pre-Norm Channel FFN, 双残差"""

    def __init__(self, num_tokens: int, embed_dim: int, num_groups: int = 4,
                 ff_mult: int = 4, channel_ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.token_mixer = TokenMixingReverting(
            num_tokens, embed_dim, num_groups, ff_mult, dropout
        )
        self.norm2 = RMSNorm(embed_dim)
        self.channel_ffn = PertokenSwiGLU(
            embed_dim, embed_dim * channel_ff_mult, embed_dim, dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.token_mixer(self.norm1(x))
        x = x + self.channel_ffn(self.norm2(x))
        return x


class TokenMixer(BaseRanker):
    """
    TokenMixer 精排模型 (RankMixer 风格)。

    每个稀疏特征 → 一个 token → Token Mixing + Channel FFN → Mean Pooling → 输出
    """

    def __init__(
        self,
        num_blocks: int = 3,
        num_groups: int = 4,
        ff_mult: int = 4,
        channel_ff_mult: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_tokens = len(self.sparse_feature_names)

        self.blocks = nn.ModuleList([
            TokenMixerBlock(
                self.num_tokens, self.embed_dim, num_groups,
                ff_mult, channel_ff_mult, self.dropout_rate,
            )
            for _ in range(num_blocks)
        ])
        self.final_norm = RMSNorm(self.embed_dim)
        self.output = self.build_mlp(self.embed_dim, out_dim=1)
        self._embed_proj = nn.Linear(self.embed_dim, self.hidden_dims[-1])

    def _shared_repr(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.embed_input_stacked(sparse_values)      # (B, T, embed_dim)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x).mean(dim=1)            # (B, embed_dim)

    def forward(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.output(self._shared_repr(sparse_values)).squeeze(-1)

    def extract_embedding(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._embed_proj(self._shared_repr(sparse_values))


# ════════════════════════════════════════════════════════════
# 工厂函数
# ════════════════════════════════════════════════════════════

RANKER_REGISTRY = {
    "dnn": DNN,
    "deepfm": DeepFM,
    "dcn": DCN,
    "autoint": AutoInt,
    "token_mixer": TokenMixer,
}


def create_ranker(
    ranker_type: str,
    sparse_feature_names: List[str],
    feature_voc_sizes: Dict[str, int],
    ranker_config,
) -> BaseRanker:
    """
    按类型创建 ranker 实例。

    Args:
        ranker_type: dnn / deepfm / dcn / autoint / token_mixer
        sparse_feature_names: 稀疏特征名列表
        feature_voc_sizes: {feat: vocab_size}
        ranker_config: RankerConfig 实例
    """
    if ranker_type not in RANKER_REGISTRY:
        raise ValueError(
            f"Unknown ranker_type: {ranker_type}, choose from {list(RANKER_REGISTRY)}"
        )

    common = dict(
        sparse_feature_names=sparse_feature_names,
        feature_voc_sizes=feature_voc_sizes,
        embed_dim=ranker_config.embed_dim,
        hidden_dims=list(ranker_config.hidden_dims),
        dropout=ranker_config.dropout,
        activation=ranker_config.activation,
    )

    if ranker_type == "dcn":
        return DCN(num_cross_layers=ranker_config.num_cross_layers, **common)
    if ranker_type == "autoint":
        return AutoInt(
            num_attention_layers=ranker_config.num_attention_layers,
            num_attention_heads=ranker_config.num_attention_heads,
            attention_dim=ranker_config.attention_dim,
            **common,
        )
    if ranker_type == "token_mixer":
        return TokenMixer(
            num_blocks=ranker_config.num_blocks,
            num_groups=ranker_config.num_groups,
            ff_mult=ranker_config.ff_mult,
            channel_ff_mult=ranker_config.channel_ff_mult,
            **common,
        )
    return RANKER_REGISTRY[ranker_type](**common)

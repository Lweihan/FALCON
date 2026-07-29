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


# ════════════════════════════════════════════════════════════
# MACRanker: 处理 MAC 数据集的多模态特征
# ════════════════════════════════════════════════════════════

class MACRanker(nn.Module):
    """
    适配 alimamaTech/MAC 数据集的精排模型。

    特征处理:
      1. 稀疏 ID (user_id, ad_feat* 等) → embedding lookup → concat
      2. 序列 ID (ad_feat*_seq) → embedding lookup → mean pool → concat
      3. 序列实值 (mm_feat*_seq) → mean pool → 线性投影 → concat

    所有特征拼接后送入 backbone (dnn / deepfm / dcn / autoint / token_mixer)。
    同时暴露 extract_embedding() 供 FM 优化器使用。
    """

    def __init__(
        self,
        ranker_type: str,
        vocab_sizes: Dict[str, int],           # {field: vocab_size}
        sparse_fields: List[str],
        seq_id_fields: List[str],
        mm_feat_dim: int,                       # mm_feat mean pool 后的原始维度
        embed_dim: int = 16,
        mm_proj_dim: int = 64,                  # mm_feat 投影后维度
        hidden_dims: List[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        num_cross_layers: int = 3,
        num_attention_layers: int = 3,
        num_attention_heads: int = 4,
        attention_dim: int = 64,
        num_blocks: int = 3,
        num_groups: int = 4,
        ff_mult: int = 4,
        channel_ff_mult: int = 4,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128, 64]

        self.sparse_fields = sparse_fields
        self.seq_id_fields = seq_id_fields
        self.mm_feat_dim = mm_feat_dim
        self.embed_dim = embed_dim

        # 稀疏 ID embedding
        self.sparse_embeds = nn.ModuleDict({
            f: nn.Embedding(vocab_sizes.get(f, 2), embed_dim, padding_idx=0)
            for f in sparse_fields
        })

        # 序列 ID embedding (与稀疏 ID 共用词表大小, 独立参数)
        self.seq_id_embeds = nn.ModuleDict({
            f: nn.Embedding(vocab_sizes.get(f, 2), embed_dim, padding_idx=0)
            for f in seq_id_fields
        })

        # mm_feat 投影层
        self.mm_proj = nn.Sequential(
            nn.Linear(mm_feat_dim, mm_proj_dim),
            nn.ReLU(),
            nn.Linear(mm_proj_dim, mm_proj_dim),
        )
        self.mm_proj_dim = mm_proj_dim

        # 总输入维度
        n_sparse = len(sparse_fields) * embed_dim
        n_seq_id = len(seq_id_fields) * embed_dim
        in_dim = n_sparse + n_seq_id + mm_proj_dim

        # Backbone
        common = dict(
            sparse_feature_names=["__fused__"],
            feature_voc_sizes={"__fused__": 2},   # 占位，不实际用 embedding
            embed_dim=embed_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
        )

        self._backbone_type = ranker_type
        if ranker_type == "dnn":
            self.backbone = self._build_mlp(in_dim, hidden_dims, dropout, activation)
        elif ranker_type == "deepfm":
            self.backbone = self._build_mlp(in_dim, hidden_dims, dropout, activation)
            # FM 二阶交叉: 把 in_dim 拆成若干 embed_dim 的片段
            self._n_fields = (in_dim + embed_dim - 1) // embed_dim
        elif ranker_type == "dcn":
            self.cross_layers = nn.ModuleList([
                nn.Linear(in_dim, in_dim) for _ in range(num_cross_layers)
            ])
            self.backbone = self._build_mlp(in_dim * 2, hidden_dims, dropout, activation)
        elif ranker_type in ("autoint", "token_mixer"):
            # 用 MLP 近似
            self.backbone = self._build_mlp(in_dim, hidden_dims, dropout, activation)
        else:
            self.backbone = self._build_mlp(in_dim, hidden_dims, dropout, activation)

        # 输出层
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        self._embed_out_dim = hidden_dims[-1]

    def _build_mlp(
        self, in_dim: int, hidden_dims: List[int],
        dropout: float, activation: str,
    ) -> nn.Sequential:
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}.get(activation, nn.ReLU)
        layers = []
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), act_fn(), nn.Dropout(dropout)]
            in_dim = h
        return nn.Sequential(*layers)

    def _fuse_features(
        self,
        sparse_ids: Dict[str, torch.Tensor],   # {field: (B,)}
        seq_id_lists: Dict[str, List[List[str]]],
        mm_feats: torch.Tensor,                # (B, mm_dim)
        vocab=None,
    ) -> torch.Tensor:
        """融合所有特征 → (B, in_dim)"""
        parts = []
        B = mm_feats.size(0)
        device = mm_feats.device

        # 稀疏 ID
        for f in self.sparse_fields:
            idx = sparse_ids.get(f)
            if idx is None:
                idx = torch.zeros(B, dtype=torch.long, device=device)
            else:
                idx = idx.to(device)
            parts.append(self.sparse_embeds[f](idx))   # (B, embed_dim)

        # 序列 ID → mean pool
        for f in self.seq_id_fields:
            seqs = seq_id_lists.get(f, [[] for _ in range(B)])
            # 构建 (B, max_seq_len) 的 idx 张量
            if vocab is not None:
                all_idx = [
                    [vocab.lookup(f, v) for v in s] if s else [0]
                    for s in seqs
                ]
            else:
                all_idx = [[0] for _ in range(B)]
            max_len = max(len(s) for s in all_idx)
            padded = torch.zeros(B, max_len, dtype=torch.long, device=device)
            mask = torch.zeros(B, max_len, dtype=torch.float32, device=device)
            for i, s in enumerate(all_idx):
                padded[i, :len(s)] = torch.tensor(s, dtype=torch.long)
                mask[i, :len(s)] = 1.0
            emb = self.seq_id_embeds[f](padded)   # (B, max_len, embed_dim)
            # mean pool
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = (emb * mask.unsqueeze(-1)).sum(dim=1) / denom  # (B, embed_dim)
            parts.append(pooled)

        # mm_feat 投影
        parts.append(self.mm_proj(mm_feats.to(device)))  # (B, mm_proj_dim)

        return torch.cat(parts, dim=-1)   # (B, in_dim)

    def extract_embedding(
        self,
        sparse_ids: Dict[str, torch.Tensor],
        seq_id_lists: Dict[str, List],
        mm_feats: torch.Tensor,
        vocab=None,
    ) -> torch.Tensor:
        """
        提取融合 embedding (不含输出层)。

        供 FM 优化器和 CF-OPD 打分使用。
        """
        x = self._fuse_features(sparse_ids, seq_id_lists, mm_feats, vocab)
        h = self.backbone(x)
        return h   # (B, hidden_dims[-1])

    @property
    def embed_output_dim(self) -> int:
        return self._embed_out_dim

    def predict_from_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        直接从融合 embedding 预测 CVR (跳过特征 lookup)。

        CF-OPD 打分时使用: FM 预测的未来 embedding → CVR 分
        """
        return torch.sigmoid(self.output_layer(embedding)).squeeze(-1)

    def forward(
        self,
        sparse_ids: Dict[str, torch.Tensor],
        seq_id_lists: Dict[str, List],
        mm_feats: torch.Tensor,
        vocab=None,
    ) -> Dict[str, torch.Tensor]:
        x = self._fuse_features(sparse_ids, seq_id_lists, mm_feats, vocab)

        if self._backbone_type == "dcn":
            # Cross layers
            x0 = x
            xc = x
            for layer in self.cross_layers:
                xc = x0 * layer(xc) + xc
            h = self.backbone(torch.cat([x, xc], dim=-1))
        else:
            h = self.backbone(x)

        logit = self.output_layer(h)
        return {
            "logit": logit.squeeze(-1),
            "prob": torch.sigmoid(logit).squeeze(-1),
            "embedding": h,
        }


def create_mac_ranker(ranker_type: str, feature_info: dict, ranker_config) -> MACRanker:
    """
    创建适配 MAC 数据集的 Ranker。

    Args:
        ranker_type: dnn / deepfm / dcn / autoint / token_mixer
        feature_info: create_mac_dataloaders 返回的 feature_info
        ranker_config: RankerConfig
    """
    return MACRanker(
        ranker_type=ranker_type,
        vocab_sizes=feature_info["vocab_sizes"],
        sparse_fields=feature_info["sparse_fields"],
        seq_id_fields=feature_info["seq_id_fields"],
        mm_feat_dim=feature_info["mm_feat_dim"],
        embed_dim=ranker_config.embed_dim,
        mm_proj_dim=ranker_config.mm_feat_proj_dim if ranker_config.mm_feat_proj_dim > 0 else 64,
        hidden_dims=list(ranker_config.hidden_dims),
        dropout=ranker_config.dropout,
        activation=ranker_config.activation,
        num_cross_layers=ranker_config.num_cross_layers,
        num_attention_layers=ranker_config.num_attention_layers,
        num_attention_heads=ranker_config.num_attention_heads,
        attention_dim=ranker_config.attention_dim,
        num_blocks=ranker_config.num_blocks,
        num_groups=ranker_config.num_groups,
        ff_mult=ranker_config.ff_mult,
        channel_ff_mult=ranker_config.channel_ff_mult,
    )

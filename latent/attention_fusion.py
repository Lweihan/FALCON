"""
attention_fusion.py — 注意力融合模块
======================================
将 4 部分分部分 embedding 融合为统一用户 embedding。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PartProjection(nn.Module):
    """单部分的线性投影层"""

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(embed_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(self.proj(x))


class AttentionFusion(nn.Module):
    """
    基于注意力机制的多部分 Embedding 融合器。

    流程:
    1. 每个部分独立投影 + section type embedding
    2. Multi-Head Self-Attention 部分间交互
    3. FFN + 残差
    4. 拼接 → 线性投影 → output_dim
    """

    NUM_SECTIONS = 4
    SECTION_NAMES = ['static_information', 'interest_profile', 'consumption_profile', 'demand_targets']

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.part_projections = nn.ModuleList([
            PartProjection(embed_dim, hidden_dim) for _ in range(self.NUM_SECTIONS)
        ])
        self.section_type_embedding = nn.Parameter(
            torch.randn(self.NUM_SECTIONS, hidden_dim) * 0.02
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * self.NUM_SECTIONS, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        static_emb: torch.Tensor,
        interest_emb: torch.Tensor,
        consumption_emb: torch.Tensor,
        demand_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args: 4 个 (batch, embed_dim)
        Returns: (batch, output_dim)
        """
        batch_size = static_emb.size(0)
        parts = [static_emb, interest_emb, consumption_emb, demand_emb]

        projected = []
        for i, part in enumerate(parts):
            proj = self.part_projections[i](part)
            proj = proj + self.section_type_embedding[i].unsqueeze(0)
            projected.append(proj)

        seq = torch.stack(projected, dim=1)  # (B, 4, H)
        attn_out, _ = self.attention(seq, seq, seq)
        seq = self.attention_norm(seq + attn_out)
        ffn_out = self.ffn(seq)
        seq = self.ffn_norm(seq + ffn_out)
        seq_flat = seq.reshape(batch_size, -1)
        return self.output_proj(seq_flat)

    def forward_from_dict(self, emb_dict: dict) -> torch.Tensor:
        return self.forward(
            static_emb=emb_dict['static_information_emb'],
            interest_emb=emb_dict['interest_profile_emb'],
            consumption_emb=emb_dict['consumption_profile_emb'],
            demand_emb=emb_dict['demand_targets_emb'],
        )

    def forward_from_stacked(self, stacked_emb: torch.Tensor) -> torch.Tensor:
        """stacked_emb: (batch, 4, embed_dim)"""
        return self.forward(*[stacked_emb[:, i] for i in range(4)])

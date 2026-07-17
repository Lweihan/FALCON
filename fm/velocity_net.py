"""
velocity_net.py — 速度场预测网络 v_θ
======================================
提供两种 backbone:
  - VelocityNetwork (MLP): 轻量快速
  - TransformerVelocityNetwork (Transformer): 更强表达力 (默认推荐)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """正弦时间编码"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# ────────────────────────────────────────────────────────────
# MLP Backbone (轻量)
# ────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.block(x))


class VelocityNetwork(nn.Module):
    """
    速度场预测网络 v_θ(x_t, t, condition) — MLP 版本。

    架构: [x_t, time_emb, cond_emb] → MLP (ResidualBlocks) → v_pred
    参数量约 3-5M (取决于 hidden_dim 和 num_layers)。
    """

    def __init__(
        self,
        data_dim: int = 256,
        time_dim: int = 128,
        cond_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.data_dim = data_dim

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim), nn.SiLU(),
        )

        input_dim = data_dim + time_dim + hidden_dim

        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 2):
            layers.append(ResidualBlock(hidden_dim, dropout))
        layers.extend([nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, data_dim)])

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size = x_t.size(0)
        device = x_t.device

        time_emb = self.time_embed(t)
        cond_emb = self.cond_proj(condition) if condition is not None else \
            torch.zeros(batch_size, self.cond_proj[0].out_features, device=device)

        h = torch.cat([x_t, time_emb, cond_emb], dim=-1)
        return self.mlp(h)


# ────────────────────────────────────────────────────────────
# Transformer Backbone (推荐默认)
# ────────────────────────────────────────────────────────────

class TransformerVelocityNetwork(nn.Module):
    """
    速度场预测网络 v_θ(x_t, t, condition) — Transformer 版本。

    架构思路 (类 DiT):
    1. 将 x_t, time_emb, cond_emb 映射为 3 个 token
    2. 叠加 learnable type embedding 区分三者
    3. 通过 TransformerEncoder 堆叠建模三者交互
    4. 从 x_t 对应位置取输出，回归到 data_dim 速度场

    参数量约 20-50M (取决于 transformer_dim 和 depth)。
    """

    def __init__(
        self,
        data_dim: int = 256,
        time_dim: int = 128,
        cond_dim: int = 256,
        transformer_dim: int = 1024,
        depth: int = 8,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.transformer_dim = transformer_dim

        # 时间编码
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, transformer_dim), nn.SiLU(),
            nn.Linear(transformer_dim, transformer_dim),
        )

        # 输入投影: 将 x_t 和 condition 映射到 transformer_dim
        self.x_proj = nn.Linear(data_dim, transformer_dim)
        self.cond_proj = nn.Linear(cond_dim, transformer_dim)

        # Learnable type embeddings: 区分 x_t / time / condition 三个 token
        self.type_embed = nn.Parameter(torch.randn(3, transformer_dim) * 0.02)

        # AdaLN-Zero 风格: 用 time embedding 调制每层
        # 简化实现: 将 time embedding 作为第一个 token 参与注意力
        # (等价于让 time token 与 x_t / cond 交互)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=heads,
            dim_feedforward=int(transformer_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN (更稳定)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth,
            norm=nn.LayerNorm(transformer_dim),
        )

        # 输出头: 从 token 0 (x_t) 位置回归速度场
        self.norm_out = nn.LayerNorm(transformer_dim)
        self.head = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.GELU(),
            nn.Linear(transformer_dim, data_dim),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x_t: (batch, data_dim) 加噪状态
            t: (batch,) 时间步
            condition: (batch, cond_dim) 条件 embedding

        Returns:
            v_pred: (batch, data_dim) 预测速度场
        """
        batch_size = x_t.size(0)
        device = x_t.device

        # 投影到 transformer_dim
        x_tok = self.x_proj(x_t)                   # (batch, D)
        t_tok = self.time_embed(t)                  # (batch, D)
        c_tok = self.cond_proj(condition) if condition is not None else \
            torch.zeros(batch_size, self.transformer_dim, device=device)

        # 堆叠为 3-token 序列: [x_t, time, condition]
        tokens = torch.stack([x_tok, t_tok, c_tok], dim=1)  # (batch, 3, D)

        # 加 type embedding
        tokens = tokens + self.type_embed.unsqueeze(0)       # (batch, 3, D)

        # Transformer 编码
        h = self.transformer(tokens)  # (batch, 3, D)

        # 取 x_t 位置的输出回归速度场
        out = self.norm_out(h[:, 0])  # (batch, D)
        v_pred = self.head(out)       # (batch, data_dim)

        return v_pred

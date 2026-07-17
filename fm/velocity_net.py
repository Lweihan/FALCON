"""
velocity_net.py — 速度场预测网络 v_θ
======================================
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
    速度场预测网络 v_θ(x_t, t, condition)。

    架构: [x_t, time_emb, cond_emb] → MLP (ResidualBlocks) → v_pred
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

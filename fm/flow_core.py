"""
flow_core.py — Flow Matching 核心数学
=======================================
OT-CFM 前向过程 (加噪) 及速度场定义。
"""

import torch


def flow_match_forward(
    x_clean: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    OT-CFM 前向过程:
        x_t = (1 - t) * x_clean + t * noise
        v_t = noise - x_clean

    Args:
        x_clean: (batch, dim) 干净 embedding
        t: (batch,) 时间步 [0, 1]
        noise: (batch, dim) 噪声，None 则随机采样

    Returns:
        x_t: (batch, dim) 加噪后状态
        v_t: (batch, dim) 真实速度场
    """
    if noise is None:
        noise = torch.randn_like(x_clean)
    t = t.view(-1, 1)
    x_t = (1.0 - t) * x_clean + t * noise
    v_t = noise - x_clean
    return x_t, v_t

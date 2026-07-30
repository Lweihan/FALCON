"""
utils.py — 通用工具函数
=========================
"""

import os
import random
import torch
import numpy as np
from typing import Optional


def set_seed(seed: int = 42):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """统计模型可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ────────────────────────────────────────────────────────────
# AMP 兼容层
# torch >= 2.0 用 torch.amp.*; 旧版 fallback 到 torch.cuda.amp.*
# ────────────────────────────────────────────────────────────

def make_grad_scaler(enabled: bool, device_type: str = "cuda"):
    """创建 GradScaler (兼容新旧 torch API)。enabled=False 返回 None。"""
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler(device_type)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def autocast(enabled: bool = True, device_type: str = "cuda"):
    """autocast 上下文 (兼容新旧 torch API)。"""
    try:
        return torch.amp.autocast(device_type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    loss: float = 0.0,
    extra: dict = None,
):
    """保存 checkpoint"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "loss": loss,
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> dict:
    """加载 checkpoint"""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt

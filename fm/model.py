"""
model.py — Flow Matching 完整模型
===================================
支持 MLP / Transformer 两种 backbone，通过配置切换。
"""

import torch
import torch.nn.functional as F

from .flow_core import flow_match_forward
from .velocity_net import VelocityNetwork, TransformerVelocityNetwork


class FlowMatchingModel(torch.nn.Module):
    """
    Flow Matching 用户特征预测模型 (Teacher)。

    支持 backbone:
    - transformer (默认): TransformerVelocityNetwork (~20-50M 参数)
    - mlp:          VelocityNetwork (~3-5M 参数)

    训练: compute_loss(x_clean, condition, time_weight_mode, time_weight_scale)
    推理: sample(condition) / predict_future_embedding(condition, delta_t)
    """

    def __init__(
        self,
        data_dim: int = 256,
        time_dim: int = 128,
        cond_dim: int = 256,
        # MLP 参数
        hidden_dim: int = 512,
        num_layers: int = 6,
        dropout: float = 0.1,
        # Transformer 参数
        backbone_type: str = "transformer",
        transformer_dim: int = 1024,
        transformer_depth: int = 8,
        transformer_heads: int = 8,
        transformer_mlp_ratio: float = 4.0,
        transformer_dropout: float = 0.1,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.backbone_type = backbone_type

        if backbone_type == "transformer":
            self.velocity_net = TransformerVelocityNetwork(
                data_dim=data_dim,
                time_dim=time_dim,
                cond_dim=cond_dim,
                transformer_dim=transformer_dim,
                depth=transformer_depth,
                heads=transformer_heads,
                mlp_ratio=transformer_mlp_ratio,
                dropout=transformer_dropout,
            )
        elif backbone_type == "mlp":
            self.velocity_net = VelocityNetwork(
                data_dim=data_dim,
                time_dim=time_dim,
                cond_dim=cond_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}, choose from transformer/mlp")

    @staticmethod
    def compute_time_weights(
        t: torch.Tensor,
        mode: str = "mid",
        scale: float = 1.0,
    ) -> torch.Tensor:
        """
        计算时间步加权函数 w(t)。

        策略:
        - none:  w(t) = 1                  (标准均匀加权)
        - mid:   w(t) = 1 + scale * 4t(1-t)  (中间高, t=0.5 时最大)
        - late:  w(t) = 1 + scale * 2t      (大 t 更高, 接近噪声端)
        - early: w(t) = 1 + scale * 2(1-t)  (小 t 更高, 接近干净端)
        """
        if mode == "none" or scale == 0.0:
            return torch.ones_like(t)

        if mode == "mid":
            w = 1.0 + scale * 4.0 * t * (1.0 - t)
        elif mode == "late":
            w = 1.0 + scale * 2.0 * t
        elif mode == "early":
            w = 1.0 + scale * 2.0 * (1.0 - t)
        else:
            raise ValueError(f"Unknown time_weight_mode: {mode}, choose from none/mid/late/early")

        return w

    def compute_loss(
        self,
        x_clean: torch.Tensor,
        condition: torch.Tensor = None,
        time_weight_mode: str = "mid",
        time_weight_scale: float = 1.0,
    ) -> dict:
        """
        Flow Matching 训练损失: w(t) * ||v_pred - v_t||^2

        Returns:
            dict with: loss, mse_raw, mse_weighted, time_weight_mean, t_mean
        """
        batch_size = x_clean.size(0)
        device = x_clean.device

        t = torch.rand(batch_size, device=device)
        noise = torch.randn_like(x_clean)
        x_t, v_t = flow_match_forward(x_clean, t, noise)
        v_pred = self.velocity_net(x_t, t, condition)

        # 逐样本 MSE
        per_sample_mse = F.mse_loss(v_pred, v_t, reduction='none').mean(dim=-1)  # (batch,)

        mse_raw = per_sample_mse.mean()

        w = self.compute_time_weights(t, mode=time_weight_mode, scale=time_weight_scale)
        mse_weighted = (w * per_sample_mse).mean()

        loss = mse_weighted

        return {
            'loss': loss,
            'mse_raw': mse_raw.item(),
            'mse_weighted': mse_weighted.item(),
            'time_weight_mean': w.mean().item(),
            't_mean': t.mean().item(),
        }

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor = None,
        num_steps: int = 100,
        method: str = 'euler',
        device: str = 'cpu',
    ) -> torch.Tensor:
        """从噪声生成预测 embedding (ODE: t=1→0)"""
        batch_size = condition.size(0) if condition is not None else 1
        x = torch.randn(batch_size, self.data_dim, device=device)
        dt = 1.0 / num_steps

        if method == 'euler':
            for step in range(num_steps):
                t_val = 1.0 - step * dt
                t = torch.full((batch_size,), t_val, device=device)
                v = self.velocity_net(x, t, condition)
                x = x - v * dt
        elif method == 'rk4':
            for step in range(num_steps):
                t_val = 1.0 - step * dt
                t = torch.full((batch_size,), t_val, device=device)
                k1 = self.velocity_net(x, t, condition)
                k2 = self.velocity_net(x - 0.5 * dt * k1, t - 0.5 * dt, condition)
                k3 = self.velocity_net(x - 0.5 * dt * k2, t - 0.5 * dt, condition)
                k4 = self.velocity_net(x - dt * k3, t - dt, condition)
                x = x - (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        else:
            raise ValueError(f"Unknown ODE method: {method}")
        return x

    @torch.no_grad()
    def predict_future_embedding(
        self,
        condition: torch.Tensor,
        delta_t: float,
        num_steps: int = 100,
        method: str = 'euler',
    ) -> torch.Tensor:
        """预测用户在 delta_t 时间后的 embedding"""
        batch_size = condition.size(0)
        device = condition.device
        condition = condition.to(device)

        noise = torch.randn_like(condition)
        x_t = (1.0 - delta_t) * condition + delta_t * noise

        dt = delta_t / num_steps
        for step in range(num_steps):
            t_val = delta_t - step * dt
            t = torch.full((batch_size,), t_val, device=device)
            v = self.velocity_net(x_t, t, condition)
            x_t = x_t - v * dt
        return x_t

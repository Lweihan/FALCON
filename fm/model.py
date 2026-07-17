"""
model.py — Flow Matching 完整模型
===================================
"""

import torch
import torch.nn.functional as F

from .flow_core import flow_match_forward
from .velocity_net import VelocityNetwork


class FlowMatchingModel(torch.nn.Module):
    """
    Flow Matching 用户特征预测模型 (Teacher)。

    训练: compute_loss(x_clean, condition)
    推理: sample(condition) / predict_future_embedding(condition, delta_t)
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
        self.velocity_net = VelocityNetwork(
            data_dim=data_dim, time_dim=time_dim, cond_dim=cond_dim,
            hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,
        )

    def compute_loss(
        self,
        x_clean: torch.Tensor,
        condition: torch.Tensor = None,
    ) -> dict:
        """Flow Matching 训练损失: ||v_pred - v_t||^2"""
        batch_size = x_clean.size(0)
        device = x_clean.device

        t = torch.rand(batch_size, device=device)
        noise = torch.randn_like(x_clean)
        x_t, v_t = flow_match_forward(x_clean, t, noise)
        v_pred = self.velocity_net(x_t, t, condition)
        loss = F.mse_loss(v_pred, v_t)
        return {'loss': loss, 'mse': loss.item()}

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
        """
        预测用户在 delta_t 时间后的 embedding。

        delta_t ∈ [0,1]: 0=现在, 1=最远未来
        """
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

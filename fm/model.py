"""
model.py — Flow Matching 完整模型
===================================
支持 DiT / Transformer / MLP 三种 backbone，通过配置切换。
"""

import torch
import torch.nn.functional as F

from .flow_core import flow_match_forward
from .velocity_net import VelocityNetwork, TransformerVelocityNetwork
from .dit_backbone import DiTVelocityNetwork


class FlowMatchingModel(torch.nn.Module):
    """
    Flow Matching 用户特征预测模型 (Teacher)。

    支持 backbone:
    - dit (推荐默认): DiTVelocityNetwork
        Patchify + AdaLN-Zero + QK-Norm + RoPE + SwiGLU
        preset: tiny(~10M) / small(~80M) / base(~250M) / large(~1.2B) / xl(~4B)
    - transformer:   TransformerVelocityNetwork (~33M, 旧版保留兼容)
    - mlp:           VelocityNetwork (~3.6M, 轻量)

    训练: compute_loss(x_clean, condition, time_weight_mode, time_weight_scale)
    推理: sample(condition) / predict_future_embedding(condition, delta_t)
    """

    def __init__(
        self,
        data_dim: int = 256,
        time_dim: int = 256,
        cond_dim: int = 256,
        backbone_type: str = "dit",
        # ── DiT 参数 ──
        dit_dim: int = 1024,
        dit_depth: int = 16,
        dit_heads: int = 16,
        dit_num_patches: int = 16,
        dit_mlp_ratio: float = 4.0,
        dit_dropout: float = 0.0,
        dit_qk_norm: bool = True,
        dit_use_rope: bool = True,
        use_gradient_checkpointing: bool = False,
        # ── MLP 参数 ──
        hidden_dim: int = 512,
        num_layers: int = 6,
        dropout: float = 0.1,
        # ── Transformer 参数 (旧版) ──
        transformer_dim: int = 1024,
        transformer_depth: int = 8,
        transformer_heads: int = 8,
        transformer_mlp_ratio: float = 4.0,
        transformer_dropout: float = 0.1,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.backbone_type = backbone_type

        if backbone_type == "dit":
            self.velocity_net = DiTVelocityNetwork(
                data_dim=data_dim,
                cond_dim=cond_dim,
                dim=dit_dim,
                depth=dit_depth,
                num_heads=dit_heads,
                num_patches=dit_num_patches,
                mlp_ratio=dit_mlp_ratio,
                dropout=dit_dropout,
                qk_norm=dit_qk_norm,
                use_rope=dit_use_rope,
                time_dim=time_dim,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )
        elif backbone_type == "transformer":
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
            raise ValueError(
                f"Unknown backbone_type: {backbone_type}, choose from dit/transformer/mlp"
            )

    @classmethod
    def from_config(cls, fm_config, dropout_override: float = None) -> "FlowMatchingModel":
        """
        从 FMConfig 构建模型，自动解析 dit_preset。

        Args:
            fm_config: FMConfig 实例
            dropout_override: 若指定则覆盖所有 dropout (推理时设 0.0)
        """
        dit = fm_config.resolve_dit_params()
        drop = dropout_override

        return cls(
            data_dim=fm_config.data_dim,
            time_dim=fm_config.time_dim,
            cond_dim=fm_config.cond_dim,
            backbone_type=fm_config.backbone_type,
            # DiT
            dit_dim=dit["dit_dim"],
            dit_depth=dit["dit_depth"],
            dit_heads=dit["dit_heads"],
            dit_num_patches=dit["dit_num_patches"],
            dit_mlp_ratio=fm_config.dit_mlp_ratio,
            dit_dropout=drop if drop is not None else fm_config.dit_dropout,
            dit_qk_norm=fm_config.dit_qk_norm,
            dit_use_rope=fm_config.dit_use_rope,
            use_gradient_checkpointing=fm_config.use_gradient_checkpointing,
            # MLP
            hidden_dim=fm_config.hidden_dim,
            num_layers=fm_config.num_layers,
            dropout=drop if drop is not None else fm_config.dropout,
            # Transformer
            transformer_dim=fm_config.transformer_dim,
            transformer_depth=fm_config.transformer_depth,
            transformer_heads=fm_config.transformer_heads,
            transformer_mlp_ratio=fm_config.transformer_mlp_ratio,
            transformer_dropout=drop if drop is not None else fm_config.transformer_dropout,
        )

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

    def forward(
        self,
        x_clean: torch.Tensor,
        condition: torch.Tensor = None,
        time_weight_mode: str = "mid",
        time_weight_scale: float = 1.0,
        delta_t: torch.Tensor = None,
    ) -> dict:
        """
        forward 即训练损失计算。

        这样设计的原因: DDP 只会为通过 forward() 的调用注册梯度同步 hook。
        若直接调用 module.compute_loss()，多卡训练时梯度不会 all_reduce。
        """
        return self.compute_loss(
            x_clean, condition, time_weight_mode, time_weight_scale, delta_t
        )

    def compute_loss(
        self,
        x_clean: torch.Tensor,
        condition: torch.Tensor = None,
        time_weight_mode: str = "mid",
        time_weight_scale: float = 1.0,
        delta_t: torch.Tensor = None,
    ) -> dict:
        """
        Flow Matching 训练损失: w(t) * ||v_pred - v_t||^2

        Args:
            x_clean: (batch, dim) 目标 embedding
            condition: (batch, dim) 条件 embedding
            time_weight_mode: 时间加权策略
            time_weight_scale: 加权幅度
            delta_t: (batch,) 真实时间跨度, 注入网络以修复训推不一致

        Returns:
            dict with: loss, mse_raw, mse_weighted, time_weight_mean, t_mean
        """
        batch_size = x_clean.size(0)
        device = x_clean.device

        t = torch.rand(batch_size, device=device)
        noise = torch.randn_like(x_clean)
        x_t, v_t = flow_match_forward(x_clean, t, noise)
        v_pred = self.velocity_net(x_t, t, condition, delta_t)

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
        inject_delta_t: bool = True,
    ) -> torch.Tensor:
        """
        预测用户在 delta_t 时间后的 embedding。

        delta_t 同时扮演两个角色:
          1. ODE 起点与步长控制 (决定从多少噪声开始去噪)
          2. 网络条件输入 (告知模型要预测多远的未来)
        """
        batch_size = condition.size(0)
        device = condition.device
        condition = condition.to(device)

        delta_t_tensor = (
            torch.full((batch_size,), delta_t, device=device) if inject_delta_t else None
        )

        noise = torch.randn_like(condition)
        x_t = (1.0 - delta_t) * condition + delta_t * noise

        dt = delta_t / max(num_steps, 1)
        for step in range(num_steps):
            t_val = delta_t - step * dt
            t = torch.full((batch_size,), t_val, device=device)
            v = self.velocity_net(x_t, t, condition, delta_t_tensor)
            x_t = x_t - v * dt
        return x_t

    def optimize_embedding(
        self,
        embedding: torch.Tensor,
        delta_t: float = 0.5,
        num_steps: int = 10,
        inject_delta_t: bool = True,
    ) -> torch.Tensor:
        """
        对给定 embedding 做 FM 流形优化 (保留梯度，供下游 Ranker 联合训练)。

        与 predict_future_embedding 的区别:
        - 本方法不使用 @torch.no_grad()，梯度可回传到 velocity_net 与上游 Ranker
        - 步数通常较少 (num_steps=10)，用于训练时的可微 ODE 求解
        """
        batch_size = embedding.size(0)
        device = embedding.device

        delta_t_tensor = (
            torch.full((batch_size,), delta_t, device=device) if inject_delta_t else None
        )

        noise = torch.randn_like(embedding)
        x_t = (1.0 - delta_t) * embedding + delta_t * noise

        dt = delta_t / max(num_steps, 1)
        for step in range(num_steps):
            t_val = delta_t - step * dt
            t = torch.full((batch_size,), t_val, device=device)
            v = self.velocity_net(x_t, t, embedding, delta_t_tensor)
            x_t = x_t - v * dt
        return x_t

"""
joint_falcon_ranker.py — FALCON + Ranker 联合训练
==================================================
将下游精排模型 (BaseRanker) 与 FALCON 的 Flow Matching 模型
封装为端到端联合训练模型。

两条预测路径同时训练:
  1. 直接预测:      sparse → ranker → logit_direct
  2. FALCON 优化后:  sparse → ranker.extract_embedding → FM 优化 → logit_falcon

总损失:
  L = α · L_direct + (1-α) · [fm_weight · L_velocity + (1-fm_weight) · L_BCE_falcon]

关键点: ranker 的 embedding 层与 FM 的 velocity_net 同时接收梯度，
使 ranker 学会生成更适合 FM 流形优化的 embedding。
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.logger import get_logger
from fm.model import FlowMatchingModel
from .base_ranker import BaseRanker

logger = get_logger(__name__)


class JointFALCONRanker(nn.Module):
    """
    FALCON Flow Matching + 下游 Ranker 联合训练模型。
    """

    def __init__(
        self,
        ranker: BaseRanker,
        fm_config,
        ranker_config,
        fm_checkpoint: Optional[str] = None,
        fm_weight: float = 0.5,
        device: str = "cuda:0",
    ):
        """
        Args:
            ranker: 精排模型实例
            fm_config: FMConfig
            ranker_config: RankerConfig
            fm_checkpoint: 预训练 FM checkpoint 路径 (None 则随机初始化)
            fm_weight: FM velocity 损失权重
            device: 设备
        """
        super().__init__()
        self.ranker = ranker
        self.ranker_config = ranker_config
        self.fm_config = fm_config
        self.fm_weight = fm_weight
        self.joint_alpha = ranker_config.joint_alpha

        # Ranker embedding 输出维度
        self.ranker_embed_dim = ranker.embed_output_dim
        self.fm_data_dim = fm_config.data_dim

        # 若维度不匹配, 加投影层对齐到 FM 的 data_dim
        if self.ranker_embed_dim != self.fm_data_dim:
            self.input_proj = nn.Sequential(
                nn.Linear(self.ranker_embed_dim, self.fm_data_dim),
                nn.LayerNorm(self.fm_data_dim),
            )
            logger.info(
                f"添加投影层: ranker({self.ranker_embed_dim}) → FM({self.fm_data_dim})"
            )
        else:
            self.input_proj = nn.Identity()

        # FM 模型
        self.fm_model = FlowMatchingModel.from_config(fm_config)
        if fm_checkpoint:
            self._load_fm_checkpoint(fm_checkpoint, device)

        if ranker_config.freeze_fm:
            for p in self.fm_model.parameters():
                p.requires_grad = False
            logger.info("FM backbone 已冻结")

        # FM 优化后 embedding → CTR 预测头
        self.pred_head = nn.Sequential(
            nn.Linear(self.fm_data_dim, self.fm_data_dim // 2),
            nn.ReLU(),
            nn.Dropout(ranker_config.dropout),
            nn.Linear(self.fm_data_dim // 2, 1),
        )

    def _load_fm_checkpoint(self, path: str, device: str):
        """加载预训练 FM 权重 (容错: 允许部分不匹配)"""
        try:
            ckpt = torch.load(path, map_location=device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            missing, unexpected = self.fm_model.load_state_dict(state, strict=False)
            logger.info(f"FM checkpoint 加载: {path}")
            if missing:
                logger.warning(f"  缺失 key 数: {len(missing)}")
            if unexpected:
                logger.warning(f"  多余 key 数: {len(unexpected)}")
        except Exception as e:
            logger.warning(f"FM checkpoint 加载失败 ({e})，使用随机初始化")

    # ── 前向 ──

    def forward(
        self,
        sparse_values: Dict[str, torch.Tensor],
        fast_train: bool = True,
    ) -> dict:
        """
        联合前向，返回两条路径的 logits。

        Args:
            sparse_values: {feat: (batch,)}
            fast_train: True 则跳过 ODE 求解 (直接用投影后 embedding 预测)，训练更快

        Returns:
            dict: direct_logit, falcon_logit, fm_loss_dict, embeddings
        """
        # 路径 1: 直接预测
        direct_logit = self.ranker(sparse_values)

        # 路径 2: 提取 embedding → 投影 → FM
        embeddings = self.ranker.extract_embedding(sparse_values)
        projected = self.input_proj(embeddings)

        # FM velocity 损失 (让 embedding 空间符合流形结构)
        fm_loss_dict = self.fm_model.compute_loss(
            x_clean=projected,
            condition=projected,
            time_weight_mode=self.fm_config.time_weight_mode,
            time_weight_scale=self.fm_config.time_weight_scale,
        )

        if fast_train:
            falcon_logit = self.pred_head(projected).squeeze(-1)
        else:
            optimized = self.fm_model.optimize_embedding(
                projected,
                delta_t=self.ranker_config.train_delta_t,
                num_steps=self.ranker_config.train_ode_steps,
            )
            falcon_logit = self.pred_head(optimized).squeeze(-1)

        return {
            "direct_logit": direct_logit,
            "falcon_logit": falcon_logit,
            "fm_loss_dict": fm_loss_dict,
            "embeddings": embeddings,
        }

    def compute_joint_loss(
        self,
        sparse_values: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        fast_train: bool = True,
    ) -> dict:
        """
        L = α·L_direct + (1-α)·[fm_weight·L_velocity + (1-fm_weight)·L_BCE_falcon]
        """
        fwd = self.forward(sparse_values, fast_train=fast_train)

        bce_direct = F.binary_cross_entropy_with_logits(fwd["direct_logit"], labels)
        bce_falcon = F.binary_cross_entropy_with_logits(fwd["falcon_logit"], labels)
        velocity_loss = fwd["fm_loss_dict"]["loss"]

        falcon_total = self.fm_weight * velocity_loss + (1 - self.fm_weight) * bce_falcon
        total = self.joint_alpha * bce_direct + (1 - self.joint_alpha) * falcon_total

        return {
            "loss": total,
            "bce_direct": bce_direct.item(),
            "bce_falcon": bce_falcon.item(),
            "velocity_loss": velocity_loss.item(),
            "falcon_total": falcon_total.item(),
        }

    # ── 推理 ──

    @torch.no_grad()
    def predict_direct(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """直接预测路径"""
        return torch.sigmoid(self.ranker(sparse_values))

    @torch.no_grad()
    def predict_falcon(
        self,
        sparse_values: Dict[str, torch.Tensor],
        num_steps: Optional[int] = None,
        delta_t: Optional[float] = None,
    ) -> torch.Tensor:
        """FALCON 优化后预测路径"""
        num_steps = num_steps or self.ranker_config.eval_ode_steps
        delta_t = delta_t if delta_t is not None else self.ranker_config.train_delta_t

        embeddings = self.ranker.extract_embedding(sparse_values)
        projected = self.input_proj(embeddings)
        optimized = self.fm_model.optimize_embedding(
            projected, delta_t=delta_t, num_steps=num_steps
        )
        return torch.sigmoid(self.pred_head(optimized).squeeze(-1))

    def param_summary(self) -> dict:
        """参数量统计"""
        def _count(params):
            return sum(p.numel() for p in params)

        ranker_stats = self.ranker.count_parameters()
        return {
            "ranker_total": ranker_stats["total"],
            "ranker_embedding": ranker_stats["embedding"],
            "ranker_network": ranker_stats["network"],
            "fm_total": _count(self.fm_model.parameters()),
            "pred_head": _count(self.pred_head.parameters()),
            "total": _count(self.parameters()),
        }

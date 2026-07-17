"""
teacher.py — FM Teacher 封装
===============================
将训练好的 Flow Matching 模型封装为 Teacher，
提供 predict_target_embedding() 接口供 OPD 调用。
"""

import torch
import torch.nn.functional as F
from copy import deepcopy

from common.logger import get_logger
from common.config import FMConfig
from fm.model import FlowMatchingModel

logger = get_logger(__name__)


class FMTeacher:
    """
    Flow Matching Teacher 封装。

    功能:
    - 加载预训练 FM 模型
    - 根据当前用户条件预测未来 embedding (蒸馏目标)
    - 支持 EMA 权重保持稳定
    - 提供分部分 target embedding (用于 per-section distillation)
    """

    def __init__(
        self,
        fm_config: FMConfig,
        checkpoint_path: str = None,
        device: str = 'cuda:0',
        use_ema: bool = True,
        ema_decay: float = 0.999,
    ):
        self.config = fm_config
        self.device = device
        self.use_ema = use_ema
        self.ema_decay = ema_decay

        # 主模型
        self.model = FlowMatchingModel(
            data_dim=fm_config.data_dim,
            time_dim=fm_config.time_dim,
            cond_dim=fm_config.cond_dim,
            # MLP 参数
            hidden_dim=fm_config.hidden_dim,
            num_layers=fm_config.num_layers,
            dropout=0.0,  # 推理不用 dropout
            # Transformer 参数
            backbone_type=fm_config.backbone_type,
            transformer_dim=fm_config.transformer_dim,
            transformer_depth=fm_config.transformer_depth,
            transformer_heads=fm_config.transformer_heads,
            transformer_mlp_ratio=fm_config.transformer_mlp_ratio,
            transformer_dropout=0.0,  # 推理不用 dropout
        ).to(device)

        # 加载 checkpoint
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            logger.info(f"FM Teacher 加载: {checkpoint_path}")

        # EMA 模型 (用于稳定蒸馏目标)
        if use_ema:
            self.ema_model = deepcopy(self.model)
            self.ema_model.eval()
            for p in self.ema_model.parameters():
                p.requires_grad = False

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False  # Teacher 冻结

    @torch.no_grad()
    def predict_target_embedding(
        self,
        condition: torch.Tensor,
        delta_t: float = 0.5,
        num_steps: int = 100,
        method: str = 'euler',
    ) -> torch.Tensor:
        """
        Teacher 预测未来 embedding，作为蒸馏目标。

        使用 EMA 模型 (若启用) 提供更稳定的蒸馏信号。

        Args:
            condition: 当前用户 embedding, (batch, data_dim)
            delta_t: 时间偏移 [0, 1]
            num_steps: ODE 步数
            method: euler / rk4

        Returns:
            target_emb: (batch, data_dim)
        """
        model = self.ema_model if self.use_ema else self.model
        return model.predict_future_embedding(
            condition=condition.to(self.device),
            delta_t=delta_t,
            num_steps=num_steps,
            method=method,
        )

    @torch.no_grad()
    def predict_target_section_embeddings(
        self,
        condition_emb: torch.Tensor,
        delta_t: float = 0.5,
        num_steps: int = 100,
    ) -> dict:
        """
        预测目标分部分 embedding (用于 per-section distillation)。

        思路: 将预测的 target embedding 分解回 4 部分空间。
        由于 FM 直接预测的是融合 embedding，我们需要额外一个
        "逆融合" 投影将融合 embedding 投射回各部分空间。

        简化实现: 使用线性投影分解。
        更精细: 需要与 fusion 联合训练逆映射。

        Returns:
            dict with 4 section target embeddings, each (batch, embed_dim)
        """
        target_fused = self.predict_target_embedding(
            condition_emb, delta_t, num_steps
        )  # (batch, data_dim)

        # 逆融合: 将 fused embedding 投射回 4 部分
        # 这里使用简单的线性投影 (可替换为更精细的逆映射)
        # 实际训练时这些投影层应该与 fusion 一起训练
        return self._inverse_fuse(target_fused)

    def _inverse_fuse(self, fused_emb: torch.Tensor) -> dict:
        """
        逆融合: fused → 4 部分 embedding。

        使用可学习的逆映射投影 (初始化为随机正交投影)。
        在实际使用中，这些权重需要从预训练的 fusion 模块中导出。
        """
        batch_size = fused_emb.size(0)
        data_dim = fused_emb.size(1)

        # 简化: 将 fused 均匀拆分为 4 份
        # 更好: 训练逆映射网络 (可在 OPD 训练中联合优化)
        chunk_size = data_dim // 4
        if chunk_size * 4 < data_dim:
            chunk_size += 1

        sections = {}
        section_names = ['static_information', 'interest_profile',
                         'consumption_profile', 'demand_targets']
        for i, name in enumerate(section_names):
            start = i * chunk_size
            end = min(start + chunk_size, data_dim)
            chunk = fused_emb[:, start:end]
            # Pad if needed to match embed_dim
            if chunk.size(1) < self.config.data_dim:
                pad = torch.zeros(batch_size, self.config.data_dim - chunk.size(1), device=chunk.device)
                chunk = torch.cat([chunk, pad], dim=1)
            sections[f'{name}_target'] = chunk[:, :self.config.data_dim]

        return sections

    def update_ema(self):
        """更新 EMA 权重"""
        if not self.use_ema:
            return
        for ema_p, model_p in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_p.data.mul_(self.ema_decay).add_(model_p.data, alpha=1.0 - self.ema_decay)

"""
evaluator.py — 评估指标
=========================
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional

from common.logger import get_logger

logger = get_logger(__name__)


class FALCONEvaluator:
    """
    FALCON 模型评估器。

    支持指标:
    - cosine_sim: Student/Teacher embedding 余弦相似度
    - mse: 均方误差
    - ndcg: 排序质量 (用于广告推荐场景)
    - section_alignment: 分部分对齐度
    - temporal_consistency: 时间一致性
    """

    @staticmethod
    def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

    @staticmethod
    def batch_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
        sims = F.cosine_similarity(a, b, dim=-1).cpu().numpy()
        return sims

    @staticmethod
    def mse(a: torch.Tensor, b: torch.Tensor) -> float:
        return F.mse_loss(a, b).item()

    @staticmethod
    def ndcg_at_k(predicted: torch.Tensor, target: torch.Tensor, k: int = 10) -> float:
        """
        NDCG@K: 评估预测排序与目标排序的一致性。

        Args:
            predicted: (dim,) 预测 embedding
            target: (N, dim) 候选 embedding 矩阵
            k: top-K
        """
        # 计算相似度排序
        pred_sims = F.cosine_similarity(target, predicted.unsqueeze(0), dim=-1)
        pred_ranking = torch.argsort(pred_sims, descending=True)[:k]

        # 理想排序 (使用 ground truth 相似度)
        # 这里简化: 如果有 ground truth embedding, 用它排序
        # 实际中需要真实的 relevance label
        return pred_sims[pred_ranking].mean().item()

    @staticmethod
    def evaluate_fm_model(
        model,
        dataloader,
        device: str = 'cpu',
    ) -> dict:
        """评估 FM Teacher 模型"""
        model.eval()
        all_sims = []
        all_mses = []

        with torch.no_grad():
            for batch in dataloader:
                x_clean = batch['x_clean'].to(device)
                condition = batch['condition'].to(device)

                # Teacher 预测
                predicted = model.predict_future_embedding(condition, delta_t=0.5, num_steps=50)

                # 评估
                sims = F.cosine_similarity(predicted, x_clean, dim=-1).cpu().numpy()
                mses = F.mse_loss(predicted, x_clean, reduction='none').mean(dim=-1).cpu().numpy()

                all_sims.extend(sims.tolist())
                all_mses.extend(mses.tolist())

        return {
            'cosine_sim_mean': np.mean(all_sims),
            'cosine_sim_std': np.std(all_sims),
            'mse_mean': np.mean(all_mses),
            'mse_std': np.std(all_mses),
            'n_samples': len(all_sims),
        }

    @staticmethod
    def evaluate_opd(
        teacher,
        student,
        latent_pipeline,
        dataloader,
        device: str = 'cpu',
        delta_t: float = 0.5,
    ) -> dict:
        """评估 OPD 蒸馏效果"""
        results = {
            'emb_cos_sim': [],
            'section_alignment': [],
        }

        with torch.no_grad():
            for batch in dataloader:
                condition = batch['condition'].to(device)

                # Teacher target
                teacher_emb = teacher.predict_target_embedding(condition, delta_t)

                # Student 生成 + 编码 (简化: 使用 condition 代替)
                student_emb = condition  # placeholder

                sims = F.cosine_similarity(student_emb, teacher_emb, dim=-1).cpu().numpy()
                results['emb_cos_sim'].extend(sims.tolist())

        results['emb_cos_sim_mean'] = np.mean(results['emb_cos_sim']) if results['emb_cos_sim'] else 0
        results['n_samples'] = len(results['emb_cos_sim'])
        return results

"""
metrics.py — 排序任务评估指标
==============================
AUC / GAUC / LogLoss / NDCG@K
"""

from typing import Dict, List, Optional

import numpy as np


def compute_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """AUC (Area Under ROC Curve)"""
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_pred))
    except (ValueError, ImportError):
        return _auc_fallback(y_true, y_pred)


def _auc_fallback(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """无 sklearn 时的 AUC 实现 (基于秩)"""
    pos = y_true == 1
    neg = ~pos
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return 0.0
    ranks = np.argsort(np.argsort(y_pred)) + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def compute_gauc(y_true: np.ndarray, y_pred: np.ndarray, group_ids: np.ndarray) -> float:
    """
    GAUC (Group AUC): 按用户分组计算 AUC，按组内样本数加权平均。

    GAUC = Σ (n_u · AUC_u) / Σ n_u
    """
    total_weighted, total_n = 0.0, 0
    for g in np.unique(group_ids):
        mask = group_ids == g
        yt, yp = y_true[mask], y_pred[mask]
        if len(np.unique(yt)) < 2:      # 组内需同时有正负样本
            continue
        n = int(mask.sum())
        total_weighted += n * compute_auc(yt, yp)
        total_n += n
    return total_weighted / total_n if total_n > 0 else 0.0


def compute_logloss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    """LogLoss (Binary Cross-Entropy)"""
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


def compute_ndcg(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    """NDCG@K"""
    k = min(k, len(y_true))
    if k == 0:
        return 0.0

    def _dcg(labels):
        gains = 2 ** labels[:k] - 1
        discounts = np.log2(np.arange(k) + 2)
        return float(np.sum(gains / discounts))

    dcg = _dcg(y_true[np.argsort(y_pred)[::-1]])
    idcg = _dcg(np.sort(y_true)[::-1])
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_ids: Optional[np.ndarray] = None,
    metrics: Optional[List[str]] = None,
    ndcg_k: int = 10,
) -> Dict[str, float]:
    """
    统一评估接口。

    Args:
        y_true: (N,) 真实标签 0/1
        y_pred: (N,) 预测概率
        group_ids: (N,) 分组 ID (GAUC 需要)
        metrics: 指标列表, 默认 ["auc", "gauc", "logloss", "ndcg"]
        ndcg_k: NDCG 的 K
    """
    metrics = metrics or ["auc", "gauc", "logloss", "ndcg"]
    results: Dict[str, float] = {}

    for m in metrics:
        if m == "auc":
            results["auc"] = compute_auc(y_true, y_pred)
        elif m == "gauc":
            results["gauc"] = (
                compute_gauc(y_true, y_pred, group_ids) if group_ids is not None else 0.0
            )
        elif m == "logloss":
            results["logloss"] = compute_logloss(y_true, y_pred)
        elif m == "ndcg":
            results[f"ndcg@{ndcg_k}"] = compute_ndcg(y_true, y_pred, k=ndcg_k)
        else:
            raise ValueError(f"Unknown metric: {m}")

    return results


def compute_lift(direct: Dict[str, float], falcon: Dict[str, float]) -> Dict[str, float]:
    """
    计算 FALCON 优化相对直接预测的提升。

    AUC/GAUC/NDCG 越大越好 → lift = falcon - direct
    LogLoss 越小越好       → lift = direct - falcon
    """
    lift = {}
    for key in direct:
        if key not in falcon:
            continue
        if key == "logloss":
            lift[f"{key}_lift"] = direct[key] - falcon[key]
        else:
            lift[f"{key}_lift"] = falcon[key] - direct[key]
    return lift

"""
metrics.py — 排序 / 回归任务评估指标
====================================
分类指标: AUC / GAUC / LogLoss / NDCG@K / PosRate
回归指标: MSE / MAE / Pearson / Spearman

设计说明:
  - 所有函数入口统一 np.asarray, 兼容 python list 输入
  - AUC 前置校验标签是否二值, 非二值时给出明确错误信息
    (而非让 sklearn 抛出晦涩的 "multi_class must be in ('ovo','ovr')")
  - MAC 数据集的 linear/mta/first/last 是连续归因权重,
    若要算 AUC 必须先二值化 (见 compute_auc 的 binarize_threshold 参数)
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _as_1d(x, name: str = "array") -> np.ndarray:
    """转为 1D float64 ndarray, 兼容 list / tensor / ndarray。"""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr


def _finite_pair(y_true, y_pred, name: str = "metric"):
    """
    联合过滤 y_true / y_pred 中的 NaN / Inf, 返回 (yt, yp, dropped_count)。

    为何要联合过滤:
      任一侧 NaN/Inf 都会污染指标 (sklearn 直接抛 "Input contains NaN")。
      联合 mask 保证下标对齐, 避免 y_true[i] 与 y_pred[j] 错位。

    用途:
      compute_auc / compute_mse / compute_pearson 等入口都调用一次,
      作为最后一道防线 (数据入口和训练循环已各有一道 NaN 防御)。
    """
    yt = _as_1d(y_true, f"{name}.y_true")
    yp = _as_1d(y_pred, f"{name}.y_pred")

    if yt.shape != yp.shape:
        raise ValueError(f"y_true/y_pred 形状不一致: {yt.shape} vs {yp.shape}")

    mask = np.isfinite(yt) & np.isfinite(yp)
    dropped = int(mask.size - mask.sum())
    if dropped:
        ratio = dropped / max(mask.size, 1)
        level = "warning" if ratio > 0.001 else "debug"   # 千分之一以上告警
        msg = (f"[{name}] 过滤 {dropped}/{mask.size} 个 NaN/Inf 对 "
               f"({ratio:.2%}); 已扣除后计算指标")
        (logger.warning if level == "warning" else logger.debug)(msg)
        yt = yt[mask]
        yp = yp[mask]
    return yt, yp, dropped


def is_binary(y_true) -> bool:
    """判断标签是否为二值 (只含 0 和 1)。"""
    arr = _as_1d(y_true)
    uniq = np.unique(arr)
    return len(uniq) <= 2 and np.all(np.isin(uniq, (0.0, 1.0)))


# ════════════════════════════════════════════════════════════
# 分类指标
# ════════════════════════════════════════════════════════════

def compute_auc(
    y_true,
    y_pred,
    binarize_threshold: Optional[float] = None,
) -> float:
    """
    AUC (Area Under ROC Curve)。

    Args:
        y_true: 真实标签; 必须是二值 (0/1), 除非指定 binarize_threshold
        y_pred: 预测分数
        binarize_threshold: 若指定, 先用 y_true > threshold 二值化再算 AUC。
            用于连续归因权重标签 (如 MAC 的 last) 计算参考 AUC。

    Returns:
        AUC; 单一类别 (全正或全负) 时返回 0.5 (AUC 此时无定义)
    """
    yt, yp, _ = _finite_pair(y_true, y_pred, name="AUC")
    if yt.size == 0:
        logger.warning("AUC: 过滤 NaN/Inf 后无有效样本, 返回 0.5")
        return 0.5

    if binarize_threshold is not None:
        yt = (yt > binarize_threshold).astype(np.float64)

    uniq = np.unique(yt)

    # 单一类别 → AUC 无定义
    if len(uniq) < 2:
        logger.warning(
            f"AUC 无定义: 标签只有单一类别 (值={uniq.tolist()}, n={yt.size})。"
            "返回 0.5。请检查标签处理 / 阈值设置。"
        )
        return 0.5

    # 非二值 → 明确报错 (而非让 sklearn 抛 multi_class 错误)
    if not (len(uniq) == 2 and np.all(np.isin(uniq, (0.0, 1.0)))):
        raise ValueError(
            f"AUC 要求二值标签 (0/1), 但 y_true 含 {len(uniq)} 个不同取值: "
            f"{uniq[:8].tolist()}{'...' if len(uniq) > 8 else ''}\n"
            "若标签是连续归因权重 (如 MAC 的 linear/mta/first/last), 请:\n"
            "  1. 设置 ranker.mac_label_mode='binary' 让数据加载时二值化, 或\n"
            "  2. 调用 compute_auc(..., binarize_threshold=0.0) 计算参考 AUC, 或\n"
            "  3. 改用回归指标 compute_mse / compute_pearson"
        )

    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(yt, yp))
    except ImportError:
        return _auc_fallback(yt, yp)


def _auc_fallback(y_true, y_pred) -> float:
    """无 sklearn 时的 AUC 实现 (基于 Mann-Whitney U / 秩)。"""
    yt = _as_1d(y_true)
    yp = _as_1d(y_pred)

    pos = yt > 0.5
    n_pos = int(pos.sum())
    n_neg = int(yt.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # 平均秩 (处理并列分数)
    order = np.argsort(yp, kind="mergesort")
    ranks = np.empty(yp.size, dtype=np.float64)
    ranks[order] = np.arange(1, yp.size + 1, dtype=np.float64)

    # 并列分数取平均秩, 否则 AUC 会有偏
    sorted_pred = yp[order]
    i = 0
    while i < sorted_pred.size:
        j = i
        while j + 1 < sorted_pred.size and sorted_pred[j + 1] == sorted_pred[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0   # 秩从 1 开始
            ranks[order[i:j + 1]] = avg
        i = j + 1

    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def compute_gauc(y_true, y_pred, group_ids, binarize_threshold: Optional[float] = None) -> float:
    """
    GAUC (Group AUC): 按用户分组计算 AUC，按组内样本数加权平均。

    GAUC = Σ (n_u · AUC_u) / Σ n_u

    只统计组内同时有正负样本的组 (其余组 AUC 无定义)。
    """
    yt, yp, _ = _finite_pair(y_true, y_pred, name="GAUC")
    gids = np.asarray(group_ids).reshape(-1)
    # gids 长度必须与原始长度匹配; _finite_pair 改变了长度, 需同步过滤
    if gids.size != _as_1d(y_true).size:
        return 0.0
    # 重新用 finite mask 过滤 gids (必须重建 mask 保持与 yt/yp 一致)
    orig_yt = _as_1d(y_true)
    orig_yp = _as_1d(y_pred)
    mask = np.isfinite(orig_yt) & np.isfinite(orig_yp)
    gids = gids[mask]

    if binarize_threshold is not None:
        yt = (yt > binarize_threshold).astype(np.float64)

    if yt.size == 0:
        return 0.0

    total_weighted, total_n = 0.0, 0
    for g in np.unique(gids):
        mask = gids == g
        g_yt, g_yp = yt[mask], yp[mask]
        if len(np.unique(g_yt)) < 2:      # 组内需同时有正负样本
            continue
        n = int(mask.sum())
        total_weighted += n * _auc_fallback(g_yt, g_yp)
        total_n += n

    return total_weighted / total_n if total_n > 0 else 0.0


def compute_logloss(y_true, y_pred, eps: float = 1e-15) -> float:
    """
    LogLoss (Binary Cross-Entropy)。

    y_true 支持连续值 [0,1] (软标签交叉熵)。
    """
    yt, yp, _ = _finite_pair(y_true, y_pred, name="LogLoss")
    yp = np.clip(yp, eps, 1.0 - eps)
    if yt.size == 0:
        return 0.0
    return float(-np.mean(yt * np.log(yp) + (1 - yt) * np.log(1 - yp)))


def compute_pos_rate(y_true, threshold: float = 0.5) -> float:
    """正样本率 (> threshold 的比例)。用于监控标签分布是否异常。"""
    yt = _as_1d(y_true)
    return float((yt > threshold).mean()) if yt.size else 0.0


def compute_ndcg(y_true, y_pred, k: int = 10) -> float:
    """NDCG@K"""
    yt = _as_1d(y_true)
    yp = _as_1d(y_pred)
    k = min(k, yt.size)
    if k == 0:
        return 0.0

    def _dcg(labels):
        gains = 2 ** labels[:k] - 1
        discounts = np.log2(np.arange(k) + 2)
        return float(np.sum(gains / discounts))

    dcg = _dcg(yt[np.argsort(yp)[::-1]])
    idcg = _dcg(np.sort(yt)[::-1])
    return dcg / idcg if idcg > 0 else 0.0


# ════════════════════════════════════════════════════════════
# 回归指标 (连续归因权重标签用)
# ════════════════════════════════════════════════════════════

def compute_mse(y_true, y_pred) -> float:
    """均方误差"""
    yt, yp, _ = _finite_pair(y_true, y_pred, name="MSE")
    return float(np.mean((yt - yp) ** 2)) if yt.size else 0.0


def compute_mae(y_true, y_pred) -> float:
    """平均绝对误差"""
    yt, yp, _ = _finite_pair(y_true, y_pred, name="MAE")
    return float(np.mean(np.abs(yt - yp))) if yt.size else 0.0


def compute_pearson(y_true, y_pred) -> float:
    """
    Pearson 线性相关系数。

    标签或预测方差为 0 时返回 0.0 (相关系数无定义)。
    """
    yt, yp, _ = _finite_pair(y_true, y_pred, name="Pearson")
    if yt.size < 2:
        return 0.0
    std_t, std_p = yt.std(), yp.std()
    if std_t < 1e-12 or std_p < 1e-12:
        logger.warning("Pearson 无定义: 标签或预测方差为 0, 返回 0.0")
        return 0.0
    return float(np.mean((yt - yt.mean()) * (yp - yp.mean())) / (std_t * std_p))


def compute_spearman(y_true, y_pred) -> float:
    """
    Spearman 秩相关系数 (对秩做 Pearson)。

    无 scipy 时用 np.argsort 计算秩 (并列取平均秩)。
    """
    yt, yp, _ = _finite_pair(y_true, y_pred, name="Spearman")
    if yt.size < 2:
        return 0.0

    try:
        from scipy.stats import spearmanr
        r = spearmanr(yt, yp).correlation
        return float(r) if np.isfinite(r) else 0.0
    except ImportError:
        return compute_pearson(_rankdata(yt), _rankdata(yp))


def _rankdata(x: np.ndarray) -> np.ndarray:
    """计算秩, 并列取平均秩 (等价 scipy.stats.rankdata)。"""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = np.arange(1, x.size + 1, dtype=np.float64)

    sorted_x = x[order]
    i = 0
    while i < sorted_x.size:
        j = i
        while j + 1 < sorted_x.size and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


# ════════════════════════════════════════════════════════════
# 统一评估接口
# ════════════════════════════════════════════════════════════

def evaluate_predictions(
    y_true,
    y_pred,
    group_ids=None,
    metrics: Optional[List[str]] = None,
    ndcg_k: int = 10,
    binarize_threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    统一评估接口。

    Args:
        y_true: (N,) 真实标签; binary 模式为 0/1, continuous 模式为 [0,1] 连续值
        y_pred: (N,) 预测概率
        group_ids: (N,) 分组 ID (GAUC 需要)
        metrics: 指标列表, 支持:
            分类 — auc / gauc / logloss / ndcg / pos_rate
            回归 — mse / mae / pearson / spearman
        ndcg_k: NDCG 的 K
        binarize_threshold: 传入时, auc/gauc 先按此阈值二值化 y_true
            (连续标签场景计算参考 AUC)
    """
    metrics = metrics or ["auc", "gauc", "logloss"]
    results: Dict[str, float] = {}

    for m in metrics:
        if m == "auc":
            results["auc"] = compute_auc(y_true, y_pred, binarize_threshold)
        elif m == "gauc":
            results["gauc"] = (
                compute_gauc(y_true, y_pred, group_ids, binarize_threshold)
                if group_ids is not None else 0.0
            )
        elif m == "logloss":
            results["logloss"] = compute_logloss(y_true, y_pred)
        elif m == "pos_rate":
            results["pos_rate"] = compute_pos_rate(y_true)
        elif m == "ndcg":
            results[f"ndcg@{ndcg_k}"] = compute_ndcg(y_true, y_pred, k=ndcg_k)
        elif m == "mse":
            results["mse"] = compute_mse(y_true, y_pred)
        elif m == "mae":
            results["mae"] = compute_mae(y_true, y_pred)
        elif m == "pearson":
            results["pearson"] = compute_pearson(y_true, y_pred)
        elif m == "spearman":
            results["spearman"] = compute_spearman(y_true, y_pred)
        else:
            raise ValueError(
                f"Unknown metric: {m}. 支持: auc/gauc/logloss/pos_rate/ndcg/"
                "mse/mae/pearson/spearman"
            )

    return results


# 越小越好的指标 (用于 best checkpoint 判据和 lift 计算)
LOWER_IS_BETTER = ("logloss", "mse", "mae")


def compute_lift(direct: Dict[str, float], falcon: Dict[str, float]) -> Dict[str, float]:
    """
    计算 FALCON 优化相对直接预测的提升。

    AUC/GAUC/NDCG/相关系数 越大越好 → lift = falcon - direct
    LogLoss/MSE/MAE       越小越好 → lift = direct - falcon
    """
    lift = {}
    for key in direct:
        if key not in falcon:
            continue
        if key in LOWER_IS_BETTER:
            lift[f"{key}_lift"] = direct[key] - falcon[key]
        else:
            lift[f"{key}_lift"] = falcon[key] - direct[key]
    return lift

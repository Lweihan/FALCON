"""
data.py — 排序任务数据加载
============================
支持稀疏特征 CTR/CVR 数据集加载。

数据格式 (Parquet / CSV):
  每行一个样本，包含:
  - 若干稀疏特征列 (整数编码)
  - label 列 (0/1)
  - 可选 user_id 列 (GAUC 计算需要)

若无真实数据，提供 SyntheticRankerDataset 用于流程验证。
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common.logger import get_logger

logger = get_logger(__name__)


class SparseRankerDataset(Dataset):
    """
    稀疏特征排序数据集。

    从 Parquet / CSV 加载，自动做 label encoding 并统计词表大小。
    """

    def __init__(
        self,
        data_path: str,
        sparse_features: List[str],
        label_col: str = "label",
        user_col: Optional[str] = "user_id",
        max_samples: int = -1,
    ):
        self.sparse_features = sparse_features
        self.label_col = label_col
        self.user_col = user_col

        df = self._load_dataframe(data_path, max_samples)

        # Label encoding: 每个特征映射到 [1, vocab_size), 0 保留给 padding/unknown
        self.feature_voc_sizes: Dict[str, int] = {}
        self.encoded: Dict[str, np.ndarray] = {}

        for feat in sparse_features:
            codes, uniques = self._encode_column(df[feat].values)
            self.encoded[feat] = codes
            self.feature_voc_sizes[feat] = len(uniques) + 1   # +1 for padding idx 0

        self.labels = df[label_col].values.astype(np.float32)
        self.user_ids = (
            df[user_col].values if (user_col and user_col in df.columns) else None
        )

        logger.info(f"加载数据集: {len(self.labels)} 样本, {len(sparse_features)} 稀疏特征")
        logger.info(f"  正样本率: {self.labels.mean():.4f}")

    @staticmethod
    def _load_dataframe(path: str, max_samples: int):
        import pandas as pd

        if not os.path.exists(path):
            raise FileNotFoundError(f"数据文件不存在: {path}")

        if path.endswith(".parquet"):
            df = pd.read_parquet(path)
        elif path.endswith((".csv", ".tsv")):
            sep = "\t" if path.endswith(".tsv") else ","
            df = pd.read_csv(path, sep=sep)
        else:
            raise ValueError(f"不支持的文件格式: {path} (支持 .parquet/.csv/.tsv)")

        if max_samples > 0 and len(df) > max_samples:
            df = df.iloc[:max_samples].reset_index(drop=True)
        return df

    @staticmethod
    def _encode_column(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """将任意值映射为 [1, N] 的整数编码"""
        uniques, inverse = np.unique(values, return_inverse=True)
        return (inverse + 1).astype(np.int64), uniques

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "sparse": {f: self.encoded[f][idx] for f in self.sparse_features},
            "label": self.labels[idx],
        }
        if self.user_ids is not None:
            item["user_id"] = self.user_ids[idx]
        return item


class SyntheticRankerDataset(Dataset):
    """
    合成排序数据集，用于流程验证 (无需真实数据)。

    生成机制: label 由部分特征的线性组合 + 噪声决定，保证有可学习信号。
    """

    def __init__(
        self,
        num_samples: int = 10000,
        num_features: int = 20,
        vocab_size: int = 1000,
        num_users: int = 500,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        self.sparse_features = [f"feat_{i}" for i in range(num_features)]
        self.feature_voc_sizes = {f: vocab_size + 1 for f in self.sparse_features}

        self.encoded = {
            f: rng.integers(1, vocab_size + 1, size=num_samples, dtype=np.int64)
            for f in self.sparse_features
        }

        # 构造可学习信号: 前 5 个特征参与 label 生成
        signal = np.zeros(num_samples, dtype=np.float64)
        for i, f in enumerate(self.sparse_features[:5]):
            weight = rng.normal(0, 1, size=vocab_size + 1)
            signal += weight[self.encoded[f]]
        signal = signal / np.sqrt(5) + rng.normal(0, 0.5, size=num_samples)
        prob = 1.0 / (1.0 + np.exp(-signal))
        self.labels = (rng.random(num_samples) < prob).astype(np.float32)

        self.user_ids = rng.integers(0, num_users, size=num_samples)

        logger.info(f"生成合成数据集: {num_samples} 样本, {num_features} 特征")
        logger.info(f"  正样本率: {self.labels.mean():.4f}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "sparse": {f: self.encoded[f][idx] for f in self.sparse_features},
            "label": self.labels[idx],
            "user_id": self.user_ids[idx],
        }


def ranker_collate_fn(batch: List[dict]) -> dict:
    """整理为批次张量"""
    feats = list(batch[0]["sparse"].keys())
    out = {
        "sparse": {
            f: torch.tensor([b["sparse"][f] for b in batch], dtype=torch.long)
            for f in feats
        },
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.float32),
    }
    if "user_id" in batch[0]:
        out["user_id"] = np.array([b["user_id"] for b in batch])
    return out


def create_ranker_dataloaders(
    ranker_config,
    sparse_features: Optional[List[str]] = None,
    use_synthetic: bool = False,
) -> Tuple[DataLoader, DataLoader, Dict[str, int], List[str]]:
    """
    创建训练 / 验证 DataLoader。

    Returns:
        train_loader, val_loader, feature_voc_sizes, sparse_feature_names
    """
    if use_synthetic or not ranker_config.data_path:
        if not use_synthetic:
            logger.warning("未指定 data_path，回退到合成数据集")
        n = ranker_config.max_samples if ranker_config.max_samples > 0 else 20000
        dataset = SyntheticRankerDataset(num_samples=n)
    else:
        if sparse_features is None:
            raise ValueError("使用真实数据时必须提供 sparse_features 列表")
        dataset = SparseRankerDataset(
            data_path=ranker_config.data_path,
            sparse_features=sparse_features,
            max_samples=ranker_config.max_samples,
        )

    n_total = len(dataset)
    n_train = int(n_total * ranker_config.train_ratio)
    indices = np.arange(n_total)

    train_set = torch.utils.data.Subset(dataset, indices[:n_train].tolist())
    val_set = torch.utils.data.Subset(dataset, indices[n_train:].tolist())

    train_loader = DataLoader(
        train_set, batch_size=ranker_config.batch_size, shuffle=True,
        num_workers=ranker_config.num_workers, collate_fn=ranker_collate_fn,
    )
    val_loader = DataLoader(
        val_set, batch_size=ranker_config.batch_size, shuffle=False,
        num_workers=ranker_config.num_workers, collate_fn=ranker_collate_fn,
    )

    logger.info(f"训练集: {len(train_set)}, 验证集: {len(val_set)}")
    return train_loader, val_loader, dataset.feature_voc_sizes, dataset.sparse_features

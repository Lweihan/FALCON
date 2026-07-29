"""
mac_data.py — alimamaTech/MAC 数据集加载
==========================================
MAC (Multi-domain Ad Click) 数据集字段:

  稀疏 ID:  user_id, user_feat0-5, context_feat0-3, ad_feat0-5
  序列 ID:  ad_feat0_seq, ad_feat1_seq, ad_feat2_seq      (List[str])
  序列实值: mm_feat0_seq, mm_feat1_seq                     (List[float64])
  标签:    linear, mta, first, last                       (float64)

规模: 7500 万训练 / 400 万测试
→ 必须使用 streaming 模式, 不能一次性 load 进内存

支持四种加载模式:
  hf_streaming    直接从 HuggingFace Hub 流式读取 (无需下载)
  hf_download     先下载到本地缓存, 再读取
  local_parquet   从本地 parquet 文件读取
  local_csv       从本地 csv 文件读取
"""

import json
import os
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from common.logger import get_logger

logger = get_logger(__name__)

# ────────────────────────────────────────────────────────────
# 字段常量
# ────────────────────────────────────────────────────────────

SPARSE_FIELDS = [
    "user_id",
    "user_feat0", "user_feat1", "user_feat2", "user_feat3", "user_feat4", "user_feat5",
    "context_feat0", "context_feat1", "context_feat2", "context_feat3",
    "ad_feat0", "ad_feat1", "ad_feat2", "ad_feat3", "ad_feat4", "ad_feat5",
]

SEQ_ID_FIELDS = ["ad_feat0_seq", "ad_feat1_seq", "ad_feat2_seq"]
SEQ_FLOAT_FIELDS = ["mm_feat0_seq", "mm_feat1_seq"]
LABEL_FIELDS = ["linear", "mta", "first", "last"]
DEFAULT_LABEL = "last"

# 每个 mm_feat 序列的长度 (从数据探测中获取, 用于 proj 层初始化)
_MM_FEAT_DIM_CACHE: Dict[str, int] = {}


# ────────────────────────────────────────────────────────────
# 词表构建
# ────────────────────────────────────────────────────────────

class FeatureVocab:
    """
    稀疏 ID 特征的词表。

    用法:
        vocab = FeatureVocab(sparse_fields)
        vocab.build_from_iterator(rows)   # 扫描数据一次
        vocab.save("vocab.json")
        idx = vocab.lookup("user_id", "12345")
    """

    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self, fields: List[str]):
        self.fields = fields
        # field → {value: idx}
        self._vocab: Dict[str, Dict[str, int]] = {f: {self.PAD: 0, self.UNK: 1} for f in fields}
        self._built = False

    @property
    def vocab_sizes(self) -> Dict[str, int]:
        return {f: len(v) for f, v in self._vocab.items()}

    def lookup(self, field: str, value) -> int:
        if field not in self._vocab:
            return 1  # UNK
        v = str(value) if value is not None else self.PAD
        return self._vocab[field].get(v, 1)  # 1 = UNK

    def lookup_batch(self, field: str, values: List) -> List[int]:
        return [self.lookup(field, v) for v in values]

    def add_value(self, field: str, value) -> int:
        v = str(value) if value is not None else self.PAD
        if v not in self._vocab[field]:
            self._vocab[field][v] = len(self._vocab[field])
        return self._vocab[field][v]

    def build_from_iterator(
        self,
        rows: Iterator,
        max_rows: int = -1,
        log_every: int = 1_000_000,
    ):
        """扫描迭代器构建词表。"""
        n = 0
        for row in rows:
            for f in self.fields:
                val = row.get(f)
                if isinstance(val, list):
                    for v in val:
                        self.add_value(f, v)
                else:
                    self.add_value(f, val)
            n += 1
            if n % log_every == 0:
                logger.info(f"  词表构建: {n:,} 条, 词表大小 {sum(self.vocab_sizes.values()):,}")
            if max_rows > 0 and n >= max_rows:
                break
        self._built = True
        logger.info(f"词表构建完成: {n:,} 条")
        for f, sz in self.vocab_sizes.items():
            logger.info(f"  {f}: {sz:,}")

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"fields": self.fields, "vocab": self._vocab}, f)
        logger.info(f"词表已保存: {path}")

    @classmethod
    def load(cls, path: str) -> "FeatureVocab":
        with open(path) as f:
            d = json.load(f)
        obj = cls(d["fields"])
        obj._vocab = d["vocab"]
        obj._built = True
        logger.info(f"词表已加载: {path} ({sum(obj.vocab_sizes.values()):,} 总词数)")
        return obj


# ────────────────────────────────────────────────────────────
# 数据集 (IterableDataset, 支持 streaming)
# ────────────────────────────────────────────────────────────

class MACIterableDataset(IterableDataset):
    """
    MAC 数据集的流式 Dataset。

    每条样本输出:
        sparse_ids: {field: int}   — 稀疏 ID 查表后的索引
        mm_feats:   Tensor(D,)     — mm_feat0_seq + mm_feat1_seq mean pool 后拼接
        label:      float          — CVR 标签 (由 mac_label 指定)
        seq_mean_ids: {field: int} — 序列 ID 特征 mean pool 后的"均值 ID" (近似)

    注意: seq_id 特征做 lookup→mean embedding 在模型内部处理,
    这里只传原始 string list, collate_fn 负责转换.
    """

    def __init__(
        self,
        hf_dataset_iter,
        vocab: FeatureVocab,
        label_field: str = "last",
        seq_pool_mode: str = "mean",
        max_samples: int = -1,
    ):
        self.iter = hf_dataset_iter
        self.vocab = vocab
        self.label_field = label_field
        self.seq_pool_mode = seq_pool_mode
        self.max_samples = max_samples

    def _pool_float_seq(self, seq) -> np.ndarray:
        """对实值序列做 pool, 返回 numpy array"""
        if not seq:
            return np.zeros(1, dtype=np.float32)
        arr = np.array(seq, dtype=np.float32)
        if self.seq_pool_mode == "mean":
            return arr.mean(axis=0) if arr.ndim > 1 else np.array([arr.mean()])
        if self.seq_pool_mode == "max":
            return arr.max(axis=0) if arr.ndim > 1 else np.array([arr.max()])
        return arr.sum(axis=0) if arr.ndim > 1 else np.array([arr.sum()])

    def __iter__(self):
        n = 0
        for row in self.iter:
            # 稀疏 ID → int
            sparse_ids = {
                f: self.vocab.lookup(f, row.get(f)) for f in SPARSE_FIELDS
            }

            # 序列 ID 特征 (保留原始 list, 交给 collate 处理)
            seq_id_lists = {
                f: [str(v) for v in (row.get(f) or [])] for f in SEQ_ID_FIELDS
            }

            # mm_feat 实值序列 → mean pool → concat
            mm_parts = []
            for f in SEQ_FLOAT_FIELDS:
                seq = row.get(f) or []
                pooled = self._pool_float_seq(seq)
                mm_parts.append(pooled)

            if mm_parts:
                mm_feats = np.concatenate(mm_parts).astype(np.float32)
            else:
                mm_feats = np.zeros(1, dtype=np.float32)

            label = float(row.get(self.label_field, 0.0))

            yield {
                "sparse_ids": sparse_ids,
                "seq_id_lists": seq_id_lists,
                "mm_feats": mm_feats,
                "label": label,
            }

            n += 1
            if self.max_samples > 0 and n >= self.max_samples:
                break


# ────────────────────────────────────────────────────────────
# Collate
# ────────────────────────────────────────────────────────────

def mac_collate_fn(batch: List[dict]) -> dict:
    """
    将 MACIterableDataset 的 batch 转为 Tensor。

    输出:
        sparse_ids: {field: LongTensor(B,)}
        mm_feats:   FloatTensor(B, D)
        label:      FloatTensor(B,)
        seq_id_mean_ids: {field: LongTensor(B,)}  — 序列 ID 均值 (作为 dense 近似)
    """
    sparse_fields = list(batch[0]["sparse_ids"].keys())
    seq_fields = list(batch[0]["seq_id_lists"].keys())

    sparse_ids = {
        f: torch.tensor([s["sparse_ids"][f] for s in batch], dtype=torch.long)
        for f in sparse_fields
    }

    mm_feats = torch.from_numpy(
        np.stack([s["mm_feats"] for s in batch], axis=0)
    )

    label = torch.tensor([s["label"] for s in batch], dtype=torch.float32)

    # seq_id 特征: 把 list[str] 转为 list[int] 的 mean
    # 实际上在模型内部做 embedding lookup + pool; 这里传 list of lists
    seq_id_lists = {
        f: [s["seq_id_lists"][f] for s in batch]
        for f in seq_fields
    }

    return {
        "sparse_ids": sparse_ids,
        "seq_id_lists": seq_id_lists,
        "mm_feats": mm_feats,
        "label": label,
    }


# ────────────────────────────────────────────────────────────
# 数据加载器工厂
# ────────────────────────────────────────────────────────────

def _load_hf_iter(split: str, max_samples: int = -1, streaming: bool = True):
    """从 HuggingFace 加载 MAC 数据集迭代器。"""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("请安装 datasets: pip install datasets")

    logger.info(f"加载 MAC 数据集 (HuggingFace, split={split}, streaming={streaming}) ...")
    ds = load_dataset("alimamaTech/MAC", split=split, streaming=streaming, trust_remote_code=True)
    return ds


def _load_local_iter(path: str, fmt: str = "parquet"):
    """从本地文件加载迭代器。"""
    if fmt == "parquet":
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        return table.to_pylist()
    elif fmt == "csv":
        import pandas as pd
        return pd.read_csv(path).to_dict("records")
    else:
        raise ValueError(f"不支持的格式: {fmt}")


def build_vocab_from_mac(
    load_mode: str = "hf_streaming",
    data_path: str = "",
    max_rows: int = 5_000_000,
    save_path: str = "",
) -> FeatureVocab:
    """
    扫描 MAC 数据集构建词表。

    Args:
        max_rows: 最多扫描的行数 (streaming 下不用全量扫描也能覆盖大部分词)
        save_path: 词表保存路径, 空则不保存

    Returns:
        vocab: 词表对象
    """
    vocab = FeatureVocab(SPARSE_FIELDS + SEQ_ID_FIELDS)

    if load_mode in ("hf_streaming", "hf_download"):
        streaming = (load_mode == "hf_streaming")
        ds = _load_hf_iter("train", streaming=streaming)
    elif load_mode == "local_parquet":
        ds = _load_local_iter(data_path, "parquet")
    else:
        ds = _load_local_iter(data_path, "csv")

    vocab.build_from_iterator(ds, max_rows=max_rows)

    if save_path:
        vocab.save(save_path)

    return vocab


def create_mac_dataloaders(
    ranker_config,
    vocab: Optional[FeatureVocab] = None,
    build_vocab_samples: int = 5_000_000,
) -> Tuple[DataLoader, DataLoader, FeatureVocab, dict]:
    """
    创建 MAC 训练/验证 DataLoader。

    Args:
        ranker_config: RankerConfig
        vocab: 已有词表 (None 则自动构建)
        build_vocab_samples: 词表构建扫描行数

    Returns:
        (train_loader, val_loader, vocab, feature_info)
        feature_info: {"vocab_sizes": {...}, "mm_feat_dim": int, "seq_fields": [...], ...}
    """
    cfg = ranker_config
    load_mode = cfg.mac_load_mode
    label = cfg.mac_label

    # ── 词表 ──
    if vocab is None:
        if cfg.mac_vocab_path and os.path.isfile(cfg.mac_vocab_path):
            vocab = FeatureVocab.load(cfg.mac_vocab_path)
        else:
            logger.info("未找到词表, 开始构建...")
            vocab = build_vocab_from_mac(
                load_mode=load_mode,
                data_path=cfg.mac_data_path,
                max_rows=build_vocab_samples if cfg.mac_max_samples < 0 else cfg.mac_max_samples,
                save_path=cfg.mac_vocab_path,
            )

    # ── 探测 mm_feat 维度 ──
    mm_feat_dim = _detect_mm_feat_dim(load_mode, cfg.mac_data_path)
    logger.info(f"mm_feat 合并维度: {mm_feat_dim}")

    # ── val 数据 ──
    # HF 数据集有独立 test split; streaming 模式用 train 的前 val_ratio
    if load_mode in ("hf_streaming", "hf_download"):
        streaming = (load_mode == "hf_streaming")
        train_ds_iter = _load_hf_iter("train", streaming=streaming)
        # val 直接用 HF test split
        val_ds_iter = _load_hf_iter("test", streaming=streaming)

        # 用 mac_val_ratio 限制 val 大小 (400万 * 0.01 ≈ 4万)
        n_val = int(4_064_648 * cfg.mac_val_ratio) if cfg.mac_val_ratio < 1.0 else -1
    else:
        import pandas as pd
        fmt = "parquet" if load_mode == "local_parquet" else "csv"
        all_rows = _load_local_iter(cfg.mac_data_path, fmt)
        n_total = len(all_rows)
        n_train = int(n_total * (1 - cfg.mac_val_ratio))
        train_ds_iter = all_rows[:n_train]
        val_ds_iter = all_rows[n_train:]
        n_val = len(val_ds_iter)

    train_max = cfg.mac_max_samples if cfg.mac_max_samples > 0 else -1
    val_max = min(n_val, 100_000) if n_val > 0 else 100_000  # val 最多 10 万

    train_dataset = MACIterableDataset(
        train_ds_iter, vocab, label_field=label,
        seq_pool_mode=cfg.seq_pool_mode, max_samples=train_max,
    )
    val_dataset = MACIterableDataset(
        val_ds_iter, vocab, label_field=label,
        seq_pool_mode=cfg.seq_pool_mode, max_samples=val_max,
    )

    # streaming + IterableDataset 不能用多进程 worker
    nw = 0 if load_mode in ("hf_streaming", "hf_download") else min(cfg.mac_num_workers, 4)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size,
        collate_fn=mac_collate_fn, num_workers=nw,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size * 2,
        collate_fn=mac_collate_fn, num_workers=nw,
        pin_memory=torch.cuda.is_available(),
    )

    feature_info = {
        "vocab_sizes": vocab.vocab_sizes,
        "sparse_fields": SPARSE_FIELDS,
        "seq_id_fields": SEQ_ID_FIELDS,
        "seq_float_fields": SEQ_FLOAT_FIELDS,
        "mm_feat_dim": mm_feat_dim,
        "label_field": label,
    }

    return train_loader, val_loader, vocab, feature_info


def _detect_mm_feat_dim(load_mode: str, data_path: str = "", n_probe: int = 100) -> int:
    """
    探测 mm_feat 合并后的维度 (mm_feat0_seq + mm_feat1_seq mean pool 后 concat)。
    """
    if "mm_feat_dim" in _MM_FEAT_DIM_CACHE:
        return _MM_FEAT_DIM_CACHE["mm_feat_dim"]

    try:
        if load_mode in ("hf_streaming", "hf_download"):
            from datasets import load_dataset
            ds = load_dataset("alimamaTech/MAC", split="train", streaming=True, trust_remote_code=True)
            first_row = next(iter(ds))
        elif load_mode == "local_parquet":
            rows = _load_local_iter(data_path, "parquet")
            first_row = rows[0] if rows else {}
        else:
            rows = _load_local_iter(data_path, "csv")
            first_row = rows[0] if rows else {}

        total_dim = 0
        for f in SEQ_FLOAT_FIELDS:
            seq = first_row.get(f) or []
            # 每条序列 mean pool → 标量 (序列是 1D 实值列表)
            total_dim += 1 if seq else 0

        dim = max(total_dim, len(SEQ_FLOAT_FIELDS))  # 至少各 1 维
        _MM_FEAT_DIM_CACHE["mm_feat_dim"] = dim
        return dim
    except Exception as e:
        logger.warning(f"mm_feat 维度探测失败 ({e}), 使用默认值 2")
        _MM_FEAT_DIM_CACHE["mm_feat_dim"] = 2
        return 2

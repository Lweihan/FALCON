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

支持五种加载模式:
  local_dir       从本地 MAC 目录加载 (推荐, 默认)
                  要求目录结构:
                    MAC/
                    ├── train/   *.parquet   训练数据
                    ├── test/    *.parquet   验证数据
                    └── vocabs/  {field}.json 每字段一个词表文件
                  parquet 逐行组流式读取, 不会一次性加载到内存
  hf_streaming    直接从 HuggingFace Hub 流式读取 (无需下载)
  hf_download     先下载到本地缓存 (~80GB), 再读取
  local_parquet   单个 parquet 文件 (全量读入内存)
  local_csv       单个 csv 文件 (全量读入内存)
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

    @classmethod
    def load_from_dir(cls, vocab_dir: str, fields: Optional[List[str]] = None) -> "FeatureVocab":
        """
        从词表目录加载 (每个字段一个 json 文件)。

        MAC 数据集的 vocabs/ 目录结构:
            vocabs/
              user_id.json
              user_feat0.json
              ...
              ad_feat2_seq.json

        兼容两种 json 内容格式:
          A) dict:  {"value1": 0, "value2": 1, ...}     直接是 value → idx 映射
          B) list:  ["value1", "value2", ...]           按下标建立映射

        统一在开头保留 PAD=0 / UNK=1;
        若原 vocab 已含 PAD/UNK 则复用, 否则整体索引偏移 +2。
        """
        if not os.path.isdir(vocab_dir):
            raise FileNotFoundError(f"词表目录不存在: {vocab_dir}")

        fields = fields or (SPARSE_FIELDS + SEQ_ID_FIELDS)
        obj = cls(fields)
        found, missing = [], []

        for field in fields:
            fpath = os.path.join(vocab_dir, f"{field}.json")
            if not os.path.isfile(fpath):
                missing.append(field)
                continue

            try:
                with open(fpath) as f:
                    raw = json.load(f)
            except Exception as e:
                logger.warning(f"词表文件解析失败 {fpath}: {e}")
                missing.append(field)
                continue

            mapping: Dict[str, int] = {}

            if isinstance(raw, dict):
                # 格式 A: {value: idx}
                # 可能嵌套一层, 如 {"vocab": {...}} / {"stoi": {...}}
                for wrapper in ("vocab", "stoi", "word2idx", "mapping"):
                    if wrapper in raw and isinstance(raw[wrapper], dict):
                        raw = raw[wrapper]
                        break
                for k, v in raw.items():
                    try:
                        mapping[str(k)] = int(v)
                    except (TypeError, ValueError):
                        continue
            elif isinstance(raw, list):
                # 格式 B: [value, ...] 按下标
                for i, v in enumerate(raw):
                    mapping[str(v)] = i
            else:
                logger.warning(f"词表文件格式无法识别 (非 dict/list): {fpath}")
                missing.append(field)
                continue

            if not mapping:
                missing.append(field)
                continue

            # 归一化: 确保 PAD=0 / UNK=1
            has_pad = cls.PAD in mapping and mapping[cls.PAD] == 0
            has_unk = cls.UNK in mapping and mapping[cls.UNK] == 1

            if has_pad and has_unk:
                obj._vocab[field] = mapping
            else:
                # 移除可能存在但位置不对的 PAD/UNK, 然后整体偏移
                mapping.pop(cls.PAD, None)
                mapping.pop(cls.UNK, None)
                offset = 2
                new_map = {cls.PAD: 0, cls.UNK: 1}
                # 按原 idx 排序保持相对顺序稳定
                for k, v in sorted(mapping.items(), key=lambda kv: kv[1]):
                    new_map[k] = v + offset if v + offset not in (0, 1) else len(new_map)
                # 保险: 如果原 idx 有重复/空洞, 重新紧凑编号
                if len(set(new_map.values())) != len(new_map):
                    new_map = {cls.PAD: 0, cls.UNK: 1}
                    for i, k in enumerate(sorted(mapping.keys())):
                        new_map[k] = i + 2
                obj._vocab[field] = new_map

            found.append(field)

        obj._built = True

        logger.info(f"词表已从目录加载: {vocab_dir}")
        logger.info(f"  成功 {len(found)}/{len(fields)} 个字段, "
                    f"总词数 {sum(obj.vocab_sizes.values()):,}")
        for f in found:
            logger.info(f"    {f}: {obj.vocab_sizes[f]:,}")
        if missing:
            logger.warning(f"  缺失字段 (仅含 PAD/UNK, vocab_size=2): {missing}")

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
        mm_feat_dim: int = -1,
    ):
        self.iter = hf_dataset_iter
        self.vocab = vocab
        self.label_field = label_field
        self.seq_pool_mode = seq_pool_mode
        self.max_samples = max_samples
        self.mm_feat_dim = mm_feat_dim   # -1 = 不做对齐

    def _pool_float_seq(self, seq) -> np.ndarray:
        """
        对实值序列做 pool, 返回 1D numpy array。

        支持:
          1D 列表 [f1, f2, ...]        → 标量 (shape (1,))
          2D 列表 [[..], [..], ...]    → 沿 axis=0 pool (shape (inner_dim,))
        """
        if seq is None or len(seq) == 0:
            return np.zeros(1, dtype=np.float32)

        arr = np.asarray(seq, dtype=np.float32)

        if arr.ndim == 1:
            if self.seq_pool_mode == "max":
                return np.array([arr.max()], dtype=np.float32)
            if self.seq_pool_mode == "sum":
                return np.array([arr.sum()], dtype=np.float32)
            return np.array([arr.mean()], dtype=np.float32)

        # arr.ndim >= 2: 沿第 0 维 pool
        if self.seq_pool_mode == "max":
            return arr.max(axis=0).astype(np.float32).reshape(-1)
        if self.seq_pool_mode == "sum":
            return arr.sum(axis=0).astype(np.float32).reshape(-1)
        return arr.mean(axis=0).astype(np.float32).reshape(-1)

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

            # mm_feat 实值序列 → pool → concat
            mm_parts = []
            for f in SEQ_FLOAT_FIELDS:
                mm_parts.append(self._pool_float_seq(row.get(f)))

            if mm_parts:
                mm_feats = np.concatenate(mm_parts).astype(np.float32)
            else:
                mm_feats = np.zeros(1, dtype=np.float32)

            # 维度对齐: 不同样本 mm_feat 长度可能不一致 (缺失/变长),
            # 统一 pad/truncate 到 mm_feat_dim, 否则 collate 时 np.stack 会报错
            if self.mm_feat_dim > 0 and mm_feats.shape[0] != self.mm_feat_dim:
                fixed = np.zeros(self.mm_feat_dim, dtype=np.float32)
                n = min(mm_feats.shape[0], self.mm_feat_dim)
                fixed[:n] = mm_feats[:n]
                mm_feats = fixed

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
    """从本地单文件加载 (全量读入内存, 仅适合小文件)。"""
    if fmt == "parquet":
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        return table.to_pylist()
    elif fmt == "csv":
        import pandas as pd
        return pd.read_csv(path).to_dict("records")
    else:
        raise ValueError(f"不支持的格式: {fmt}")


class ParquetDirIterable:
    """
    可重复迭代的 parquet 目录流式读取器。

    逐文件、逐行组 (row group batch) 读取, 避免一次性加载数十 GB 到内存。
    实现 __iter__ 使其可被多个 epoch 重复遍历 (普通 generator 只能迭代一次)。

    用法:
        it = ParquetDirIterable("/data/MAC/train", batch_size=10000)
        for row in it:   # row 是 dict
            ...
    """

    def __init__(self, dir_path: str, batch_size: int = 10000, columns: Optional[List[str]] = None):
        import glob as _glob

        self.dir_path = dir_path
        self.batch_size = batch_size
        self.columns = columns

        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"数据目录不存在: {dir_path}")

        self.files = sorted(_glob.glob(os.path.join(dir_path, "*.parquet")))
        if not self.files:
            # 兼容 snappy 后缀等变体
            self.files = sorted(
                p for p in _glob.glob(os.path.join(dir_path, "*"))
                if p.endswith((".parquet", ".pq"))
            )
        if not self.files:
            raise FileNotFoundError(f"目录下未找到 parquet 文件: {dir_path}")

        logger.info(f"parquet 目录: {dir_path} ({len(self.files)} 个文件)")

    def num_rows(self) -> int:
        """统计总行数 (只读 metadata, 很快)。"""
        import pyarrow.parquet as pq
        total = 0
        for f in self.files:
            try:
                total += pq.ParquetFile(f).metadata.num_rows
            except Exception:
                pass
        return total

    def __iter__(self) -> Iterator[dict]:
        import pyarrow.parquet as pq

        for fpath in self.files:
            try:
                pf = pq.ParquetFile(fpath)
            except Exception as e:
                logger.warning(f"跳过无法打开的 parquet 文件 {fpath}: {e}")
                continue

            for batch in pf.iter_batches(batch_size=self.batch_size, columns=self.columns):
                # RecordBatch → list of dict
                for row in batch.to_pylist():
                    yield row


def build_vocab_from_mac(
    load_mode: str = "local_dir",
    data_path: str = "",
    max_rows: int = 5_000_000,
    save_path: str = "",
    parquet_batch_size: int = 10000,
    train_subdir: str = "train",
) -> FeatureVocab:
    """
    扫描 MAC 数据集构建词表 (fallback 路径)。

    local_dir 模式下应优先使用 FeatureVocab.load_from_dir() 读取预置词表,
    只有在 vocabs/ 目录缺失时才走这里。

    Args:
        max_rows: 最多扫描的行数
        save_path: 词表保存路径, 空则不保存
    """
    vocab = FeatureVocab(SPARSE_FIELDS + SEQ_ID_FIELDS)

    if load_mode == "local_dir":
        ds = ParquetDirIterable(
            _resolve_subdir(data_path, train_subdir, ["train"]),
            batch_size=parquet_batch_size,
        )
    elif load_mode in ("hf_streaming", "hf_download"):
        streaming = (load_mode == "hf_streaming")
        ds = _load_hf_iter("train", streaming=streaming)
    elif load_mode == "local_parquet":
        ds = _load_local_iter(data_path, "parquet")
    else:
        ds = _load_local_iter(data_path, "csv")

    logger.info(f"开始扫描数据构建词表 (最多 {max_rows:,} 行)...")
    vocab.build_from_iterator(ds, max_rows=max_rows)

    if save_path:
        vocab.save(save_path)

    return vocab


def _resolve_subdir(root: str, configured: str, fallbacks: List[str]) -> str:
    """
    解析子目录路径, 支持命名兜底。

    优先用配置指定的名字; 若不存在则依次尝试 fallbacks
    (如 test / val / valid), 避免因目录命名差异导致训练中断。

    Returns:
        实际存在的目录路径; 全都不存在时返回配置指定的路径
        (让后续 ParquetDirIterable / load_from_dir 抛出明确错误)
    """
    primary = os.path.join(root, configured)
    if os.path.isdir(primary):
        return primary

    for name in fallbacks:
        if name == configured:
            continue
        alt = os.path.join(root, name)
        if os.path.isdir(alt):
            logger.warning(
                f"子目录 '{configured}/' 不存在, 自动改用 '{name}/' "
                f"(可通过配置 mac_*_subdir 显式指定)"
            )
            return alt

    return primary


def create_mac_dataloaders(
    ranker_config,
    vocab: Optional[FeatureVocab] = None,
    build_vocab_samples: int = 5_000_000,
) -> Tuple[DataLoader, DataLoader, FeatureVocab, dict]:
    """
    创建 MAC 训练/验证 DataLoader。

    local_dir 模式 (推荐, 默认):
        {mac_data_path}/
          {mac_train_subdir}/   *.parquet   → train_loader
          {mac_val_subdir}/     *.parquet   → val_loader   (默认 test/)
          {mac_vocab_subdir}/   {field}.json → vocab        (默认 vocabs/)

    Args:
        ranker_config: RankerConfig
        vocab: 已有词表 (None 则自动加载/构建)
        build_vocab_samples: 词表 fallback 构建时扫描的行数

    Returns:
        (train_loader, val_loader, vocab, feature_info)
    """
    cfg = ranker_config
    load_mode = cfg.mac_load_mode
    label = cfg.mac_label

    # ════════════════════════════════════════════════
    # local_dir 模式 (推荐)
    # ════════════════════════════════════════════════
    if load_mode == "local_dir":
        root = cfg.mac_data_path
        if not root:
            raise ValueError("local_dir 模式需要设置 mac_data_path (MAC 根目录)")
        if not os.path.isdir(root):
            raise FileNotFoundError(f"MAC 根目录不存在: {root}")

        train_dir = _resolve_subdir(root, cfg.mac_train_subdir, ["train"])
        val_dir = _resolve_subdir(root, cfg.mac_val_subdir, ["test", "val", "valid"])
        vocab_dir = _resolve_subdir(root, cfg.mac_vocab_subdir, ["vocabs", "vocab"])

        logger.info(f"MAC local_dir 模式: {root}")
        logger.info(f"  train: {train_dir}")
        logger.info(f"  eval:  {val_dir}")
        logger.info(f"  vocab: {vocab_dir}")

        # ── 词表: 优先读 vocabs/ 目录 ──
        if vocab is None:
            if os.path.isdir(vocab_dir):
                vocab = FeatureVocab.load_from_dir(vocab_dir)
            elif cfg.mac_vocab_path and os.path.isfile(cfg.mac_vocab_path):
                vocab = FeatureVocab.load(cfg.mac_vocab_path)
            else:
                logger.warning(
                    f"未找到词表目录 {vocab_dir}, fallback 到扫描 train 数据构建 "
                    "(耗时较长)"
                )
                vocab = build_vocab_from_mac(
                    load_mode="local_dir",
                    data_path=root,
                    max_rows=build_vocab_samples,
                    save_path=cfg.mac_vocab_path,
                    parquet_batch_size=cfg.mac_parquet_batch_size,
                    train_subdir=cfg.mac_train_subdir,
                )

        # ── 数据迭代器 ──
        train_src = ParquetDirIterable(train_dir, batch_size=cfg.mac_parquet_batch_size)
        val_src = ParquetDirIterable(val_dir, batch_size=cfg.mac_parquet_batch_size)

        # 打印规模 (只读 metadata)
        try:
            logger.info(f"  train 总行数: {train_src.num_rows():,}")
            logger.info(f"  val   总行数: {val_src.num_rows():,}")
        except Exception:
            pass

        # ── mm_feat 维度探测 (从第一行) ──
        mm_feat_dim = _detect_mm_feat_dim_from_row(_peek_first_row(train_src))

        train_max = cfg.mac_max_samples if cfg.mac_max_samples > 0 else -1
        val_max = cfg.mac_val_max_samples if cfg.mac_val_max_samples > 0 else -1

    # ════════════════════════════════════════════════
    # 其他模式 (hf_streaming / hf_download / local_parquet / local_csv)
    # ════════════════════════════════════════════════
    else:
        # ── 词表 ──
        if vocab is None:
            if cfg.mac_vocab_path and os.path.isfile(cfg.mac_vocab_path):
                vocab = FeatureVocab.load(cfg.mac_vocab_path)
            else:
                logger.info("未找到词表, 开始构建...")
                vocab = build_vocab_from_mac(
                    load_mode=load_mode,
                    data_path=cfg.mac_data_path,
                    max_rows=(build_vocab_samples if cfg.mac_max_samples < 0
                              else cfg.mac_max_samples),
                    save_path=cfg.mac_vocab_path,
                )

        if load_mode in ("hf_streaming", "hf_download"):
            streaming = (load_mode == "hf_streaming")
            train_src = _load_hf_iter("train", streaming=streaming)
            val_src = _load_hf_iter("test", streaming=streaming)
            n_val = (int(4_064_648 * cfg.mac_val_ratio)
                     if cfg.mac_val_ratio < 1.0 else -1)
        else:
            fmt = "parquet" if load_mode == "local_parquet" else "csv"
            all_rows = _load_local_iter(cfg.mac_data_path, fmt)
            n_total = len(all_rows)
            n_train = int(n_total * (1 - cfg.mac_val_ratio))
            train_src = all_rows[:n_train]
            val_src = all_rows[n_train:]
            n_val = len(val_src)

        mm_feat_dim = _detect_mm_feat_dim(load_mode, cfg.mac_data_path)

        train_max = cfg.mac_max_samples if cfg.mac_max_samples > 0 else -1
        val_max = min(n_val, cfg.mac_val_max_samples) if n_val > 0 else cfg.mac_val_max_samples

    logger.info(f"mm_feat 合并维度: {mm_feat_dim}")

    # ── 构建 Dataset ──
    train_dataset = MACIterableDataset(
        train_src, vocab, label_field=label,
        seq_pool_mode=cfg.seq_pool_mode, max_samples=train_max,
        mm_feat_dim=mm_feat_dim,
    )
    val_dataset = MACIterableDataset(
        val_src, vocab, label_field=label,
        seq_pool_mode=cfg.seq_pool_mode, max_samples=val_max,
        mm_feat_dim=mm_feat_dim,
    )

    # IterableDataset 用多 worker 会导致数据重复, 固定为 0
    nw = 0

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


def _peek_first_row(iterable) -> dict:
    """取迭代器/序列的第一行 (不消耗可重复迭代的 ParquetDirIterable)。"""
    try:
        if isinstance(iterable, (list, tuple)):
            return iterable[0] if iterable else {}
        return next(iter(iterable))
    except (StopIteration, IndexError, Exception):
        return {}


def _detect_mm_feat_dim_from_row(row: dict) -> int:
    """
    从单行样本探测 mm_feat 合并后的维度。

    mm_feat*_seq 是 List[float]; mean pool 后:
      - 若是 1D 实值列表 → 标量 (1 维)
      - 若是嵌套 2D → 内层维度
    """
    if not row:
        logger.warning("无法探测 mm_feat 维度 (空样本), 使用默认值 2")
        return len(SEQ_FLOAT_FIELDS)

    total = 0
    for f in SEQ_FLOAT_FIELDS:
        seq = row.get(f)
        if not seq:
            total += 1   # 缺失也占 1 维 (输出 0)
            continue
        first = seq[0]
        if isinstance(first, (list, tuple)):
            total += len(first)   # 2D: mean pool 沿 axis=0 → 内层维度
        else:
            total += 1            # 1D: mean → 标量
    return max(total, len(SEQ_FLOAT_FIELDS))


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

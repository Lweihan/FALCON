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
        label_mode: str = "binary",
        label_threshold: float = 0.0,
    ):
        self.iter = hf_dataset_iter
        self.vocab = vocab
        self.label_field = label_field
        self.seq_pool_mode = seq_pool_mode
        self.max_samples = max_samples
        self.mm_feat_dim = mm_feat_dim   # -1 = 不做对齐
        self.label_mode = label_mode
        self.label_threshold = label_threshold

        # NaN/Inf 统计: MAC 的 mm_feat0_seq / mm_feat1_seq 是真实多模态特征,
        # 存在缺失/异常值。经平均池化后 NaN 会传染整个样本 embedding,
        # 训练时导致 loss=NaN, 验证时导致 sklearn 崩溃。
        # 每 _nan_warn_interval 条样本 rank 0 累计打印一次警告。
        self._nan_bad_values = 0     # 累计被替换的 NaN/Inf 值个数
        self._nan_bad_samples = 0    # 累计触及 NaN 的样本数
        self._nan_total_samples = 0
        self._nan_warn_interval = 10000

        if label_mode not in ("binary", "continuous"):
            raise ValueError(
                f"label_mode 必须是 binary / continuous, got {label_mode!r}"
            )

    def _maybe_warn_nan(self):
        """
        rank 0 定期汇报 NaN 统计。DDP 下每个 rank 独立打印, 加 rank 前缀。
        """
        from common.distributed import is_main_process, get_rank
        if not is_main_process():
            return
        if self._nan_bad_samples == 0:
            return
        ratio = self._nan_bad_samples / max(self._nan_total_samples, 1)
        # 超过 1% 认为异常; 需上游检查
        level = "warning" if ratio > 0.01 else "debug"
        msg = (f"[数据 NaN] 已扫描 {self._nan_total_samples:,} 样本, "
               f"{self._nan_bad_samples:,} ({ratio:.2%}) 含 NaN/Inf, "
               f"共清理 {self._nan_bad_values:,} 个异常值")
        if level == "warning":
            logger.warning(msg)
        else:
            logger.debug(msg)

    def _transform_label(self, raw: float) -> float:
        """
        标签变换。

        MAC 的 linear/mta/first/last 是多触点归因 (MTA) 权重, 为连续实数。
          binary:     > threshold 视为正例 → 0.0 / 1.0
          continuous: clip 到 [0,1] 保留强度 (BCE 软标签要求目标在 [0,1])
        """
        if self.label_mode == "binary":
            return float(raw > self.label_threshold)
        return float(min(max(raw, 0.0), 1.0))

    def _pool_float_seq(self, seq) -> Tuple[np.ndarray, int]:
        """
        对实值序列做 pool, 返回 (1D numpy array, 被替换的 NaN/Inf 值个数)。

        为何要主动处理 NaN:
          MAC 数据集中 mm_feat0_seq / mm_feat1_seq 是真实多模态特征,
          原始数据中存在 NaN。若不处理, 平均池化后 NaN 会传染整个样本的
          embedding, 前向后 loss=NaN, sklearn 验证时抛
          "Input contains NaN" 直接崩溃。

        替换策略: NaN → 0, +Inf → 0, -Inf → 0 (nan_to_num 默认行为)。

        支持:
          1D 列表 [f1, f2, ...]        → 标量 (shape (1,))
          2D 列表 [[..], [..], ...]    → 沿 axis=0 pool (shape (inner_dim,))
        """
        if seq is None or len(seq) == 0:
            return np.zeros(1, dtype=np.float32), 0

        arr = np.asarray(seq, dtype=np.float32)

        # 先清理 NaN/Inf, 再池化 (顺序不能反, 否则 NaN 会污染 pool 结果)
        n_bad = int(np.count_nonzero(~np.isfinite(arr)))
        if n_bad:
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        if arr.ndim == 1:
            if self.seq_pool_mode == "max":
                out = np.array([arr.max()], dtype=np.float32)
            elif self.seq_pool_mode == "sum":
                out = np.array([arr.sum()], dtype=np.float32)
            else:
                out = np.array([arr.mean()], dtype=np.float32)
            return out, n_bad

        # arr.ndim >= 2: 沿第 0 维 pool
        if self.seq_pool_mode == "max":
            out = arr.max(axis=0).astype(np.float32).reshape(-1)
        elif self.seq_pool_mode == "sum":
            out = arr.sum(axis=0).astype(np.float32).reshape(-1)
        else:
            out = arr.mean(axis=0).astype(np.float32).reshape(-1)
        return out, n_bad

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

            # mm_feat 实值序列 → pool → concat (同时累计 NaN 统计)
            mm_parts = []
            sample_bad = 0
            for f in SEQ_FLOAT_FIELDS:
                pooled, bad = self._pool_float_seq(row.get(f))
                mm_parts.append(pooled)
                sample_bad += bad

            if mm_parts:
                mm_feats = np.concatenate(mm_parts).astype(np.float32)
            else:
                mm_feats = np.zeros(1, dtype=np.float32)

            # 池化后二次清理: 极端情况下上游全 NaN 或 sum overflow 可能残留 NaN
            if not np.isfinite(mm_feats).all():
                mm_feats = np.nan_to_num(mm_feats, nan=0.0, posinf=0.0, neginf=0.0)
                sample_bad = max(sample_bad, 1)

            # 累计 NaN 统计, 每 _nan_warn_interval 样本汇报一次
            self._nan_total_samples += 1
            if sample_bad > 0:
                self._nan_bad_values += sample_bad
                self._nan_bad_samples += 1
            if self._nan_total_samples % self._nan_warn_interval == 0:
                self._maybe_warn_nan()

            # 维度对齐: 不同样本 mm_feat 长度可能不一致 (缺失/变长),
            # 统一 pad/truncate 到 mm_feat_dim, 否则 collate 时 np.stack 会报错
            if self.mm_feat_dim > 0 and mm_feats.shape[0] != self.mm_feat_dim:
                fixed = np.zeros(self.mm_feat_dim, dtype=np.float32)
                n = min(mm_feats.shape[0], self.mm_feat_dim)
                fixed[:n] = mm_feats[:n]
                mm_feats = fixed

            raw_label = row.get(self.label_field)
            # 标签本身也可能是 NaN (数据缺失)
            if raw_label is None or not (isinstance(raw_label, (int, float))
                                          and np.isfinite(raw_label)):
                raw_label = 0.0
            raw_label = float(raw_label)
            label = self._transform_label(raw_label)

            yield {
                "sparse_ids": sparse_ids,
                "seq_id_lists": seq_id_lists,
                "mm_feats": mm_feats,
                "label": label,
                "raw_label": raw_label,   # 原始归因权重, 便于分析/调试
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
    raw_label = torch.tensor(
        [s.get("raw_label", s["label"]) for s in batch], dtype=torch.float32
    )

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
        "raw_label": raw_label,
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
    可重复迭代的 parquet 目录流式读取器, 支持 DDP 按 rank 切分。

    逐文件、逐行组 (row group batch) 读取, 避免一次性加载数十 GB 到内存。
    实现 __iter__ 使其可被多个 epoch 重复遍历 (普通 generator 只能迭代一次)。

    DDP 切分策略 (IterableDataset 无法使用 DistributedSampler, 必须自己切):
      策略 A — 文件级切分 (文件数 >= world_size, I/O 最优):
        files[rank::world_size], 每卡只读自己那批文件, I/O 真正并行
      策略 B — 行级切分 (文件数 < world_size, 退化方案):
        每卡读全部文件但按 idx % world_size == rank 取行
        (I/O 无法并行, 但保证数据不重复)

    用法:
        it = ParquetDirIterable("/data/MAC/train", batch_size=10000,
                                rank=0, world_size=4)
        for row in it:   # row 是 dict
            ...
    """

    def __init__(
        self,
        dir_path: str,
        batch_size: int = 10000,
        columns: Optional[List[str]] = None,
        rank: int = 0,
        world_size: int = 1,
        name: str = "",
    ):
        import glob as _glob

        self.dir_path = dir_path
        self.batch_size = batch_size
        self.columns = columns
        self.rank = max(rank, 0)
        self.world_size = max(world_size, 1)

        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"数据目录不存在: {dir_path}")

        all_files = sorted(_glob.glob(os.path.join(dir_path, "*.parquet")))
        if not all_files:
            # 兼容 snappy 后缀等变体
            all_files = sorted(
                p for p in _glob.glob(os.path.join(dir_path, "*"))
                if p.endswith((".parquet", ".pq"))
            )
        if not all_files:
            raise FileNotFoundError(f"目录下未找到 parquet 文件: {dir_path}")

        self.all_files = all_files

        # ── DDP 切分 ──
        if self.world_size <= 1:
            self.files = all_files
            self._row_level_split = False
        elif len(all_files) >= self.world_size:
            # 策略 A: 文件级切分
            self.files = all_files[self.rank::self.world_size]
            self._row_level_split = False
        else:
            # 策略 B: 行级切分
            self.files = all_files
            self._row_level_split = True

        tag = f"[{name}] " if name else ""
        if self.world_size > 1:
            mode = "行级" if self._row_level_split else "文件级"
            logger.info(
                f"{tag}parquet 目录: {dir_path} "
                f"(全局 {len(all_files)} 文件, rank {self.rank}/{self.world_size} "
                f"分到 {len(self.files)} 文件, {mode}切分)"
            )
        else:
            logger.info(f"{tag}parquet 目录: {dir_path} ({len(all_files)} 个文件)")

    def num_rows(self, local_only: bool = True) -> int:
        """
        统计行数 (只读 metadata, 很快)。

        Args:
            local_only: True 返回本 rank 实际会读的行数;
                        False 返回全局总行数
        """
        import pyarrow.parquet as pq

        files = self.files if local_only else self.all_files
        total = 0
        for f in files:
            try:
                total += pq.ParquetFile(f).metadata.num_rows
            except Exception:
                pass

        # 行级切分下, 本 rank 只取 1/world_size 的行
        if local_only and self._row_level_split:
            total = total // self.world_size
        return total

    def __iter__(self) -> Iterator[dict]:
        import pyarrow.parquet as pq

        idx = 0
        for fpath in self.files:
            try:
                pf = pq.ParquetFile(fpath)
            except Exception as e:
                logger.warning(f"跳过无法打开的 parquet 文件 {fpath}: {e}")
                continue

            for batch in pf.iter_batches(batch_size=self.batch_size, columns=self.columns):
                # RecordBatch → list of dict
                for row in batch.to_pylist():
                    if self._row_level_split:
                        if idx % self.world_size != self.rank:
                            idx += 1
                            continue
                    yield row
                    idx += 1


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

    # ── DDP: 训练集按 rank 切分 ──
    from common.distributed import get_rank, get_world_size, is_main_process
    rank, world_size = get_rank(), get_world_size()
    is_main = is_main_process()

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

        if is_main:
            logger.info(f"MAC local_dir 模式: {root}"
                        + (f" (DDP world_size={world_size})" if world_size > 1 else ""))
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
        # 训练集按 rank 切分 (DDP 数据并行);
        # 验证集不切分 — 各 rank 读完整验证集算指标后 all_reduce 取平均。
        #   原因: 验证集限制在 10 万条以内, 重复计算开销可接受;
        #   且切分后可能出现某 rank 拿到全负样本 → AUC 无定义
        train_src = ParquetDirIterable(
            train_dir, batch_size=cfg.mac_parquet_batch_size,
            rank=rank, world_size=world_size, name="train",
        )
        val_src = ParquetDirIterable(
            val_dir, batch_size=cfg.mac_parquet_batch_size,
            rank=0, world_size=1, name="eval",   # 不切分
        )

        # 打印规模 (只读 metadata)
        try:
            if world_size > 1:
                logger.info(f"  train 全局行数: {train_src.num_rows(local_only=False):,}"
                            f", 本 rank: {train_src.num_rows():,}")
            else:
                logger.info(f"  train 总行数: {train_src.num_rows():,}")
            logger.info(f"  eval  总行数: {val_src.num_rows():,} (各 rank 均读完整验证集)")
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
            if world_size > 1:
                raise RuntimeError(
                    f"{load_mode} 模式不支持 DDP 多卡 (world_size={world_size})。\n"
                    "HuggingFace streaming 的 IterableDataset 无法按 rank 切分, "
                    "各卡会读到完全相同的数据 → 无加速且梯度重复。\n"
                    "请改用 local_dir 模式 (先把数据集下载到本地 MAC/{train,test,vocabs}/), "
                    "或去掉 torchrun 用单卡运行。"
                )
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
            train_rows = all_rows[:n_train]
            val_src = all_rows[n_train:]
            n_val = len(val_src)
            # list 可直接按 rank 切片 (验证集不切)
            train_src = train_rows[rank::world_size] if world_size > 1 else train_rows
            if world_size > 1:
                logger.info(f"  rank {rank}/{world_size}: "
                            f"train {len(train_src):,}/{len(train_rows):,} 行")

        mm_feat_dim = _detect_mm_feat_dim(load_mode, cfg.mac_data_path)

        train_max = cfg.mac_max_samples if cfg.mac_max_samples > 0 else -1
        val_max = min(n_val, cfg.mac_val_max_samples) if n_val > 0 else cfg.mac_val_max_samples

    if is_main:
        logger.info(f"mm_feat 合并维度: {mm_feat_dim}")

    # ── 构建 Dataset ──
    ds_kwargs = dict(
        label_field=label,
        seq_pool_mode=cfg.seq_pool_mode,
        mm_feat_dim=mm_feat_dim,
        label_mode=cfg.mac_label_mode,
        label_threshold=cfg.mac_label_threshold,
    )
    train_dataset = MACIterableDataset(
        train_src, vocab, max_samples=train_max, **ds_kwargs
    )
    val_dataset = MACIterableDataset(
        val_src, vocab, max_samples=val_max, **ds_kwargs
    )

    if is_main:
        logger.info(f"标签口径: mode={cfg.mac_label_mode}, field={label}, "
                    f"threshold={cfg.mac_label_threshold}")

        # 标签分布探测会消耗迭代器: 只对可重复迭代的源做
        # (ParquetDirIterable 实现了 __iter__ 可重复; HF streaming 的 IterableDataset
        #  一旦消耗就取不回来, 跳过以免训练少样本)
        if isinstance(train_src, (ParquetDirIterable, list, tuple)):
            _log_label_stats(train_src, label, cfg.mac_label_mode, cfg.mac_label_threshold)
        else:
            logger.info("(streaming 源跳过标签分布探测, 避免消耗数据流)")


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
        "label_mode": cfg.mac_label_mode,
        "label_threshold": cfg.mac_label_threshold,
        "world_size": world_size,
        "rank": rank,
    }

    return train_loader, val_loader, vocab, feature_info


def _log_label_stats(
    source,
    label_field: str,
    label_mode: str,
    threshold: float,
    n_probe: int = 20000,
):
    """
    抽样探测标签分布并打印, 异常时明确 warning。

    这类问题在大数据集上很容易踩到且不报错:
      - 阈值设错 → 全负样本 → AUC 恒 0.5
      - 标签字段选错 → 全 0 或方差为 0 → 模型学不到东西
    """
    try:
        raws = []
        for i, row in enumerate(source):
            if i >= n_probe:
                break
            raws.append(float(row.get(label_field) or 0.0))
    except Exception as e:
        logger.warning(f"标签分布探测失败 ({e}), 跳过")
        return

    if not raws:
        logger.warning("标签分布探测: 未读到任何样本")
        return

    arr = np.asarray(raws, dtype=np.float64)
    n = arr.size
    uniq = np.unique(arr)

    logger.info(f"标签分布 (抽样 {n:,} 条, 字段 '{label_field}'):")
    logger.info(f"  原始值: min={arr.min():.4f}, max={arr.max():.4f}, "
                f"mean={arr.mean():.4f}, std={arr.std():.4f}, "
                f"唯一值数={len(uniq)}")

    if label_mode == "binary":
        pos_rate = float((arr > threshold).mean())
        logger.info(f"  二值化 (> {threshold}): 正样本率={pos_rate:.4%} "
                    f"({int(pos_rate * n):,} / {n:,})")
        if pos_rate == 0.0:
            logger.warning(
                f"  [!] 正样本率为 0! 所有标签都 <= {threshold}。"
                f"AUC 将恒为 0.5, 模型无法学习。\n"
                f"      建议: 检查 mac_label 字段 (当前 '{label_field}') 或调低 "
                f"mac_label_threshold (当前 {threshold}, 数据 max={arr.max():.4f})"
            )
        elif pos_rate == 1.0:
            logger.warning(
                f"  [!] 正样本率为 100%! 所有标签都 > {threshold}。"
                f"建议调高 mac_label_threshold (数据 min={arr.min():.4f})"
            )
        elif pos_rate < 0.0001:
            logger.warning(
                f"  [!] 正样本率极低 ({pos_rate:.6%}), 训练可能不稳定。"
                "考虑正负样本加权或下采样负例。"
            )
    else:
        clipped = np.clip(arr, 0.0, 1.0)
        qs = np.percentile(clipped, [25, 50, 75, 90, 99])
        logger.info(f"  clip 到 [0,1] 后: mean={clipped.mean():.4f}, "
                    f"std={clipped.std():.4f}")
        logger.info(f"  分位数 p25/p50/p75/p90/p99 = "
                    + " / ".join(f"{q:.4f}" for q in qs))
        n_clipped = int(((arr < 0) | (arr > 1)).sum())
        if n_clipped:
            logger.warning(f"  [!] {n_clipped:,}/{n:,} 条标签超出 [0,1] 已被截断")
        if clipped.std() < 1e-8:
            logger.warning(
                "  [!] 标签方差接近 0, 回归指标 (Pearson/Spearman) 将无定义。"
                f"检查 mac_label 字段 (当前 '{label_field}')"
            )
        if len(uniq) <= 2:
            logger.warning(
                f"  [!] 标签只有 {len(uniq)} 个唯一值 {uniq.tolist()}, "
                "本质是二值数据。建议改用 mac_label_mode='binary'"
            )


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

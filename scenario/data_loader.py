"""
data_loader.py — 用户 Embedding 时间序列数据加载
=================================================
支持三种配对策略 (pair_mode):

| 模式          | 说明                        | T=5 时样本数 |
|--------------|----------------------------|------------|
| last_target  | 随机早期步 → 最后一步 (默认)  | 1          |
| all_pairs    | 枚举所有 (i,j) i<j          | 10         |
| adjacent     | 相邻配对 (i, i+1)           | 4          |

同时输出 delta_t (归一化时间跨度)，用于注入网络修复训推不一致:
  - 训练时模型能感知 "预测多远的未来"
  - 推理时 delta_t 语义与训练一致

数据文件格式 (自动检测):

  ① 逐用户格式 (旧格式，user_XXXXXX.pt):
       { "embeddings": Tensor(T, dim), "times": [...], "user_id": "xxx" }
     每个用户一个文件，init 时扫描所有文件建索引。

  ② shard 格式 (shard_XXXXXX.pt, 推荐):
       { "embeddings": Tensor(N, T, dim), "times": [...],
         "user_ids":   ["uid1", ...],    # 可选
         "snap_date":  "20240101" }      # 可选，仅快照文件含此字段
     多用户打包为一个文件，init 时只读元信息（N/T）不加载 embeddings，
     __getitem__ 时懒加载整个 shard 并按行索引取用户，整个 shard 缓存
     至下一次加载不同 shard 时替换（单 worker LRU=1；多 worker 每进程独立缓存）。

性能优化:
  - shard 格式: 文件数 = 快照数（如 49），而非用户数（如 1M）；
    DataLoader init 极快，inode 压力极小。
  - lazy_load: shard 模式下始终懒加载（每次 __getitem__ 按需加载 shard）；
    逐用户格式下支持 lazy_load=True（每次加载单文件）/False（init 时全量缓存）。
  - num_workers>0: 配合 lazy_load 实现并行 IO，避免主进程成为瓶颈。
"""

import os
from datetime import datetime
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from common.logger import get_logger

logger = get_logger(__name__)


# ────────────────────────────────────────────────────────────
# 时间戳解析
# ────────────────────────────────────────────────────────────

_TIME_FORMATS = ["%Y%m%d%H%M", "%Y%m%d%H%M%S", "%Y%m%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]


def parse_timestamp(ts) -> Optional[datetime]:
    """解析时间戳字符串, 失败返回 None"""
    if ts is None:
        return None
    s = str(ts).strip()
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except (ValueError, TypeError):
            continue
    return None


def compute_delta_t(
    times: list,
    i: int,
    j: int,
    from_timestamp: bool = True,
) -> float:
    """
    计算从时间步 i 到 j 的归一化时间跨度 delta_t ∈ (0, 1]。

    策略:
      1. 优先用真实时间戳: (t_j - t_i) / (t_last - t_first)
      2. 回退到步数比例:   (j - i) / (T - 1)

    Args:
        times: 时间戳列表
        i, j: 起止索引 (i < j)
        from_timestamp: 是否优先使用真实时间戳

    Returns:
        delta_t ∈ (0, 1]
    """
    T = len(times)
    step_ratio = (j - i) / max(T - 1, 1)

    if not from_timestamp:
        return step_ratio

    t_i, t_j = parse_timestamp(times[i]), parse_timestamp(times[j])
    t_first, t_last = parse_timestamp(times[0]), parse_timestamp(times[-1])

    if None in (t_i, t_j, t_first, t_last):
        return step_ratio

    total_span = (t_last - t_first).total_seconds()
    if total_span <= 0:
        return step_ratio

    gap = (t_j - t_i).total_seconds()
    return max(min(gap / total_span, 1.0), 1e-4)


# ────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────

class UserEmbeddingDataset(Dataset):
    """
    从预保存的 .pt 文件加载用户 embedding 时间序列。

    自动识别两种文件格式:
      - shard_XXXXXX.pt   → shard 格式 (推荐，多用户打包)
      - user_XXXXXX.pt    → 逐用户格式 (旧格式，兼容)

    split 行为:
        - split="train" / "val": 按 train_ratio 从 data_dir 内部划分（旧行为）
        - use_all=True: 加载 data_dir 全部文件，忽略 split/train_ratio
          （适用于已外部预分好 train/test 目录的场景）
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        train_ratio: float = 0.8,
        pair_mode: str = "last_target",
        delta_t_from_timestamp: bool = True,
        lazy_load: bool = True,
        min_timesteps: int = 2,
        use_all: bool = False,
    ):
        if pair_mode not in ("last_target", "all_pairs", "adjacent"):
            raise ValueError(
                f"Unknown pair_mode: {pair_mode}, "
                f"choose from last_target/all_pairs/adjacent"
            )

        self.pair_mode = pair_mode
        self.delta_t_from_timestamp = delta_t_from_timestamp
        self.lazy_load = lazy_load

        # ── 扫描文件，区分 shard / user 两种格式 ──
        all_shard_files = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.startswith("shard_") and f.endswith(".pt")
        )
        all_user_files = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.startswith("user_") and f.endswith(".pt")
        )

        # 优先使用 shard 格式
        use_shard = len(all_shard_files) > 0
        if use_shard and len(all_user_files) > 0:
            logger.warning(
                f"目录同时含 shard_*.pt 和 user_*.pt，优先使用 shard 格式 "
                f"({len(all_shard_files)} 个 shard 文件)"
            )

        if use_shard:
            self._init_shard(
                all_shard_files, split, train_ratio, min_timesteps, use_all
            )
        else:
            self._init_user(
                all_user_files, split, train_ratio, min_timesteps, use_all
            )

        tag = "all" if use_all else split
        logger.info(
            f"[{tag}] 模式={'shard' if use_shard else 'user'}, "
            f"用户数={self._n_users}, 样本对数={len(self.pairs)}, "
            f"pair_mode={pair_mode}"
        )

    # ── 初始化：shard 格式 ──

    def _init_shard(
        self,
        all_files: List[str],
        split: str,
        train_ratio: float,
        min_timesteps: int,
        use_all: bool,
    ):
        """
        Shard 格式初始化：只读取每个 shard 的元信息（N/T/times），
        不加载完整的 embeddings Tensor，从而避免 init 时内存峰值。

        pairs 索引: (shard_file_idx, user_within_shard, i, j)
        """
        if use_all:
            files = all_files
        else:
            n_train = int(len(all_files) * train_ratio)
            files = all_files[:n_train] if split == "train" else all_files[n_train:]

        # shard 元信息缓存，延迟加载 embeddings
        self._shard_files: List[str] = []
        self._shard_meta: List[dict] = []   # [{N, T, times}, ...]
        self.pairs: List[Tuple[int, int, int, int]] = []  # (shard_idx, user_in_shard, i, j)

        n_users_total = 0

        for fpath in files:
            try:
                # 只加载非 embeddings 字段（times/user_ids/snap_date 都很小）
                data = torch.load(fpath, map_location="cpu", weights_only=False)
                emb_tensor = data["embeddings"]  # Tensor(N, T, dim)
                N, T, _ = emb_tensor.shape
                if T < min_timesteps:
                    logger.warning(f"跳过 {fpath}: T={T} < min_timesteps={min_timesteps}")
                    continue

                times = data.get("times", list(range(T)))
                shard_idx = len(self._shard_files)
                self._shard_files.append(fpath)
                self._shard_meta.append({"N": N, "T": T, "times": times})
                del data  # 释放，避免 init 阶段内存积压

                # 为该 shard 的每个用户建配对索引
                for u in range(N):
                    self.pairs.extend(
                        (shard_idx, u, i, j)
                        for i, j in self._iter_pairs(T)
                    )
                n_users_total += N

            except Exception as e:
                logger.debug(f"跳过 {fpath}: {e}")
                continue

        self._n_users = n_users_total
        self._is_shard = True
        # 懒加载缓存：当前已加载的 shard（仅缓存最近一个，多 worker 各自独立）
        self._cached_shard_idx: int = -1
        self._cached_shard_emb: Optional[torch.Tensor] = None   # Tensor(N, T, dim)

    # ── 初始化：逐用户格式 ──

    def _init_user(
        self,
        all_files: List[str],
        split: str,
        train_ratio: float,
        min_timesteps: int,
        use_all: bool,
    ):
        """逐用户格式（user_XXXXXX.pt）初始化，保持原有行为。"""
        if use_all:
            files = all_files
        else:
            n_train = int(len(all_files) * train_ratio)
            files = all_files[:n_train] if split == "train" else all_files[n_train:]

        self.files: List[str] = []
        self.times_cache: List[list] = []
        self.cached_data: List[dict] = []     # 非懒加载模式下缓存完整数据
        self.pairs: List[Tuple[int, int, int]] = []  # (file_idx, i, j)

        for fpath in files:
            try:
                data = torch.load(fpath, map_location="cpu", weights_only=False)
                T = data["embeddings"].size(0)
                if T < min_timesteps:
                    continue

                file_idx = len(self.files)
                self.files.append(fpath)
                self.times_cache.append(data.get("times", list(range(T))))
                if not self.lazy_load:
                    self.cached_data.append(data)

                self.pairs.extend(
                    (file_idx, i, j) for i, j in self._iter_pairs(T)
                )
            except Exception as e:
                logger.debug(f"跳过 {fpath}: {e}")
                continue

        self._n_users = len(self.files)
        self._is_shard = False

    # ── 配对迭代器 ──

    def _iter_pairs(self, T: int):
        """按 pair_mode 返回 (i, j) 索引对"""
        if self.pair_mode == "last_target":
            # 用 (-1, T-1) 作占位，__getitem__ 中运行时随机采样 i
            yield -1, T - 1
        elif self.pair_mode == "all_pairs":
            for i in range(T - 1):
                for j in range(i + 1, T):
                    yield i, j
        else:  # adjacent
            for i in range(T - 1):
                yield i, i + 1

    # ── 数据加载 ──

    def _load_shard(self, shard_idx: int) -> torch.Tensor:
        """懒加载 shard embeddings，缓存最近一个"""
        if self._cached_shard_idx != shard_idx:
            data = torch.load(
                self._shard_files[shard_idx],
                map_location="cpu",
                weights_only=False,
            )
            self._cached_shard_emb = data["embeddings"]   # Tensor(N, T, dim)
            self._cached_shard_idx = shard_idx
        return self._cached_shard_emb

    def _load_user(self, file_idx: int) -> dict:
        """取用户数据（懒加载或内存缓存）"""
        if self.lazy_load:
            return torch.load(
                self.files[file_idx], map_location="cpu", weights_only=False
            )
        return self.cached_data[file_idx]

    # ── Dataset 接口 ──

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        if self._is_shard:
            return self._getitem_shard(idx)
        else:
            return self._getitem_user(idx)

    def _getitem_shard(self, idx: int) -> dict:
        shard_idx, u, i, j = self.pairs[idx]
        shard_emb = self._load_shard(shard_idx)   # Tensor(N, T, dim)
        embeddings = shard_emb[u]                  # Tensor(T, dim)
        times = self._shard_meta[shard_idx]["times"]
        T = embeddings.size(0)

        if i < 0:
            i = torch.randint(0, T - 1, (1,)).item() if T > 1 else 0

        condition = embeddings[i].float()
        x_clean = embeddings[j].float()
        delta_t = compute_delta_t(times, i, j, self.delta_t_from_timestamp)

        return {
            "x_clean": x_clean,
            "condition": condition,
            "delta_t": delta_t,
            "step_gap": j - i,
            "num_timesteps": T,
        }

    def _getitem_user(self, idx: int) -> dict:
        file_idx, i, j = self.pairs[idx]
        data = self._load_user(file_idx)
        embeddings = data["embeddings"]
        times = self.times_cache[file_idx]
        T = embeddings.size(0)

        if i < 0:
            i = torch.randint(0, T - 1, (1,)).item() if T > 1 else 0

        condition = embeddings[i].float()
        x_clean = embeddings[j].float()
        delta_t = compute_delta_t(times, i, j, self.delta_t_from_timestamp)

        return {
            "x_clean": x_clean,
            "condition": condition,
            "delta_t": delta_t,
            "step_gap": j - i,
            "num_timesteps": T,
        }


# ────────────────────────────────────────────────────────────
# Collate / DataLoader
# ────────────────────────────────────────────────────────────

def collate_fn(batch: List[dict]) -> dict:
    return {
        "x_clean": torch.stack([b["x_clean"] for b in batch]),
        "condition": torch.stack([b["condition"] for b in batch]),
        "delta_t": torch.tensor([b["delta_t"] for b in batch], dtype=torch.float32),
        "step_gap": torch.tensor([b["step_gap"] for b in batch], dtype=torch.long),
        "num_timesteps": [b["num_timesteps"] for b in batch],
    }


def create_dataloader(
    data_dir: str,
    batch_size: int = 32,
    split: str = "train",
    train_ratio: float = 0.8,
    num_workers: int = 0,
    pair_mode: str = "last_target",
    delta_t_from_timestamp: bool = True,
    lazy_load: bool = True,
    prefetch_factor: int = 2,
    drop_last: bool = False,
    distributed: bool = False,
    use_all: bool = False,
) -> DataLoader:
    """
    创建 DataLoader。

    Args:
        num_workers:  并行 IO 进程数 (>0 时配合 lazy_load 显著提升吞吐)
        pair_mode:    配对策略
        distributed:  True 则使用 DistributedSampler (DDP 多卡)
        drop_last:    DDP 下建议 True，保证各卡 batch 一致
        use_all:      True 则加载 data_dir 全部文件，忽略 split/train_ratio
                      （搭配外部预分好的 train/test 目录使用）
    """
    dataset = UserEmbeddingDataset(
        data_dir,
        split=split,
        train_ratio=train_ratio,
        pair_mode=pair_mode,
        delta_t_from_timestamp=delta_t_from_timestamp,
        lazy_load=lazy_load,
        use_all=use_all,
    )

    sampler = None
    shuffle = split == "train"
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)
        shuffle = False   # sampler 与 shuffle 互斥

    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )
    # 这两个参数仅在 num_workers > 0 时有效
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(dataset, **kwargs)

"""
data_loader.py — 用户 Embedding 时间序列数据加载
=================================================
支持三种配对策略 (pair_mode):

| 模式          | 说明                        | T=4 时样本数 |
|--------------|----------------------------|------------|
| last_target  | 随机早期步 → 最后一步 (默认)  | 1          |
| all_pairs    | 枚举所有 (i,j) i<j          | 6          |
| adjacent     | 相邻配对 (i, i+1)           | 3          |

同时输出 delta_t (归一化时间跨度)，用于注入网络修复训推不一致:
  - 训练时模型能感知 "预测多远的未来"
  - 推理时 delta_t 语义与训练一致

性能优化:
  - lazy_load: 只在 __init__ 扫描元信息, __getitem__ 时才 torch.load
                配合 num_workers>0 实现并行 IO, 避免主进程成为瓶颈
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

    每个 .pt 内含:
        embeddings: (T, dim) 各时间步融合 embedding
        times:      [str] × T 时间戳
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
    ):
        if pair_mode not in ("last_target", "all_pairs", "adjacent"):
            raise ValueError(
                f"Unknown pair_mode: {pair_mode}, "
                f"choose from last_target/all_pairs/adjacent"
            )

        self.pair_mode = pair_mode
        self.delta_t_from_timestamp = delta_t_from_timestamp
        self.lazy_load = lazy_load

        all_files = sorted(
            os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")
        )
        # 排除 fusion_weights.pt 等非用户数据
        all_files = [f for f in all_files if os.path.basename(f).startswith("user_")]

        n_train = int(len(all_files) * train_ratio)
        files = all_files[:n_train] if split == "train" else all_files[n_train:]

        # 扫描元信息 (T 和 times)，同时构建配对索引
        self.files: List[str] = []
        self.times_cache: List[list] = []
        self.cached_data: List[dict] = []          # 非懒加载模式下缓存完整数据
        self.pairs: List[Tuple[int, int, int]] = []  # (file_idx, i, j)

        for fpath in files:
            try:
                data = torch.load(fpath, map_location="cpu")
                T = data["embeddings"].size(0)
                if T < min_timesteps:
                    continue

                file_idx = len(self.files)
                self.files.append(fpath)
                self.times_cache.append(data.get("times", list(range(T))))
                if not lazy_load:
                    self.cached_data.append(data)

                self.pairs.extend(self._build_pairs(file_idx, T))
            except Exception as e:
                logger.debug(f"跳过 {fpath}: {e}")
                continue

        logger.info(
            f"[{split}] 用户数={len(self.files)}, 样本对数={len(self.pairs)}, "
            f"pair_mode={pair_mode}, lazy_load={lazy_load}"
        )

    def _build_pairs(self, file_idx: int, T: int) -> List[Tuple[int, int, int]]:
        """按 pair_mode 构建该用户的配对索引"""
        if self.pair_mode == "last_target":
            # 单条占位, 实际 cond_idx 在 __getitem__ 中随机 (保持原行为)
            return [(file_idx, -1, T - 1)]

        if self.pair_mode == "all_pairs":
            return [(file_idx, i, j) for i in range(T - 1) for j in range(i + 1, T)]

        # adjacent
        return [(file_idx, i, i + 1) for i in range(T - 1)]

    def _load(self, file_idx: int) -> dict:
        """取用户数据 (懒加载或缓存)"""
        if self.lazy_load:
            return torch.load(self.files[file_idx], map_location="cpu")
        return self.cached_data[file_idx]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        file_idx, i, j = self.pairs[idx]
        data = self._load(file_idx)
        embeddings = data["embeddings"]
        times = self.times_cache[file_idx]
        T = embeddings.size(0)

        # last_target 模式: i = -1 占位, 运行时随机采样条件步
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
) -> DataLoader:
    """
    创建 DataLoader。

    Args:
        num_workers: 并行 IO 进程数 (>0 时配合 lazy_load 显著提升吞吐)
        pair_mode: 配对策略
        distributed: True 则使用 DistributedSampler (DDP 多卡)
        drop_last: DDP 下建议 True, 保证各卡 batch 一致
    """
    dataset = UserEmbeddingDataset(
        data_dir,
        split=split,
        train_ratio=train_ratio,
        pair_mode=pair_mode,
        delta_t_from_timestamp=delta_t_from_timestamp,
        lazy_load=lazy_load,
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

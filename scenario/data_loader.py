"""
data_loader.py — 数据加载
===========================
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional

from common.logger import get_logger

logger = get_logger(__name__)


class UserEmbeddingDataset(Dataset):
    """从预保存 .pt 文件加载用户 embedding 时间序列"""

    def __init__(self, data_dir: str, split: str = 'train', train_ratio: float = 0.8):
        self.data = []
        all_files = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pt')
        ])
        n_total = len(all_files)
        n_train = int(n_total * train_ratio)

        files = all_files[:n_train] if split == 'train' else all_files[n_train:]

        for fpath in files:
            try:
                data = torch.load(fpath, map_location='cpu')
                if data['embeddings'].size(0) >= 2:
                    self.data.append(data)
            except Exception:
                continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        embeddings = item['embeddings']
        times = item['times']
        T = embeddings.size(0)

        x_clean = embeddings[-1].float()
        if T > 1:
            cond_idx = torch.randint(0, T - 1, (1,)).item()
            condition = embeddings[cond_idx].float()
        else:
            cond_idx = 0
            condition = x_clean

        return {
            'x_clean': x_clean,
            'condition': condition,
            'num_timesteps': T,
        }


def collate_fn(batch: list[dict]) -> dict:
    x_clean = torch.stack([item['x_clean'] for item in batch])
    condition = torch.stack([item['condition'] for item in batch])
    return {
        'x_clean': x_clean,
        'condition': condition,
        'num_timesteps': [item['num_timesteps'] for item in batch],
    }


def create_dataloader(
    data_dir: str,
    batch_size: int = 32,
    split: str = 'train',
    train_ratio: float = 0.8,
    num_workers: int = 0,
) -> DataLoader:
    dataset = UserEmbeddingDataset(data_dir, split=split, train_ratio=train_ratio)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=(split == 'train'),
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )

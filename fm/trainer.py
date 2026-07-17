"""
trainer.py — Flow Matching 训练器
==================================
"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from common.logger import get_logger
from common.config import FMConfig
from common.utils import save_checkpoint, count_parameters
from .model import FlowMatchingModel
from scenario.data_loader import create_dataloader

logger = get_logger(__name__)


class FMTrainer:
    """Flow Matching Teacher 训练器"""

    def __init__(self, config: FMConfig, device: str = 'cuda:0'):
        self.config = config
        self.device = device

        self.model = FlowMatchingModel(
            data_dim=config.data_dim,
            time_dim=config.time_dim,
            cond_dim=config.cond_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        ).to(device)

        logger.info(f"FM Teacher 参数量: {count_parameters(self.model):,}")

    def train(
        self,
        data_dir: str,
        output_dir: str,
        train_ratio: float = 0.8,
    ):
        """执行训练"""
        os.makedirs(output_dir, exist_ok=True)

        train_loader = create_dataloader(data_dir, batch_size=self.config.batch_size,
                                          split='train', train_ratio=train_ratio)
        val_loader = create_dataloader(data_dir, batch_size=self.config.batch_size,
                                       split='val', train_ratio=train_ratio)

        logger.info(f"训练集: {len(train_loader.dataset)}, 验证集: {len(val_loader.dataset)}")

        optimizer = AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs, eta_min=self.config.lr * 0.01)

        best_val = float('inf')
        log = []

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(train_loader, optimizer)
            val_loss = self._validate(val_loader) if len(val_loader.dataset) > 0 else 0.0
            scheduler.step()

            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            logger.info(
                f"Epoch {epoch}/{self.config.epochs} | "
                f"train={train_loss:.6f} val={val_loss:.6f} | "
                f"lr={lr:.2e} | {elapsed:.1f}s"
            )
            log.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'lr': lr})

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    os.path.join(output_dir, 'best_fm.pt'), self.model, optimizer,
                    epoch=epoch, loss=val_loss, extra={'config': vars(self.config)}
                )

            if epoch % self.config.save_every == 0:
                save_checkpoint(
                    os.path.join(output_dir, f'fm_epoch_{epoch}.pt'), self.model, optimizer,
                    epoch=epoch, loss=val_loss
                )

        with open(os.path.join(output_dir, 'fm_train_log.json'), 'w') as f:
            json.dump(log, f, indent=2)
        logger.info(f"FM 训练完成, 最佳 val={best_val:.6f}")

    def _train_epoch(self, loader, optimizer) -> float:
        self.model.train()
        total = 0.0
        n = 0
        for batch in loader:
            x_clean = batch['x_clean'].to(self.device)
            cond = batch['condition'].to(self.device)
            loss_dict = self.model.compute_loss(x_clean, cond)
            optimizer.zero_grad()
            loss_dict['loss'].backward()
            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            optimizer.step()
            total += loss_dict['mse']
            n += 1
        return total / max(n, 1)

    @torch.no_grad()
    def _validate(self, loader) -> float:
        self.model.eval()
        total = 0.0
        n = 0
        for batch in loader:
            x_clean = batch['x_clean'].to(self.device)
            cond = batch['condition'].to(self.device)
            loss_dict = self.model.compute_loss(x_clean, cond)
            total += loss_dict['mse']
            n += 1
        return total / max(n, 1)

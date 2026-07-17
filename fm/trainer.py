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

from torch.utils.tensorboard import SummaryWriter

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

        # TensorBoard
        tb_dir = os.path.join(output_dir, 'tensorboard')
        writer = SummaryWriter(log_dir=tb_dir)
        logger.info(f"TensorBoard 日志目录: {tb_dir}")

        train_loader = create_dataloader(data_dir, batch_size=self.config.batch_size,
                                          split='train', train_ratio=train_ratio)
        val_loader = create_dataloader(data_dir, batch_size=self.config.batch_size,
                                       split='val', train_ratio=train_ratio)

        logger.info(f"训练集: {len(train_loader.dataset)}, 验证集: {len(val_loader.dataset)}")

        optimizer = AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs, eta_min=self.config.lr * 0.01)

        best_val = float('inf')
        log = []
        global_step = 0

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            train_loss, train_mse, train_steps = self._train_epoch(train_loader, optimizer, writer, global_step)
            val_loss, val_cos = self._validate(val_loader) if len(val_loader.dataset) > 0 else (0.0, 0.0)
            scheduler.step()
            global_step += train_steps

            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]

            logger.info(
                f"Epoch {epoch}/{self.config.epochs} | "
                f"train={train_loss:.6f} val={val_loss:.6f} | "
                f"val_cos_sim={val_cos:.4f} | "
                f"lr={lr:.2e} | {elapsed:.1f}s"
            )

            # TensorBoard epoch-level
            writer.add_scalar('epoch/train_loss', train_loss, epoch)
            writer.add_scalar('epoch/val_loss', val_loss, epoch)
            writer.add_scalar('epoch/val_cosine_sim', val_cos, epoch)
            writer.add_scalar('epoch/lr', lr, epoch)

            log.append({
                'epoch': epoch, 'train_loss': train_loss,
                'val_loss': val_loss, 'val_cos_sim': val_cos, 'lr': lr,
            })

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    os.path.join(output_dir, 'best_fm.pt'), self.model, optimizer,
                    epoch=epoch, loss=val_loss, extra={'config': vars(self.config)}
                )
                logger.info(f"  ★ 新最佳模型 val={best_val:.6f}")

            if epoch % self.config.save_every == 0:
                save_checkpoint(
                    os.path.join(output_dir, f'fm_epoch_{epoch}.pt'), self.model, optimizer,
                    epoch=epoch, loss=val_loss
                )

        writer.close()
        with open(os.path.join(output_dir, 'fm_train_log.json'), 'w') as f:
            json.dump(log, f, indent=2)
        logger.info(f"FM 训练完成, 最佳 val={best_val:.6f}")
        logger.info(f"查看 TensorBoard: tensorboard --logdir {tb_dir}")

    def _train_epoch(self, loader, optimizer, writer, step_offset) -> tuple:
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
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

            total_loss += loss_dict['loss'].item()
            total_mse += loss_dict['mse']
            n += 1

            # TensorBoard step-level
            global_step = step_offset + n
            writer.add_scalar('step/loss', loss_dict['loss'].item(), global_step)
            writer.add_scalar('step/mse', loss_dict['mse'], global_step)

        avg_loss = total_loss / max(n, 1)
        avg_mse = total_mse / max(n, 1)
        return avg_loss, avg_mse, n

    @torch.no_grad()
    def _validate(self, loader) -> tuple:
        self.model.eval()
        import torch.nn.functional as F
        total_loss = 0.0
        total_cos = 0.0
        n = 0
        for batch in loader:
            x_clean = batch['x_clean'].to(self.device)
            cond = batch['condition'].to(self.device)

            loss_dict = self.model.compute_loss(x_clean, cond)
            total_loss += loss_dict['mse']

            # 额外计算 cosine similarity
            pred = self.model.predict_future_embedding(cond, delta_t=0.5, num_steps=50)
            cos = F.cosine_similarity(pred, x_clean, dim=-1).mean().item()
            total_cos += cos

            n += 1

        avg_loss = total_loss / max(n, 1)
        avg_cos = total_cos / max(n, 1)
        return avg_loss, avg_cos

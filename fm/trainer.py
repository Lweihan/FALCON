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
            # MLP 参数
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            # Transformer 参数
            backbone_type=config.backbone_type,
            transformer_dim=config.transformer_dim,
            transformer_depth=config.transformer_depth,
            transformer_heads=config.transformer_heads,
            transformer_mlp_ratio=config.transformer_mlp_ratio,
            transformer_dropout=config.transformer_dropout,
        ).to(device)

        total_params = count_parameters(self.model)
        vel_params = count_parameters(self.model.velocity_net)
        logger.info(f"FM Teacher 初始化:")
        logger.info(f"  Backbone: {config.backbone_type}")
        logger.info(f"  总参数量: {total_params:,}")
        logger.info(f"  速度网络参数量: {vel_params:,}")
        logger.info(f"  时间加权: mode={config.time_weight_mode}, scale={config.time_weight_scale}")
        logger.info(f"  EMA loss alpha: {config.ema_loss_alpha}")

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

        # 记录元信息
        writer.add_text('meta/backbone_type', self.config.backbone_type, 0)
        writer.add_scalar('meta/param_count_total', count_parameters(self.model), 0)
        writer.add_scalar('meta/param_count_velocity_net', count_parameters(self.model.velocity_net), 0)

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

        # EMA loss 跟踪
        ema_train_loss = None
        ema_val_loss = None
        alpha = self.config.ema_loss_alpha

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            train_stats, train_steps = self._train_epoch(train_loader, optimizer, writer, global_step)
            val_stats = self._validate(val_loader) if len(val_loader.dataset) > 0 else {}
            scheduler.step()
            global_step += train_steps

            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]

            # 提取统计量
            train_loss = train_stats.get('avg_loss', 0.0)
            train_mse_raw = train_stats.get('avg_mse_raw', 0.0)
            train_mse_weighted = train_stats.get('avg_mse_weighted', 0.0)
            val_loss = val_stats.get('avg_loss', 0.0)
            val_cos = val_stats.get('avg_cos_sim', 0.0)

            # 更新 EMA
            if ema_train_loss is None:
                ema_train_loss = train_loss
                ema_val_loss = val_loss
            else:
                ema_train_loss = alpha * train_loss + (1 - alpha) * ema_train_loss
                ema_val_loss = alpha * val_loss + (1 - alpha) * ema_val_loss

            logger.info(
                f"Epoch {epoch}/{self.config.epochs} | "
                f"train={train_loss:.6f} (ema={ema_train_loss:.6f}) | "
                f"val={val_loss:.6f} (ema={ema_val_loss:.6f}) | "
                f"cos_sim={val_cos:.4f} | "
                f"mse_raw={train_mse_raw:.6f} mse_w={train_mse_weighted:.6f} | "
                f"lr={lr:.2e} | {elapsed:.1f}s"
            )

            # TensorBoard epoch-level
            writer.add_scalar('epoch/train_loss', train_loss, epoch)
            writer.add_scalar('epoch/val_loss', val_loss, epoch)
            writer.add_scalar('epoch/ema_train_loss', ema_train_loss, epoch)
            writer.add_scalar('epoch/ema_val_loss', ema_val_loss, epoch)
            writer.add_scalar('epoch/val_cosine_sim', val_cos, epoch)
            writer.add_scalar('epoch/lr', lr, epoch)
            writer.add_scalar('epoch/mse_raw', train_mse_raw, epoch)
            writer.add_scalar('epoch/mse_weighted', train_mse_weighted, epoch)

            log.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'ema_train_loss': ema_train_loss,
                'val_loss': val_loss,
                'ema_val_loss': ema_val_loss,
                'val_cos_sim': val_cos,
                'mse_raw': train_mse_raw,
                'mse_weighted': train_mse_weighted,
                'lr': lr,
            })

            # 用 EMA val loss 判断最佳模型
            if ema_val_loss < best_val:
                best_val = ema_val_loss
                save_checkpoint(
                    os.path.join(output_dir, 'best_fm.pt'), self.model, optimizer,
                    epoch=epoch, loss=val_loss, extra={'ema_val_loss': ema_val_loss}
                )
                logger.info(f"  ★ 新最佳模型 (ema_val={best_val:.6f})")

            if epoch % self.config.save_every == 0:
                save_checkpoint(
                    os.path.join(output_dir, f'fm_epoch_{epoch}.pt'), self.model, optimizer,
                    epoch=epoch, loss=val_loss
                )

        writer.close()
        with open(os.path.join(output_dir, 'fm_train_log.json'), 'w') as f:
            json.dump(log, f, indent=2)
        logger.info(f"FM 训练完成, 最佳 ema_val={best_val:.6f}")
        logger.info(f"查看 TensorBoard: tensorboard --logdir {tb_dir}")

    def _train_epoch(self, loader, optimizer, writer, step_offset) -> tuple:
        self.model.train()
        total_loss = 0.0
        total_mse_raw = 0.0
        total_mse_weighted = 0.0
        total_time_weight = 0.0
        total_t_mean = 0.0
        n = 0
        for batch in loader:
            x_clean = batch['x_clean'].to(self.device)
            cond = batch['condition'].to(self.device)
            loss_dict = self.model.compute_loss(
                x_clean, cond,
                time_weight_mode=self.config.time_weight_mode,
                time_weight_scale=self.config.time_weight_scale,
            )

            optimizer.zero_grad()
            loss_dict['loss'].backward()
            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            optimizer.step()

            total_loss += loss_dict['loss'].item()
            total_mse_raw += loss_dict['mse_raw']
            total_mse_weighted += loss_dict['mse_weighted']
            total_time_weight += loss_dict['time_weight_mean']
            total_t_mean += loss_dict['t_mean']
            n += 1

            # TensorBoard step-level
            global_step = step_offset + n
            writer.add_scalar('step/loss', loss_dict['loss'].item(), global_step)
            writer.add_scalar('step/mse_raw', loss_dict['mse_raw'], global_step)
            writer.add_scalar('step/mse_weighted', loss_dict['mse_weighted'], global_step)
            writer.add_scalar('step/time_weight_mean', loss_dict['time_weight_mean'], global_step)
            writer.add_scalar('step/t_mean', loss_dict['t_mean'], global_step)

        stats = {
            'avg_loss': total_loss / max(n, 1),
            'avg_mse_raw': total_mse_raw / max(n, 1),
            'avg_mse_weighted': total_mse_weighted / max(n, 1),
            'avg_time_weight': total_time_weight / max(n, 1),
            'avg_t_mean': total_t_mean / max(n, 1),
        }
        return stats, n

    @torch.no_grad()
    def _validate(self, loader) -> dict:
        self.model.eval()
        import torch.nn.functional as F
        total_loss = 0.0
        total_cos = 0.0
        n = 0
        for batch in loader:
            x_clean = batch['x_clean'].to(self.device)
            cond = batch['condition'].to(self.device)

            loss_dict = self.model.compute_loss(
                x_clean, cond,
                time_weight_mode='none',  # 验证时不用加权
            )
            total_loss += loss_dict['mse_raw']

            # cosine similarity
            pred = self.model.predict_future_embedding(cond, delta_t=0.5, num_steps=50)
            cos = F.cosine_similarity(pred, x_clean, dim=-1).mean().item()
            total_cos += cos

            n += 1

        return {
            'avg_loss': total_loss / max(n, 1),
            'avg_cos_sim': total_cos / max(n, 1),
        }

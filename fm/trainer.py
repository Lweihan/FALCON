"""
trainer.py — Flow Matching 训练器
==================================
支持:
- DDP 多卡训练 (torchrun 启动)
- AMP 混合精度 + 梯度累积
- 验证阶段优化 (可调 ODE 步数 / 频率 / batch 数)
- TF32 / cudnn benchmark / torch.compile
- delta_t 条件注入
- 三种配对策略
"""

import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from common.config import FMConfig
from common.distributed import (
    barrier, get_local_rank, get_world_size, is_main_process,
    reduce_value, setup_performance_flags, unwrap_model,
)
from common.logger import get_logger
from common.utils import count_parameters, save_checkpoint, make_grad_scaler, autocast
from scenario.data_loader import create_dataloader
from .model import FlowMatchingModel

logger = get_logger(__name__)


class _NullWriter:
    """非主进程使用的空 writer，屏蔽所有 TensorBoard 调用"""

    def add_scalar(self, *a, **k): pass
    def add_text(self, *a, **k): pass
    def close(self): pass


class FMTrainer:
    """Flow Matching Teacher 训练器"""

    def __init__(
        self,
        config: FMConfig,
        device: str = "cuda:0",
        distributed: bool = False,
    ):
        self.config = config
        self.device = device
        self.distributed = distributed
        self.world_size = get_world_size()
        self.is_main = is_main_process()

        # 性能开关 (TF32 / cudnn benchmark)
        setup_performance_flags(config.use_tf32, config.cudnn_benchmark)

        model = FlowMatchingModel.from_config(config).to(device)

        # torch.compile (可选)
        if config.use_compile:
            try:
                model = torch.compile(model)
                self._log("已启用 torch.compile (首个 epoch 会较慢)")
            except Exception as e:
                self._log(f"torch.compile 启用失败, 已跳过: {e}", level="warning")

        # DDP 包装
        if distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[get_local_rank()] if torch.cuda.is_available() else None,
                output_device=get_local_rank() if torch.cuda.is_available() else None,
                find_unused_parameters=False,   # DiT 无条件分支, 关闭可提速
            )

        self.model = model
        self.raw_model = unwrap_model(model)

        self._log_init_info()

    # ── 日志辅助 ──

    def _log(self, msg: str, level: str = "info"):
        """只在主进程打印"""
        if self.is_main:
            getattr(logger, level)(msg)

    def _log_init_info(self):
        cfg = self.config
        total = count_parameters(self.raw_model)
        vel = count_parameters(self.raw_model.velocity_net)
        dit = cfg.resolve_dit_params()

        self._log("FM Teacher 初始化:")
        self._log(f"  Backbone: {cfg.backbone_type}")
        if cfg.backbone_type == "dit":
            self._log(f"  DiT preset: {cfg.dit_preset}")
            self._log(f"    dim={dit['dit_dim']}, depth={dit['dit_depth']}, "
                      f"heads={dit['dit_heads']}, patches={dit['dit_num_patches']}")
            self._log(f"    qk_norm={cfg.dit_qk_norm}, rope={cfg.dit_use_rope}, "
                      f"grad_ckpt={cfg.use_gradient_checkpointing}")
        self._log(f"  参数量: 总 {total:,} ({total / 1e6:.1f}M), "
                  f"速度网络 {vel:,} ({vel / 1e6:.1f}M)")
        self._log(f"  配对策略: {cfg.pair_mode}, delta_t 条件注入: {cfg.use_delta_t_condition}")
        self._log(f"  时间加权: {cfg.time_weight_mode} (scale={cfg.time_weight_scale})")
        self._log("  性能: "
                  f"AMP={cfg.use_amp}, TF32={cfg.use_tf32}, compile={cfg.use_compile}, "
                  f"grad_accum={cfg.grad_accum_steps}")
        if self.distributed:
            self._log(f"  分布式: DDP, world_size={self.world_size}")
            self._log(f"  Batch: per_gpu={cfg.batch_size}, "
                      f"global={cfg.batch_size * self.world_size * cfg.grad_accum_steps}")
        else:
            self._log(f"  Batch: {cfg.batch_size} "
                      f"(effective={cfg.batch_size * cfg.grad_accum_steps})")

    # ── 训练主流程 ──

    def train(
        self,
        data_dir: str,
        output_dir: str,
        train_ratio: float = 0.8,
        val_data_dir: str = None,
    ):
        cfg = self.config

        if self.is_main:
            os.makedirs(output_dir, exist_ok=True)
        barrier()

        tb_dir = os.path.join(output_dir, "tensorboard")
        writer = SummaryWriter(log_dir=tb_dir) if self.is_main else _NullWriter()
        self._log(f"TensorBoard: {tb_dir}")

        loader_kwargs = dict(
            batch_size=cfg.batch_size,
            train_ratio=train_ratio,
            num_workers=cfg.num_workers,
            pair_mode=cfg.pair_mode,
            delta_t_from_timestamp=cfg.delta_t_from_timestamp,
            lazy_load=cfg.lazy_load,
            prefetch_factor=cfg.prefetch_factor,
            distributed=self.distributed,
        )
        train_loader = create_dataloader(
            data_dir, split="train", drop_last=self.distributed,
            use_all=(val_data_dir is not None),
            **loader_kwargs,
        )
        if val_data_dir is not None:
            # 外部预分好的 test 目录：全量加载，不 shuffle，不 drop_last
            val_loader = create_dataloader(
                val_data_dir, split="val", drop_last=False,
                use_all=True,
                **loader_kwargs,
            )
            if is_main_process():
                logger.info(f"验证集使用独立目录: {val_data_dir}")
        else:
            val_loader = create_dataloader(
                data_dir, split="val", drop_last=False,
                use_all=False,
                **loader_kwargs,
            )

        n_train, n_val = len(train_loader.dataset), len(val_loader.dataset)
        self._log(f"训练样本对: {n_train}, 验证样本对: {n_val}")

        # 元信息
        dit = cfg.resolve_dit_params()
        writer.add_text("meta/backbone_type", cfg.backbone_type, 0)
        writer.add_text("meta/pair_mode", cfg.pair_mode, 0)
        if cfg.backbone_type == "dit":
            writer.add_text("meta/dit_preset", cfg.dit_preset, 0)
            writer.add_scalar("meta/dit_dim", dit["dit_dim"], 0)
            writer.add_scalar("meta/dit_depth", dit["dit_depth"], 0)
        writer.add_scalar("meta/param_count_total", count_parameters(self.raw_model), 0)
        writer.add_scalar("meta/num_train_pairs", n_train, 0)
        writer.add_scalar("meta/world_size", self.world_size, 0)
        writer.add_scalar("meta/global_batch_size",
                          cfg.batch_size * self.world_size * cfg.grad_accum_steps, 0)

        optimizer = AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)

        use_amp = cfg.use_amp and "cuda" in str(self.device)
        scaler = make_grad_scaler(use_amp)

        best_val = float("inf")
        ema_train = ema_val = None
        alpha = cfg.ema_loss_alpha
        log_history = []
        global_step = 0

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()

            # DDP: 每 epoch 重设 sampler seed 保证 shuffle 有效
            if self.distributed and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            train_stats, steps = self._train_epoch(
                train_loader, optimizer, writer, global_step, scaler
            )
            global_step += steps

            # 按 val_every 控制验证频率
            do_val = (epoch % cfg.val_every == 0) or (epoch == cfg.epochs)
            val_stats = self._validate(val_loader) if (do_val and n_val > 0) else {}

            scheduler.step()
            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]

            train_loss = train_stats["avg_loss"]
            has_val = bool(val_stats)           # 本 epoch 是否真的跑了验证
            val_loss = val_stats.get("avg_loss") if has_val else None
            val_cos = val_stats.get("avg_cos_sim") if has_val else None

            # EMA train loss: 每 epoch 更新
            if ema_train is None:
                ema_train = train_loss
            else:
                ema_train = alpha * train_loss + (1 - alpha) * ema_train

            # EMA val loss: 仅在真正做了验证时更新, 否则沿用上一次值
            if val_loss is not None:
                if ema_val is None:
                    ema_val = val_loss
                else:
                    ema_val = alpha * val_loss + (1 - alpha) * ema_val

            # 日志: val/cos 仅在有验证时打印
            if has_val:
                self._log(
                    f"Epoch {epoch}/{cfg.epochs} ({elapsed:.1f}s) | "
                    f"train={train_loss:.6f} (ema={ema_train:.6f}) | "
                    f"val={val_loss:.6f} (ema={ema_val:.6f}) | "
                    f"cos={val_cos:.4f} | lr={lr:.2e}"
                )
            else:
                self._log(
                    f"Epoch {epoch}/{cfg.epochs} ({elapsed:.1f}s) | "
                    f"train={train_loss:.6f} (ema={ema_train:.6f}) | "
                    f"[val skipped] | lr={lr:.2e}"
                )

            # TensorBoard: train 每 epoch 写; val/cos 仅有数据时写, 避免空值点
            writer.add_scalar("epoch/train_loss", train_loss, epoch)
            writer.add_scalar("epoch/ema_train_loss", ema_train, epoch)
            writer.add_scalar("epoch/lr", lr, epoch)
            writer.add_scalar("epoch/mse_raw", train_stats["avg_mse_raw"], epoch)
            writer.add_scalar("epoch/mse_weighted", train_stats["avg_mse_weighted"], epoch)
            writer.add_scalar("epoch/delta_t_mean", train_stats["avg_delta_t"], epoch)
            writer.add_scalar("perf/epoch_seconds", elapsed, epoch)
            writer.add_scalar("perf/samples_per_sec",
                              n_train / max(elapsed, 1e-6), epoch)
            if has_val:
                writer.add_scalar("epoch/val_loss", val_loss, epoch)
                writer.add_scalar("epoch/val_cosine_sim", val_cos, epoch)
                if ema_val is not None:
                    writer.add_scalar("epoch/ema_val_loss", ema_val, epoch)

            log_history.append({
                "epoch": epoch, "train_loss": train_loss, "ema_train_loss": ema_train,
                "val_loss": val_loss, "ema_val_loss": ema_val, "val_cos_sim": val_cos,
                "mse_raw": train_stats["avg_mse_raw"],
                "mse_weighted": train_stats["avg_mse_weighted"],
                "delta_t_mean": train_stats["avg_delta_t"],
                "lr": lr, "seconds": elapsed,
            })

            # 保存 (仅主进程); 只在有真实验证值时才判断是否为最佳
            if self.is_main:
                if has_val and ema_val is not None and ema_val < best_val:
                    best_val = ema_val
                    save_checkpoint(
                        os.path.join(output_dir, "best_fm.pt"),
                        self.raw_model, optimizer,
                        epoch=epoch, loss=val_loss,
                        extra={"ema_val_loss": ema_val, "config": vars(cfg)},
                    )
                    self._log(f"  * 新最佳 (ema_val={best_val:.6f})")

                if epoch % cfg.save_every == 0:
                    save_checkpoint(
                        os.path.join(output_dir, f"fm_epoch_{epoch}.pt"),
                        self.raw_model, optimizer, epoch=epoch, loss=train_loss,
                    )
            barrier()

        writer.close()
        if self.is_main:
            with open(os.path.join(output_dir, "fm_train_log.json"), "w") as f:
                json.dump(log_history, f, indent=2)
            self._log(f"训练完成, 最佳 ema_val={best_val:.6f}")
            self._log(f"TensorBoard: tensorboard --logdir {tb_dir}")

    # ── 单 epoch 训练 ──

    def _train_epoch(self, loader, optimizer, writer, step_offset, scaler=None) -> tuple:
        cfg = self.config
        self.model.train()
        use_amp = scaler is not None
        accum = max(cfg.grad_accum_steps, 1)

        totals = {"loss": 0.0, "mse_raw": 0.0, "mse_weighted": 0.0,
                  "time_weight": 0.0, "t_mean": 0.0, "delta_t": 0.0}
        n = 0

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(loader):
            x_clean = batch["x_clean"].to(self.device, non_blocking=True)
            cond = batch["condition"].to(self.device, non_blocking=True)
            delta_t = (
                batch["delta_t"].to(self.device, non_blocking=True)
                if cfg.use_delta_t_condition else None
            )

            is_accum_step = (batch_idx + 1) % accum != 0

            # DDP + 梯度累积: 累积中间步跳过梯度同步以提速
            sync_ctx = (
                self.model.no_sync()
                if (self.distributed and is_accum_step)
                else _nullcontext()
            )

            with sync_ctx:
                if use_amp:
                    with autocast():
                        loss_dict = self._compute_loss(x_clean, cond, delta_t)
                    scaler.scale(loss_dict["loss"] / accum).backward()
                else:
                    loss_dict = self._compute_loss(x_clean, cond, delta_t)
                    (loss_dict["loss"] / accum).backward()

            if not is_accum_step:
                if use_amp:
                    if cfg.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if cfg.grad_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            totals["loss"] += loss_dict["loss"].item()
            totals["mse_raw"] += loss_dict["mse_raw"]
            totals["mse_weighted"] += loss_dict["mse_weighted"]
            totals["time_weight"] += loss_dict["time_weight_mean"]
            totals["t_mean"] += loss_dict["t_mean"]
            totals["delta_t"] += (
                batch["delta_t"].mean().item() if "delta_t" in batch else 0.0
            )
            n += 1

            step = step_offset + n
            writer.add_scalar("step/loss", loss_dict["loss"].item(), step)
            writer.add_scalar("step/mse_raw", loss_dict["mse_raw"], step)
            writer.add_scalar("step/mse_weighted", loss_dict["mse_weighted"], step)
            writer.add_scalar("step/t_mean", loss_dict["t_mean"], step)
            if "delta_t" in batch:
                writer.add_scalar("step/delta_t_mean", batch["delta_t"].mean().item(), step)

        d = max(n, 1)
        stats = {
            "avg_loss": reduce_value(totals["loss"] / d, device=self.device),
            "avg_mse_raw": reduce_value(totals["mse_raw"] / d, device=self.device),
            "avg_mse_weighted": reduce_value(totals["mse_weighted"] / d, device=self.device),
            "avg_time_weight": totals["time_weight"] / d,
            "avg_t_mean": totals["t_mean"] / d,
            "avg_delta_t": totals["delta_t"] / d,
        }
        return stats, n

    def _compute_loss(self, x_clean, cond, delta_t):
        """
        统一走 self.model(...) 调用 forward。

        DDP 下这是必须的: 只有经过 DDP wrapper 的 forward 才会注册
        梯度同步 hook，否则多卡梯度不会 all_reduce。
        """
        cfg = self.config
        return self.model(
            x_clean, cond,
            time_weight_mode=cfg.time_weight_mode,
            time_weight_scale=cfg.time_weight_scale,
            delta_t=delta_t,
        )

    # ── 验证 ──

    @torch.no_grad()
    def _validate(self, loader) -> dict:
        """
        验证阶段。

        GPU 利用率优化:
        - val_ode_steps 控制 ODE 步数 (默认 10, 原 50)
        - val_max_batches 限制 batch 数
        - val_compute_cos_sim=False 则完全跳过 ODE, 验证极快
        """
        cfg = self.config
        self.model.eval()

        total_loss = total_cos = 0.0
        n = 0

        for batch_idx, batch in enumerate(loader):
            if cfg.val_max_batches > 0 and batch_idx >= cfg.val_max_batches:
                break

            x_clean = batch["x_clean"].to(self.device, non_blocking=True)
            cond = batch["condition"].to(self.device, non_blocking=True)
            delta_t = (
                batch["delta_t"].to(self.device, non_blocking=True)
                if cfg.use_delta_t_condition else None
            )

            loss_dict = self.raw_model.compute_loss(
                x_clean, cond, time_weight_mode="none", delta_t=delta_t
            )
            total_loss += loss_dict["mse_raw"]

            # ODE 求解是串行的, 是验证阶段主要开销
            if cfg.val_compute_cos_sim:
                mean_dt = float(batch["delta_t"].mean().item()) if "delta_t" in batch else 0.5
                pred = self.raw_model.predict_future_embedding(
                    cond, delta_t=mean_dt, num_steps=cfg.val_ode_steps,
                    inject_delta_t=cfg.use_delta_t_condition,
                )
                total_cos += F.cosine_similarity(pred, x_clean, dim=-1).mean().item()

            n += 1

        d = max(n, 1)
        return {
            "avg_loss": reduce_value(total_loss / d, device=self.device),
            "avg_cos_sim": reduce_value(total_cos / d, device=self.device),
        }


# ────────────────────────────────────────────────────────────
# 辅助
# ────────────────────────────────────────────────────────────

class _nullcontext:
    """空上下文管理器 (用于 DDP 梯度累积时的条件 no_sync)"""
    def __enter__(self): return None
    def __exit__(self, *a): return False

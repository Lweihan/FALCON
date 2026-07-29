"""
trainer.py — Ranker 联合训练器
================================
训练 JointFALCONRanker，对比「直接预测」与「FALCON 优化后预测」的效果。
"""

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from common.logger import get_logger
from .joint_falcon_ranker import JointFALCONRanker
from .metrics import compute_lift, evaluate_predictions
from .models import create_ranker

logger = get_logger(__name__)


class RankerTrainer:
    """
    下游 Ranker 联合训练器。

    支持多模型 × 多 fm_weight 的网格实验，输出对比报告。
    """

    def __init__(
        self,
        ranker_config,
        fm_config,
        fm_checkpoint: Optional[str] = None,
        device: str = "cuda:0",
    ):
        self.ranker_config = ranker_config
        self.fm_config = fm_config
        self.fm_checkpoint = fm_checkpoint
        self.device = device

    def run_all(
        self,
        train_loader,
        val_loader,
        feature_voc_sizes: Dict[str, int],
        sparse_features: List[str],
        output_dir: str,
    ) -> dict:
        """
        执行完整网格实验: ranker_types × fm_weights。

        Returns:
            {(ranker_type, fm_weight): {direct: {...}, falcon: {...}, lift: {...}}}
        """
        os.makedirs(output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))

        all_results = {}
        total_runs = len(self.ranker_config.ranker_types) * len(self.ranker_config.fm_weights)
        run_idx = 0

        for ranker_type in self.ranker_config.ranker_types:
            for fm_weight in self.ranker_config.fm_weights:
                run_idx += 1
                tag = f"{ranker_type}_fmw{fm_weight}"
                logger.info("=" * 70)
                logger.info(f"[{run_idx}/{total_runs}] {ranker_type} | fm_weight={fm_weight}")
                logger.info("=" * 70)

                result = self._train_single(
                    ranker_type=ranker_type,
                    fm_weight=fm_weight,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    feature_voc_sizes=feature_voc_sizes,
                    sparse_features=sparse_features,
                    output_dir=output_dir,
                    writer=writer,
                    tag=tag,
                )
                all_results[tag] = result

        writer.close()
        self._save_report(all_results, output_dir)
        return all_results

    def _train_single(
        self,
        ranker_type: str,
        fm_weight: float,
        train_loader,
        val_loader,
        feature_voc_sizes: Dict[str, int],
        sparse_features: List[str],
        output_dir: str,
        writer: SummaryWriter,
        tag: str,
    ) -> dict:
        """训练单个 (ranker_type, fm_weight) 组合"""
        cfg = self.ranker_config

        ranker = create_ranker(ranker_type, sparse_features, feature_voc_sizes, cfg)
        model = JointFALCONRanker(
            ranker=ranker,
            fm_config=self.fm_config,
            ranker_config=cfg,
            fm_checkpoint=self.fm_checkpoint,
            fm_weight=fm_weight,
            device=self.device,
        ).to(self.device)

        params = model.param_summary()
        logger.info(f"参数量: ranker={params['ranker_total']:,} "
                    f"(emb={params['ranker_embedding']:,}, net={params['ranker_network']:,}), "
                    f"fm={params['fm_total']:,}, total={params['total']:,}")

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)

        best_auc = 0.0
        patience = 0
        history = []

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()
            train_stats = self._train_epoch(model, train_loader, optimizer, epoch)
            eval_stats = self._evaluate(model, val_loader)
            scheduler.step()

            direct = eval_stats["direct"]
            falcon = eval_stats["falcon"]
            lift = compute_lift(direct, falcon)

            logger.info(
                f"Epoch {epoch}/{cfg.epochs} ({time.time() - t0:.1f}s) | "
                f"loss={train_stats['loss']:.4f} | "
                f"AUC direct={direct['auc']:.4f} falcon={falcon['auc']:.4f} "
                f"(lift={lift.get('auc_lift', 0):+.4f})"
            )

            # TensorBoard
            step = epoch
            writer.add_scalar(f"{tag}/train_loss", train_stats["loss"], step)
            writer.add_scalar(f"{tag}/velocity_loss", train_stats["velocity_loss"], step)
            for k, v in direct.items():
                writer.add_scalar(f"{tag}/direct_{k}", v, step)
            for k, v in falcon.items():
                writer.add_scalar(f"{tag}/falcon_{k}", v, step)
            for k, v in lift.items():
                writer.add_scalar(f"{tag}/{k}", v, step)

            history.append({
                "epoch": epoch,
                "train": train_stats,
                "direct": direct,
                "falcon": falcon,
                "lift": lift,
            })

            # 早停 (以 falcon AUC 为准)
            current = falcon["auc"]
            if current > best_auc:
                best_auc = current
                patience = 0
                torch.save(
                    {"model_state_dict": model.state_dict(), "epoch": epoch, "auc": current},
                    os.path.join(output_dir, f"best_{tag}.pt"),
                )
            else:
                patience += 1
                if patience >= cfg.early_stop_patience:
                    logger.info(f"早停触发 (patience={patience})")
                    break

        final = history[-1] if history else {}
        return {
            "ranker_type": ranker_type,
            "fm_weight": fm_weight,
            "best_falcon_auc": best_auc,
            "params": params,
            "final": final,
            "history": history,
        }

    def _train_epoch(self, model, loader, optimizer, epoch: int) -> dict:
        model.train()
        cfg = self.ranker_config
        totals = {"loss": 0.0, "bce_direct": 0.0, "bce_falcon": 0.0, "velocity_loss": 0.0}
        n = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            sparse = {k: v.to(self.device) for k, v in batch["sparse"].items()}
            labels = batch["label"].to(self.device)

            loss_dict = model.compute_joint_loss(sparse, labels, fast_train=cfg.fast_train)

            optimizer.zero_grad(set_to_none=True)
            loss_dict["loss"].backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), cfg.grad_clip
                )
            optimizer.step()

            totals["loss"] += loss_dict["loss"].item()
            totals["bce_direct"] += loss_dict["bce_direct"]
            totals["bce_falcon"] += loss_dict["bce_falcon"]
            totals["velocity_loss"] += loss_dict["velocity_loss"]
            n += 1

            pbar.set_postfix({"loss": f"{loss_dict['loss'].item():.4f}"})

        return {k: v / max(n, 1) for k, v in totals.items()}

    @torch.no_grad()
    def _evaluate(self, model, loader) -> dict:
        """评估两条路径"""
        model.eval()
        cfg = self.ranker_config

        y_true, pred_direct, pred_falcon, user_ids = [], [], [], []

        for batch in loader:
            sparse = {k: v.to(self.device) for k, v in batch["sparse"].items()}
            y_true.append(batch["label"].numpy())
            pred_direct.append(model.predict_direct(sparse).cpu().numpy())
            pred_falcon.append(model.predict_falcon(sparse).cpu().numpy())
            if "user_id" in batch:
                user_ids.append(batch["user_id"])

        y_true = np.concatenate(y_true)
        pred_direct = np.concatenate(pred_direct)
        pred_falcon = np.concatenate(pred_falcon)
        groups = np.concatenate(user_ids) if user_ids else None

        return {
            "direct": evaluate_predictions(
                y_true, pred_direct, groups, cfg.eval_metrics, cfg.ndcg_k
            ),
            "falcon": evaluate_predictions(
                y_true, pred_falcon, groups, cfg.eval_metrics, cfg.ndcg_k
            ),
        }

    def _save_report(self, results: dict, output_dir: str):
        """保存 JSON 结果 + 打印对比表格"""
        path = os.path.join(output_dir, "ranker_results.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info("=" * 90)
        logger.info("Ranker 对比实验结果汇总")
        logger.info("=" * 90)
        header = f"{'模型':<14}{'fm_w':<8}{'AUC(direct)':<14}{'AUC(FALCON)':<14}{'Lift':<12}{'GAUC Lift':<12}"
        logger.info(header)
        logger.info("-" * 90)

        for tag, res in results.items():
            final = res.get("final", {})
            if not final:
                continue
            d, fa, lift = final["direct"], final["falcon"], final["lift"]
            logger.info(
                f"{res['ranker_type']:<14}{res['fm_weight']:<8}"
                f"{d['auc']:<14.4f}{fa['auc']:<14.4f}"
                f"{lift.get('auc_lift', 0):<+12.4f}{lift.get('gauc_lift', 0):<+12.4f}"
            )
        logger.info("=" * 90)
        logger.info(f"详细结果: {path}")

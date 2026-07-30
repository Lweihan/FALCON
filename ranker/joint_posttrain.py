"""
joint_posttrain.py — FM + Ranker 联合后训练 (Joint Post-train)
=============================================================
FALCON 后训练 Stage 1:

  在 alimamaTech/MAC 广告数据集上对预训练 FM 和 Ranker 进行联合微调。

  核心思想:
    FM 的 condition 不再来自 LLM 用户画像 embedding，
    而是来自 Ranker 从广告日志中提取的特征 embedding。
    让 FM 学会 "在 CVR 信号约束下" 优化用户 embedding 的演变方向。

  联合 Loss:
    L_cvr   = BCELoss(ranker_prob, label)         # CVR 预测监督
    L_flow  = FM.compute_loss(ranker_emb)         # Flow Matching 正则
    L_total = alpha * L_cvr + (1-alpha) * L_flow

  后续 Stage 2 (CF-OPD) 使用本阶段产出的 joint checkpoint
  对 LLM 进行 DPO 学习, 使 LLM 生成高 CVR 潜力的用户画像。

输出:
  {output_dir}/
    best_joint.pt          最佳 checkpoint (val AUC 最高)
    joint_epoch_{N}.pt     定期 checkpoint
    joint_train_log.json   训练日志
    tensorboard/           TensorBoard 日志
"""

import json
import os
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except ImportError:
    _HAS_TB = False

from common.logger import get_logger
from common.utils import count_parameters, save_checkpoint

logger = get_logger(__name__)


class _NullWriter:
    def add_scalar(self, *a, **k): pass
    def add_text(self, *a, **k): pass
    def close(self): pass


class JointPostTrainer:
    """
    FM + MAC Ranker 联合后训练 Trainer。

    典型用法:
        trainer = JointPostTrainer(ranker_config, fm_config, device)
        trainer.load_fm(fm_checkpoint)
        trainer.build_ranker(ranker_type, feature_info)
        trainer.train(train_loader, val_loader, vocab, output_dir)
    """

    def __init__(self, ranker_config, fm_config, device: str = "cuda:0"):
        self.rcfg = ranker_config
        self.fcfg = fm_config
        self.device = device

        self.fm_model = None
        self.ranker = None
        self.vocab = None

        self._is_main = True   # 单卡模式; 多卡时由外部设置

    # ── 初始化 ──

    def load_fm(self, fm_checkpoint: Optional[str] = None):
        """加载预训练 FM 模型。"""
        from fm.model import FlowMatchingModel
        self.fm_model = FlowMatchingModel.from_config(self.fcfg).to(self.device)

        if fm_checkpoint and os.path.isfile(fm_checkpoint):
            ckpt = torch.load(fm_checkpoint, map_location=self.device)
            state = ckpt.get("model_state_dict", ckpt)

            # ── 维度校验: 避免 strict=False 静默丢弃形状不匹配的权重 ──
            own_state = self.fm_model.state_dict()
            shape_mismatch = []
            for k, v in state.items():
                if k in own_state and own_state[k].shape != v.shape:
                    shape_mismatch.append((k, tuple(v.shape), tuple(own_state[k].shape)))

            if shape_mismatch:
                msg_lines = ["FM checkpoint 权重形状不匹配 (共 %d 项):" % len(shape_mismatch)]
                for k, ck_shape, my_shape in shape_mismatch[:10]:
                    msg_lines.append(f"  {k}: checkpoint={ck_shape} vs model={my_shape}")
                if len(shape_mismatch) > 10:
                    msg_lines.append(f"  ... 还有 {len(shape_mismatch)-10} 项")
                msg_lines.append("")
                msg_lines.append("常见原因: fm 配置 (time_dim / data_dim / cond_dim / dit_dim / "
                                 "dit_depth / dit_num_patches) 与训练该 checkpoint 时不一致。")
                msg_lines.append(f"当前配置: time_dim={getattr(self.fcfg, 'time_dim', '?')}, "
                                 f"data_dim={getattr(self.fcfg, 'data_dim', '?')}, "
                                 f"cond_dim={getattr(self.fcfg, 'cond_dim', '?')}, "
                                 f"backbone={getattr(self.fcfg, 'backbone_type', '?')}")
                # 若 checkpoint 里存了配置, 打印出来便于对比
                ck_cfg = ckpt.get("fm_config") or ckpt.get("config")
                if isinstance(ck_cfg, dict):
                    msg_lines.append(
                        "checkpoint 配置: " + ", ".join(
                            f"{k}={ck_cfg.get(k)}" for k in
                            ("time_dim", "data_dim", "cond_dim", "backbone_type",
                             "dit_dim", "dit_depth", "dit_num_patches")
                            if k in ck_cfg
                        )
                    )
                raise ValueError("\n".join(msg_lines))

            missing, unexpected = self.fm_model.load_state_dict(state, strict=False)
            logger.info(f"FM checkpoint 加载: {fm_checkpoint}")
            n_loaded = len([k for k in state if k in own_state])
            logger.info(f"  成功加载 {n_loaded}/{len(own_state)} 个参数张量")
            if missing:
                logger.info(f"  模型新增参数 (checkpoint 中不存在, 随机初始化): "
                            f"{len(missing)} 项, 前 5: {missing[:5]}")
            if unexpected:
                logger.warning(f"  checkpoint 冗余参数 (模型中不存在, 已忽略): "
                               f"{len(unexpected)} 项, 前 5: {unexpected[:5]}")
        else:
            logger.info("FM 随机初始化 (未指定 checkpoint)")

        if self.rcfg.freeze_fm:
            for p in self.fm_model.parameters():
                p.requires_grad = False
            logger.info("FM 全部参数已冻结")
        elif self.rcfg.freeze_fm_backbone:
            # 只冻结 velocity_net 的前向网络, 保留 condition projection 可训
            for name, p in self.fm_model.named_parameters():
                if "velocity_net" in name and "cond" not in name and "delta_t" not in name:
                    p.requires_grad = False
            logger.info("FM backbone 已冻结 (condition proj 仍可训)")

        total = count_parameters(self.fm_model)
        trainable = sum(p.numel() for p in self.fm_model.parameters() if p.requires_grad)
        logger.info(f"FM 参数: 总 {total:,} ({total/1e6:.1f}M), 可训 {trainable:,}")

    def build_ranker(self, ranker_type: str, feature_info: dict):
        """构建 MACRanker。"""
        from ranker.models import create_mac_ranker
        self.ranker = create_mac_ranker(ranker_type, feature_info, self.rcfg).to(self.device)

        # 保存构建信息, checkpoint 里一并写入, 便于 CF-OPD 阶段重建
        self._ranker_type = ranker_type
        self._feature_info = {
            k: v for k, v in feature_info.items()
            if k != "vocab_sizes"   # vocab_sizes 可能很大, 从 vocab 目录重新读
        }
        self._mm_feat_dim = feature_info.get("mm_feat_dim", 2)

        total = count_parameters(self.ranker)
        logger.info(f"MACRanker ({ranker_type}) 参数: {total:,} ({total/1e6:.1f}M)")
        logger.info(f"  embed_dim={self.rcfg.embed_dim}, "
                    f"hidden={self.rcfg.hidden_dims}, "
                    f"mm_proj_dim={self.rcfg.mm_feat_proj_dim}, "
                    f"mm_feat_dim={self._mm_feat_dim}")

    # ── 训练主流程 ──

    def train(
        self,
        train_loader,
        val_loader,
        vocab,
        output_dir: str,
    ):
        """
        执行联合后训练。

        Args:
            vocab: FeatureVocab (用于序列 ID 的 lookup)
        """
        cfg = self.rcfg
        self.vocab = vocab

        if self.fm_model is None or self.ranker is None:
            raise RuntimeError("请先调用 load_fm() 和 build_ranker()")

        os.makedirs(output_dir, exist_ok=True)
        tb_dir = os.path.join(output_dir, "tensorboard")
        if _HAS_TB and self._is_main:
            writer = SummaryWriter(log_dir=tb_dir)
        else:
            writer = _NullWriter()
        logger.info(f"Joint Post-train 开始: output={output_dir}")
        logger.info(f"  alpha={cfg.joint_alpha} (CVR loss 权重), "
                    f"(1-alpha)={1-cfg.joint_alpha:.2f} (FM loss 权重)")
        logger.info(f"  epochs={cfg.epochs}, batch={cfg.batch_size}, lr={cfg.lr}")
        logger.info(f"  grad_accum={cfg.grad_accum}, amp={cfg.use_amp}")

        all_params = list(self.fm_model.parameters()) + list(self.ranker.parameters())
        trainable_params = [p for p in all_params if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)

        use_amp = cfg.use_amp and "cuda" in str(self.device)
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        bce_loss = nn.BCEWithLogitsLoss()
        best_auc = 0.0
        global_step = 0
        log_history = []

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()

            self.fm_model.train()
            self.ranker.train()

            train_stats, steps = self._train_epoch(
                train_loader, optimizer, bce_loss, writer, global_step, scaler
            )
            global_step += steps

            val_stats = self._validate(val_loader, bce_loss)
            scheduler.step()

            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]

            logger.info(
                f"Epoch {epoch}/{cfg.epochs} ({elapsed:.1f}s) | "
                f"train_loss={train_stats['avg_loss']:.4f} "
                f"(cvr={train_stats['avg_cvr_loss']:.4f}, "
                f"flow={train_stats['avg_flow_loss']:.4f}) | "
                f"val_auc={val_stats['auc']:.4f} "
                f"val_logloss={val_stats['logloss']:.4f} | "
                f"lr={lr:.2e}"
            )

            writer.add_scalar("epoch/train_loss", train_stats["avg_loss"], epoch)
            writer.add_scalar("epoch/cvr_loss", train_stats["avg_cvr_loss"], epoch)
            writer.add_scalar("epoch/flow_loss", train_stats["avg_flow_loss"], epoch)
            writer.add_scalar("epoch/val_auc", val_stats["auc"], epoch)
            writer.add_scalar("epoch/val_logloss", val_stats["logloss"], epoch)
            writer.add_scalar("epoch/lr", lr, epoch)
            writer.add_scalar("perf/epoch_seconds", elapsed, epoch)

            log_history.append({
                "epoch": epoch,
                "train_loss": train_stats["avg_loss"],
                "cvr_loss": train_stats["avg_cvr_loss"],
                "flow_loss": train_stats["avg_flow_loss"],
                "val_auc": val_stats["auc"],
                "val_logloss": val_stats["logloss"],
                "lr": lr, "seconds": elapsed,
            })

            if val_stats["auc"] > best_auc:
                best_auc = val_stats["auc"]
                self._save_joint(output_dir, "best_joint.pt", optimizer, epoch, val_stats)
                logger.info(f"  * 新最佳 AUC={best_auc:.4f}")

            if epoch % cfg.save_every == 0:
                self._save_joint(output_dir, f"joint_epoch_{epoch}.pt", optimizer, epoch, val_stats)

        writer.close()
        with open(os.path.join(output_dir, "joint_train_log.json"), "w") as f:
            json.dump(log_history, f, indent=2)
        logger.info(f"Joint Post-train 完成, 最佳 AUC={best_auc:.4f}")
        logger.info(f"TensorBoard: tensorboard --logdir {tb_dir}")

        return os.path.join(output_dir, "best_joint.pt")

    # ── 单 epoch ──

    def _train_epoch(self, loader, optimizer, bce_loss, writer, step_offset, scaler):
        cfg = self.rcfg
        accum = max(cfg.grad_accum, 1)
        alpha = cfg.joint_alpha

        total_loss = total_cvr = total_flow = 0.0
        n = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(loader):
            sparse_ids = {
                k: v.to(self.device, non_blocking=True)
                for k, v in batch["sparse_ids"].items()
            }
            mm_feats = batch["mm_feats"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            seq_id_lists = batch["seq_id_lists"]

            is_accum_step = (batch_idx + 1) % accum != 0

            if scaler:
                with torch.cuda.amp.autocast():
                    loss_dict = self._compute_joint_loss(
                        sparse_ids, seq_id_lists, mm_feats, labels, bce_loss, alpha
                    )
                scaler.scale(loss_dict["loss"] / accum).backward()
            else:
                loss_dict = self._compute_joint_loss(
                    sparse_ids, seq_id_lists, mm_feats, labels, bce_loss, alpha
                )
                (loss_dict["loss"] / accum).backward()

            if not is_accum_step:
                if scaler:
                    if cfg.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(
                            [p for p in (list(self.fm_model.parameters()) +
                                         list(self.ranker.parameters())) if p.requires_grad],
                            cfg.grad_clip
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if cfg.grad_clip > 0:
                        nn.utils.clip_grad_norm_(
                            [p for p in (list(self.fm_model.parameters()) +
                                         list(self.ranker.parameters())) if p.requires_grad],
                            cfg.grad_clip
                        )
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss_dict["loss"].item()
            total_cvr += loss_dict["cvr_loss"].item()
            total_flow += loss_dict["flow_loss"].item()
            n += 1

            step = step_offset + n
            writer.add_scalar("step/loss", loss_dict["loss"].item(), step)
            writer.add_scalar("step/cvr_loss", loss_dict["cvr_loss"].item(), step)
            writer.add_scalar("step/flow_loss", loss_dict["flow_loss"].item(), step)

        d = max(n, 1)
        return {
            "avg_loss": total_loss / d,
            "avg_cvr_loss": total_cvr / d,
            "avg_flow_loss": total_flow / d,
        }, n

    def _compute_joint_loss(
        self, sparse_ids, seq_id_lists, mm_feats, labels, bce_loss, alpha
    ) -> dict:
        """
        联合损失:
            1. Ranker 提取 embedding → CVR 预测
            2. FM 以该 embedding 为 condition 计算 flow matching loss
        """
        # Ranker 前向: 提取 embedding 和 CVR logit
        ranker_out = self.ranker(sparse_ids, seq_id_lists, mm_feats, self.vocab)
        ranker_emb = ranker_out["embedding"]   # (B, hidden_dim)
        cvr_logit = ranker_out["logit"]        # (B,)

        # CVR 监督损失
        l_cvr = bce_loss(cvr_logit, labels)

        # FM Flow Matching 损失
        # condition = ranker_emb 投影到 fm.data_dim
        cond_dim = ranker_emb.size(-1)
        fm_data_dim = self.fcfg.data_dim if hasattr(self.fcfg, "data_dim") else 256

        if not hasattr(self, "_cond_proj") or self._cond_proj.in_features != cond_dim:
            self._cond_proj = nn.Linear(cond_dim, fm_data_dim, bias=False).to(self.device)

        condition = self._cond_proj(ranker_emb)   # (B, fm_data_dim)
        # x_clean = condition 本身 (让 FM 学习 identity + 残差变换)
        fm_loss_dict = self.fm_model.compute_loss(
            x_clean=condition.detach(),   # target 不传梯度给 ranker
            condition=condition,
            time_weight_mode=self.fcfg.time_weight_mode,
            time_weight_scale=self.fcfg.time_weight_scale,
        )
        l_flow = fm_loss_dict["loss"]

        loss = alpha * l_cvr + (1 - alpha) * l_flow

        return {"loss": loss, "cvr_loss": l_cvr, "flow_loss": l_flow}

    # ── 验证 ──

    @torch.no_grad()
    def _validate(self, loader, bce_loss) -> dict:
        self.fm_model.eval()
        self.ranker.eval()

        all_probs = []
        all_labels = []
        total_loss = 0.0
        n = 0

        for batch in loader:
            sparse_ids = {k: v.to(self.device) for k, v in batch["sparse_ids"].items()}
            mm_feats = batch["mm_feats"].to(self.device)
            labels = batch["label"].to(self.device)
            seq_id_lists = batch["seq_id_lists"]

            out = self.ranker(sparse_ids, seq_id_lists, mm_feats, self.vocab)
            probs = out["prob"].cpu().numpy()
            lbl = labels.cpu().numpy()

            all_probs.extend(probs.tolist())
            all_labels.extend(lbl.tolist())
            total_loss += bce_loss(out["logit"], labels).item()
            n += 1

        d = max(n, 1)
        from ranker.metrics import compute_auc, compute_logloss
        auc = compute_auc(all_labels, all_probs)
        logloss = compute_logloss(all_labels, all_probs)

        return {"auc": auc, "logloss": logloss, "avg_loss": total_loss / d}

    # ── 保存 ──

    def _save_joint(self, output_dir, filename, optimizer, epoch, val_stats):
        path = os.path.join(output_dir, filename)
        cond_proj = getattr(self, "_cond_proj", None)
        torch.save({
            "epoch": epoch,
            "fm_state_dict": self.fm_model.state_dict(),
            "ranker_state_dict": self.ranker.state_dict(),
            "cond_proj_state_dict": cond_proj.state_dict() if cond_proj is not None else None,
            "optimizer_state_dict": optimizer.state_dict(),
            "val_stats": val_stats,
            # 重建 Ranker 所需的元信息 (CF-OPD 阶段使用)
            "ranker_type": getattr(self, "_ranker_type", None),
            "mm_feat_dim": getattr(self, "_mm_feat_dim", None),
            "feature_info": getattr(self, "_feature_info", None),
            "ranker_config": vars(self.rcfg),
            "fm_config": vars(self.fcfg),
        }, path)
        logger.info(f"  Checkpoint 已保存: {path}")

    @classmethod
    def load_joint_checkpoint(cls, path: str, device: str = "cuda:0") -> dict:
        """
        加载 joint checkpoint, 返回模型状态。

        CF-OPD 阶段调用此方法获取 joint-trained FM 和 Ranker。
        """
        ckpt = torch.load(path, map_location=device)
        logger.info(f"Joint checkpoint 加载: {path}")
        logger.info(f"  Epoch={ckpt.get('epoch')}, "
                    f"val_auc={ckpt.get('val_stats', {}).get('auc', 'N/A')}")
        return ckpt

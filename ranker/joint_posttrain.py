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
from common.utils import count_parameters, save_checkpoint, make_grad_scaler, autocast

logger = get_logger(__name__)


class _NullWriter:
    def add_scalar(self, *a, **k): pass
    def add_text(self, *a, **k): pass
    def close(self): pass


# AMP 兼容层已抽到 common/utils.py, 这里保留别名以兼容既有调用
_make_grad_scaler = make_grad_scaler
_autocast = autocast


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
        # 标签模式在 train() 中按配置设定; 这里给默认值以便单独调用 _validate
        self._label_mode = getattr(ranker_config, "mac_label_mode", "binary")
        self._cvr_loss_name = "BCEWithLogits"

    # ── 初始化 ──

    def load_fm(self, fm_checkpoint: Optional[str] = None, auto_align_dims: bool = True):
        """
        加载预训练 FM 模型。

        Args:
            fm_checkpoint: checkpoint 路径; None 则随机初始化
            auto_align_dims: 若 checkpoint 的结构参数 (time_dim / data_dim /
                dit_dim / ...) 与当前配置不一致, 自动按 checkpoint 重建模型。
                这样切换不同时期训练的 checkpoint 都不会因 shape mismatch 崩溃。
                设为 False 则严格使用当前配置, 不匹配时报错。
        """
        from fm.model import FlowMatchingModel

        if not (fm_checkpoint and os.path.isfile(fm_checkpoint)):
            self.fm_model = FlowMatchingModel.from_config(self.fcfg).to(self.device)
            logger.info("FM 随机初始化 (未指定 checkpoint)")
            self._apply_fm_freeze()
            return

        # checkpoint 存了 config dict (非纯 tensor), 必须 weights_only=False
        ckpt = torch.load(fm_checkpoint, map_location=self.device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)

        # ── 结构对齐: 按 checkpoint 修正当前配置 ──
        if auto_align_dims:
            self._align_fm_config(ckpt, state)

        self.fm_model = FlowMatchingModel.from_config(self.fcfg).to(self.device)

        # ── 维度校验: 避免 strict=False 静默丢弃形状不匹配的权重 ──
        own_state = self.fm_model.state_dict()
        shape_mismatch = [
            (k, tuple(v.shape), tuple(own_state[k].shape))
            for k, v in state.items()
            if k in own_state and own_state[k].shape != v.shape
        ]

        if shape_mismatch:
            msg_lines = [f"FM checkpoint 权重形状不匹配 (共 {len(shape_mismatch)} 项):"]
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
            if not auto_align_dims:
                msg_lines.append("")
                msg_lines.append("提示: 可用 load_fm(..., auto_align_dims=True) 自动按 "
                                 "checkpoint 重建模型。")
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

        self._apply_fm_freeze()

    # 会被自动对齐的结构参数 (影响权重形状)
    _STRUCT_KEYS = (
        "data_dim", "time_dim", "cond_dim", "backbone_type",
        "dit_dim", "dit_depth", "dit_heads", "dit_num_patches",
        "hidden_dim", "num_layers",
        "transformer_dim", "transformer_depth", "transformer_heads",
    )

    def _align_fm_config(self, ckpt: dict, state: dict):
        """
        按 checkpoint 修正 self.fcfg 的结构参数。

        优先级:
          1. checkpoint 里存的 fm_config / config dict
          2. 从 state_dict 权重形状反推 (兼容未存 config 的老 checkpoint)
        """
        changed = []

        # ── 来源 1: checkpoint 内嵌配置 ──
        ck_cfg = ckpt.get("fm_config") or ckpt.get("config")
        if isinstance(ck_cfg, dict):
            for k in self._STRUCT_KEYS:
                if k not in ck_cfg:
                    continue
                ck_v, cur_v = ck_cfg[k], getattr(self.fcfg, k, None)
                if cur_v is not None and ck_v != cur_v:
                    setattr(self.fcfg, k, ck_v)
                    changed.append((k, cur_v, ck_v))

            # dit_preset 会在 resolve_dit_params() 里覆盖显式的 dit_* 参数,
            # 对齐后必须置空, 否则 preset 会把刚对齐的值改回去
            if any(c[0].startswith("dit_") for c in changed):
                if getattr(self.fcfg, "dit_preset", "") not in ("", None, "custom"):
                    old_preset = self.fcfg.dit_preset
                    self.fcfg.dit_preset = ""
                    changed.append(("dit_preset", old_preset, "'' (改用显式 dit_* 参数)"))

        # ── 来源 2: 从权重形状反推 (老 checkpoint 未存 config) ──
        else:
            inferred = self._infer_struct_from_state(state)
            for k, ck_v in inferred.items():
                cur_v = getattr(self.fcfg, k, None)
                if cur_v is not None and ck_v != cur_v:
                    setattr(self.fcfg, k, ck_v)
                    changed.append((k, cur_v, f"{ck_v} (从权重形状反推)"))

        if changed:
            logger.warning("FM 结构参数已按 checkpoint 自动对齐:")
            for k, old, new in changed:
                logger.warning(f"  {k}: {old} → {new}")
            logger.warning("如需严格使用当前配置, 请设 load_fm(auto_align_dims=False)")
        else:
            logger.info("FM 结构参数与 checkpoint 一致, 无需对齐")

    @staticmethod
    def _infer_struct_from_state(state: dict) -> dict:
        """
        从 state_dict 权重形状反推关键结构参数。

        DiT backbone 的形状关系:
          velocity_net.time_embed.1.weight    (dit_dim, time_dim)
          velocity_net.cond_embed.0.weight    (dit_dim, cond_dim)
          velocity_net.patch_type_embed       (1, num_patches, dit_dim)
        """
        out = {}

        w = state.get("velocity_net.time_embed.1.weight")
        if w is not None and w.dim() == 2:
            out["dit_dim"] = int(w.shape[0])
            out["time_dim"] = int(w.shape[1])

        w = state.get("velocity_net.cond_embed.0.weight")
        if w is not None and w.dim() == 2:
            out["cond_dim"] = int(w.shape[1])

        w = state.get("velocity_net.patch_type_embed")
        if w is not None and w.dim() == 3:
            out["dit_num_patches"] = int(w.shape[1])

        # data_dim = num_patches * patch_dim; patch_embed: (dit_dim, patch_dim)
        w = state.get("velocity_net.patch_embed.weight")
        if w is not None and w.dim() == 2 and "dit_num_patches" in out:
            out["data_dim"] = int(w.shape[1]) * out["dit_num_patches"]

        # DiT block 层数
        depths = [
            int(k.split(".")[2])
            for k in state
            if k.startswith("velocity_net.blocks.") and len(k.split(".")) > 3
        ]
        if depths:
            out["dit_depth"] = max(depths) + 1

        return out

    def _apply_fm_freeze(self):
        """按配置冻结 FM 参数并打印统计。"""
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
        scaler = _make_grad_scaler(use_amp)

        # ── 标签模式决定损失函数与评估指标 ──
        self._label_mode = getattr(cfg, "mac_label_mode", "binary")
        cvr_loss_fn = self._build_cvr_loss()
        best_metric, best_higher = self._resolve_best_metric()
        best_value = -float("inf") if best_higher else float("inf")

        logger.info(f"  标签模式: {self._label_mode}; "
                    f"CVR loss: {self._cvr_loss_name}; "
                    f"best 判据: {best_metric} ({'越大' if best_higher else '越小'}越好)")

        global_step = 0
        log_history = []

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()

            self.fm_model.train()
            self.ranker.train()

            train_stats, steps = self._train_epoch(
                train_loader, optimizer, cvr_loss_fn, writer, global_step, scaler
            )
            global_step += steps

            val_stats = self._validate(val_loader, cvr_loss_fn)
            scheduler.step()

            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]

            # 指标摘要 (随模式变化)
            metric_str = " ".join(
                f"val_{k}={v:.4f}" for k, v in val_stats.items()
                if k not in ("avg_loss",)
            )
            logger.info(
                f"Epoch {epoch}/{cfg.epochs} ({elapsed:.1f}s) | "
                f"train_loss={train_stats['avg_loss']:.4f} "
                f"(cvr={train_stats['avg_cvr_loss']:.4f}, "
                f"flow={train_stats['avg_flow_loss']:.4f}) | "
                f"{metric_str} | lr={lr:.2e}"
            )

            writer.add_scalar("epoch/train_loss", train_stats["avg_loss"], epoch)
            writer.add_scalar("epoch/cvr_loss", train_stats["avg_cvr_loss"], epoch)
            writer.add_scalar("epoch/flow_loss", train_stats["avg_flow_loss"], epoch)
            for k, v in val_stats.items():
                writer.add_scalar(f"epoch/val_{k}", v, epoch)
            writer.add_scalar("epoch/lr", lr, epoch)
            writer.add_scalar("perf/epoch_seconds", elapsed, epoch)

            log_history.append({
                "epoch": epoch,
                "train_loss": train_stats["avg_loss"],
                "cvr_loss": train_stats["avg_cvr_loss"],
                "flow_loss": train_stats["avg_flow_loss"],
                **{f"val_{k}": v for k, v in val_stats.items()},
                "lr": lr, "seconds": elapsed,
            })

            cur = val_stats.get(best_metric)
            if cur is not None:
                improved = (cur > best_value) if best_higher else (cur < best_value)
                if improved:
                    best_value = cur
                    self._save_joint(output_dir, "best_joint.pt", optimizer, epoch, val_stats)
                    logger.info(f"  * 新最佳 {best_metric}={best_value:.4f}")

            if epoch % cfg.save_every == 0:
                self._save_joint(output_dir, f"joint_epoch_{epoch}.pt", optimizer, epoch, val_stats)

        writer.close()
        with open(os.path.join(output_dir, "joint_train_log.json"), "w") as f:
            json.dump(log_history, f, indent=2)
        logger.info(f"Joint Post-train 完成, 最佳 {best_metric}={best_value:.4f}")
        logger.info(f"TensorBoard: tensorboard --logdir {tb_dir}")

        return os.path.join(output_dir, "best_joint.pt")

    # ── 损失函数 / best 判据 ──

    def _build_cvr_loss(self):
        """
        按标签模式构建 CVR 损失。

        binary:     BCEWithLogitsLoss (标准二分类)
        continuous: 默认 BCEWithLogitsLoss (等价软标签交叉熵, 对 [0,1] 目标
                    比 MSE 更合适 — 梯度在饱和区不消失);
                    可通过 mac_continuous_loss='mse' 切换到 MSE
        """
        mode = self._label_mode
        loss_kind = getattr(self.rcfg, "mac_continuous_loss", "bce")

        if mode == "continuous" and loss_kind == "mse":
            self._cvr_loss_name = "MSE(on sigmoid)"
            mse = nn.MSELoss()

            def _loss(logit, target):
                return mse(torch.sigmoid(logit), target)

            return _loss

        self._cvr_loss_name = "BCEWithLogits" + ("(软标签)" if mode == "continuous" else "")
        return nn.BCEWithLogitsLoss()

    def _resolve_best_metric(self):
        """
        返回 (metric_name, higher_is_better)。

        auto: binary → auc (越大越好); continuous → mse (越小越好)
        """
        from ranker.metrics import LOWER_IS_BETTER

        choice = getattr(self.rcfg, "mac_best_metric", "auto")
        if choice == "auto":
            metric = "auc" if self._label_mode == "binary" else "mse"
        else:
            metric = choice
        return metric, metric not in LOWER_IS_BETTER

    # ── 单 epoch ──

    def _train_epoch(self, loader, optimizer, cvr_loss_fn, writer, step_offset, scaler):
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
                with _autocast():
                    loss_dict = self._compute_joint_loss(
                        sparse_ids, seq_id_lists, mm_feats, labels, cvr_loss_fn, alpha
                    )
                scaler.scale(loss_dict["loss"] / accum).backward()
            else:
                loss_dict = self._compute_joint_loss(
                    sparse_ids, seq_id_lists, mm_feats, labels, cvr_loss_fn, alpha
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
        self, sparse_ids, seq_id_lists, mm_feats, labels, cvr_loss_fn, alpha
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
        l_cvr = cvr_loss_fn(cvr_logit, labels)

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
    def _validate(self, loader, cvr_loss_fn) -> dict:
        """
        验证。指标集随标签模式变化:

          binary:     auc / gauc / logloss / pos_rate
          continuous: mse / mae / pearson / spearman / auc_bin (参考) / label_mean

        auc_bin 是把连续标签按 > 0 二值化后算的 AUC, 便于与 binary 模式横向对比。
        """
        self.fm_model.eval()
        self.ranker.eval()

        all_probs, all_labels, all_uids = [], [], []
        total_loss = 0.0
        n = 0

        for batch in loader:
            sparse_ids = {k: v.to(self.device) for k, v in batch["sparse_ids"].items()}
            mm_feats = batch["mm_feats"].to(self.device)
            labels = batch["label"].to(self.device)
            seq_id_lists = batch["seq_id_lists"]

            out = self.ranker(sparse_ids, seq_id_lists, mm_feats, self.vocab)

            all_probs.extend(out["prob"].float().cpu().numpy().tolist())
            all_labels.extend(labels.float().cpu().numpy().tolist())
            # GAUC 需要用户分组
            uid = batch["sparse_ids"].get("user_id")
            if uid is not None:
                all_uids.extend(uid.cpu().numpy().tolist())

            total_loss += cvr_loss_fn(out["logit"], labels).item()
            n += 1

        d = max(n, 1)
        stats = {"avg_loss": total_loss / d}

        if not all_labels:
            logger.warning("验证集为空, 跳过指标计算")
            return stats

        from ranker.metrics import (
            compute_auc, compute_gauc, compute_logloss, compute_pos_rate,
            compute_mse, compute_mae, compute_pearson, compute_spearman,
        )

        mode = getattr(self, "_label_mode", "binary")
        gids = all_uids if len(all_uids) == len(all_labels) else None

        if mode == "binary":
            stats["auc"] = compute_auc(all_labels, all_probs)
            if gids is not None:
                stats["gauc"] = compute_gauc(all_labels, all_probs, gids)
            stats["logloss"] = compute_logloss(all_labels, all_probs)
            stats["pos_rate"] = compute_pos_rate(all_labels)
        else:
            thr = getattr(self.rcfg, "mac_label_threshold", 0.0)
            stats["mse"] = compute_mse(all_labels, all_probs)
            stats["mae"] = compute_mae(all_labels, all_probs)
            stats["pearson"] = compute_pearson(all_labels, all_probs)
            stats["spearman"] = compute_spearman(all_labels, all_probs)
            stats["logloss"] = compute_logloss(all_labels, all_probs)
            # 二值化后的参考 AUC (便于与 binary 模式对比)
            stats["auc_bin"] = compute_auc(all_labels, all_probs, binarize_threshold=thr)
            stats["label_mean"] = float(sum(all_labels) / len(all_labels))

        return stats

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
            "label_mode": getattr(self, "_label_mode", "binary"),
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
        ckpt = torch.load(path, map_location=device, weights_only=False)
        logger.info(f"Joint checkpoint 加载: {path}")
        logger.info(f"  Epoch={ckpt.get('epoch')}, "
                    f"val_auc={ckpt.get('val_stats', {}).get('auc', 'N/A')}")
        return ckpt

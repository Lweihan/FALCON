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

try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from common.logger import get_logger
from common.utils import count_parameters, save_checkpoint, make_grad_scaler, autocast
from common.distributed import (
    barrier, get_local_rank, get_world_size, is_main_process,
    reduce_value, unwrap_model,
)

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

    单卡:
        trainer = JointPostTrainer(ranker_config, fm_config, device)
        trainer.load_fm(fm_checkpoint)
        trainer.build_ranker(ranker_type, feature_info)
        trainer.train(train_loader, val_loader, vocab, output_dir)

    多卡 (torchrun --nproc_per_node=4):
        is_dist, rank, world, local_rank, device = setup_distributed()
        trainer = JointPostTrainer(rc, fc, device,
                                   distributed=is_dist, local_rank=local_rank)
        ...  # 其余调用相同
    """

    def __init__(
        self,
        ranker_config,
        fm_config,
        device: str = "cuda:0",
        distributed: bool = False,
        local_rank: int = 0,
    ):
        self.rcfg = ranker_config
        self.fcfg = fm_config
        self.device = device

        self.fm_model = None
        self.ranker = None
        self.vocab = None
        self.cond_proj = None      # ranker_emb → fm.data_dim 投影

        # DDP 包装后的引用 (前向/反向走这些); 未包装时就是原模型
        self._fm_ddp = None
        self._ranker_ddp = None
        self._cond_proj_ddp = None

        # ── 分布式状态 ──
        self.distributed = distributed
        self.local_rank = local_rank
        self.world_size = get_world_size()
        self._is_main = is_main_process()

        # 标签模式在 train() 中按配置设定; 这里给默认值以便单独调用 _validate
        self._label_mode = getattr(ranker_config, "mac_label_mode", "binary")
        self._cvr_loss_name = "BCEWithLogits"
        self._max_steps = -1   # DDP step 对齐上限, -1 = 不限制

    def _log(self, msg: str, level: str = "info"):
        """只在主进程打印"""
        if self._is_main:
            getattr(logger, level)(msg)

    def _logw(self, msg: str):
        """只在主进程打印 warning"""
        if self._is_main:
            logger.warning(msg)

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
            self._log("FM 随机初始化 (未指定 checkpoint)")
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
        self._log(f"FM checkpoint 加载: {fm_checkpoint}")
        n_loaded = len([k for k in state if k in own_state])
        self._log(f"  成功加载 {n_loaded}/{len(own_state)} 个参数张量")
        if missing:
            self._log(f"  模型新增参数 (checkpoint 中不存在, 随机初始化): "
                        f"{len(missing)} 项, 前 5: {missing[:5]}")
        if unexpected:
            self._logw(f"  checkpoint 冗余参数 (模型中不存在, 已忽略): "
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
            self._logw("FM 结构参数已按 checkpoint 自动对齐:")
            for k, old, new in changed:
                self._logw(f"  {k}: {old} → {new}")
            self._logw("如需严格使用当前配置, 请设 load_fm(auto_align_dims=False)")
        else:
            self._log("FM 结构参数与 checkpoint 一致, 无需对齐")

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
            self._log("FM 全部参数已冻结")
        elif self.rcfg.freeze_fm_backbone:
            # 只冻结 velocity_net 的前向网络, 保留 condition projection 可训
            for name, p in self.fm_model.named_parameters():
                if "velocity_net" in name and "cond" not in name and "delta_t" not in name:
                    p.requires_grad = False
            self._log("FM backbone 已冻结 (condition proj 仍可训)")

        total = count_parameters(self.fm_model)
        trainable = sum(p.numel() for p in self.fm_model.parameters() if p.requires_grad)
        self._log(f"FM 参数: 总 {total:,} ({total/1e6:.1f}M), 可训 {trainable:,}")

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

        # ── cond_proj: ranker_emb → fm.data_dim ──
        # 必须在此提前创建 (而非训练中途懒加载)。
        # DDP 要求所有参数在包装时就已存在, 否则该层永远不参与梯度同步,
        # 各卡独立更新导致权重分叉。
        cond_in = self.ranker.embed_output_dim
        fm_data_dim = getattr(self.fcfg, "data_dim", 256)
        self.cond_proj = nn.Linear(cond_in, fm_data_dim, bias=False).to(self.device)

        total = count_parameters(self.ranker)
        self._log(f"MACRanker ({ranker_type}) 参数: {total:,} ({total/1e6:.1f}M)")
        self._log(f"  embed_dim={self.rcfg.embed_dim}, "
                  f"hidden={self.rcfg.hidden_dims}, "
                  f"mm_proj_dim={self.rcfg.mm_feat_proj_dim}, "
                  f"mm_feat_dim={self._mm_feat_dim}")
        self._log(f"  cond_proj: {cond_in} → {fm_data_dim}")

    # ── DDP 包装 ──

    def _wrap_ddp(self):
        """
        用 DistributedDataParallel 包装 FM / Ranker / cond_proj。

        三个模块独立包装, 各自维护梯度同步桶。
        前向必须走 self._fm_ddp / _ranker_ddp / _cond_proj_ddp,
        否则 DDP 的梯度同步 hook 不会触发 → 各卡梯度不同步。
        """
        if not self.distributed:
            # 单卡: 直接指向原模型
            self._fm_ddp = self.fm_model
            self._ranker_ddp = self.ranker
            self._cond_proj_ddp = self.cond_proj
            return

        from torch.nn.parallel import DistributedDataParallel

        # find_unused_parameters:
        #   开启有性能损失 (每步需额外遍历计算图), 故按需决定。
        #   auto 判定条件:
        #     1. freeze_fm / freeze_fm_backbone — 部分参数无梯度
        #     2. use_delta_t_condition=False — FM 的 delta_t_embed 不参与 loss
        #        (_compute_joint_loss 会传 delta_t, 正常情况该层有梯度)
        choice = getattr(self.rcfg, "ddp_find_unused_parameters", "auto")
        if choice == "auto":
            find_unused = bool(
                getattr(self.rcfg, "freeze_fm", False)
                or getattr(self.rcfg, "freeze_fm_backbone", False)
                or not getattr(self.fcfg, "use_delta_t_condition", True)
            )
        elif isinstance(choice, bool):
            find_unused = choice
        else:
            find_unused = str(choice).lower() in ("true", "1", "yes")

        ids = [self.local_rank] if "cuda" in str(self.device) else None

        def _wrap(m, name):
            # 全部参数都冻结的模块不能包装 DDP (没有可训练参数会报错)
            if not any(p.requires_grad for p in m.parameters()):
                self._log(f"  {name}: 全部参数已冻结, 跳过 DDP 包装")
                return m
            return DistributedDataParallel(
                m, device_ids=ids, output_device=self.local_rank if ids else None,
                find_unused_parameters=find_unused,
            )

        self._fm_ddp = _wrap(self.fm_model, "FM")
        self._ranker_ddp = _wrap(self.ranker, "Ranker")
        self._cond_proj_ddp = _wrap(self.cond_proj, "cond_proj")

        self._log(f"  DDP 包装完成 (find_unused_parameters={find_unused})")

    def _precompute_max_steps(self, loader) -> int:
        """
        训练开始前算出各 rank 的 step 数, 取全局最小值。

        为何必须"事前"算: DDP 要求所有 rank 执行相同次数 forward/backward,
        否则先耗尽数据的 rank 去等 epoch 末的 all_reduce, 而其他 rank 仍在
        做梯度同步 → 双方永久互等。死锁发生在**首个 epoch 进行中**,
        事后同步来不及。

        两级策略:
          1. metadata 路径 (默认, 零开销): parquet footer 直接给出行数,
             7500 万行也是毫秒级
          2. fallback 路径: 数据源无法提供行数时 (list 切片 / HF streaming),
             空跑一遍 loader 计数, 有额外 I/O 开销但保证正确

        Returns:
            全局最小 step 数; 非分布式返回 -1 (不限制)
        """
        if not self.distributed:
            return -1

        import math
        import torch.distributed as dist

        local_steps = None

        # ── 路径 1: 从 parquet metadata 推算 ──
        n_rows = self._probe_num_rows(loader)
        if n_rows is not None and n_rows > 0:
            bs = getattr(loader, "batch_size", None) or self.rcfg.batch_size
            local_steps = math.ceil(n_rows / bs)
            self._log(f"  step 数预算 (metadata): {n_rows:,} 行 / batch {bs} "
                      f"= {local_steps} steps")

        # ── 路径 2: 空跑计数 ──
        if local_steps is None:
            self._log("  数据源无法提供行数, 空跑一遍 loader 计数 (有额外 I/O 开销)...",
                      level="warning")
            local_steps = sum(1 for _ in loader)
            self._log(f"  step 数预算 (迭代计数): {local_steps} steps")

        t = torch.tensor([local_steps], dtype=torch.long, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        min_steps = int(t.item())

        if min_steps < local_steps:
            self._logw(
                f"  各 rank 数据量不均 (本 rank {local_steps} steps, "
                f"全局最小 {min_steps}), 统一截断到 {min_steps} 步以避免 DDP 死锁。"
                f"本 epoch 将丢弃约 {local_steps - min_steps} 个 batch。"
            )
        elif min_steps == local_steps:
            self._log(f"  各 rank step 数一致 ({min_steps}), 无需截断")

        return max(min_steps, 1)

    def _probe_num_rows(self, loader) -> Optional[int]:
        """
        尝试从 loader 的数据源获取本 rank 行数 (不消耗迭代器)。

        支持:
          ParquetDirIterable  → num_rows() 读 parquet metadata
          list / tuple        → len()
        其他 (HF streaming 等) → None, 交由调用方走 fallback
        """
        ds = getattr(loader, "dataset", None)
        if ds is None:
            return None

        src = getattr(ds, "iter", None)     # MACIterableDataset.iter
        if src is None:
            return None

        n = None
        if hasattr(src, "num_rows"):        # ParquetDirIterable
            try:
                n = int(src.num_rows(local_only=True))
            except TypeError:
                n = int(src.num_rows())
            except Exception:
                return None
        elif isinstance(src, (list, tuple)):
            n = len(src)

        if n is None:
            return None

        # max_samples 限制也要纳入 (Dataset 会在到达上限时 break)
        max_s = getattr(ds, "max_samples", -1)
        if max_s and max_s > 0:
            n = min(n, max_s)
        return n

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

        if self._is_main:
            os.makedirs(output_dir, exist_ok=True)
        barrier()   # 确保目录已创建再继续

        tb_dir = os.path.join(output_dir, "tensorboard")
        if _HAS_TB and self._is_main:
            writer = SummaryWriter(log_dir=tb_dir)
        else:
            writer = _NullWriter()

        self._log(f"Joint Post-train 开始: output={output_dir}")
        if self.distributed:
            self._log(f"  DDP: world_size={self.world_size}, "
                      f"local_rank={self.local_rank}")
            self._log(f"  等效 batch: {cfg.batch_size} × {self.world_size} 卡 × "
                      f"{cfg.grad_accum} 累积 = "
                      f"{cfg.batch_size * self.world_size * cfg.grad_accum}")
        self._log(f"  alpha={cfg.joint_alpha} (CVR loss 权重), "
                  f"(1-alpha)={1-cfg.joint_alpha:.2f} (FM loss 权重)")
        self._log(f"  epochs={cfg.epochs}, batch={cfg.batch_size}/卡, lr={cfg.lr}")
        self._log(f"  grad_accum={cfg.grad_accum}, amp={cfg.use_amp}")

        # ── DDP 包装 ──
        self._wrap_ddp()

        all_params = (list(self.fm_model.parameters())
                      + list(self.ranker.parameters())
                      + list(self.cond_proj.parameters()))
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

        self._log(f"  标签模式: {self._label_mode}; "
                  f"CVR loss: {self._cvr_loss_name}; "
                  f"best 判据: {best_metric} ({'越大' if best_higher else '越小'}越好)")

        global_step = 0
        log_history = []
        # DDP 下各 rank 数据量可能不均 → step 数不一致会在 all_reduce 处死锁。
        # 必须在训练开始前 (而非首个 epoch 后) 就算出全局最小 step 数。
        self._max_steps = self._precompute_max_steps(train_loader)

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()

            self.fm_model.train()
            self.ranker.train()
            self.cond_proj.train()

            train_stats, steps = self._train_epoch(
                train_loader, optimizer, cvr_loss_fn, writer, global_step, scaler, epoch
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
            self._log(
                f"Epoch {epoch}/{cfg.epochs} ({elapsed:.1f}s, {steps} steps) | "
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
            writer.add_scalar("perf/steps", steps, epoch)

            log_history.append({
                "epoch": epoch,
                "train_loss": train_stats["avg_loss"],
                "cvr_loss": train_stats["avg_cvr_loss"],
                "flow_loss": train_stats["avg_flow_loss"],
                **{f"val_{k}": v for k, v in val_stats.items()},
                "lr": lr, "seconds": elapsed, "steps": steps,
            })

            # 保存只在主进程 (val_stats 已 all_reduce, 各 rank 判据一致)
            cur = val_stats.get(best_metric)
            if cur is not None:
                improved = (cur > best_value) if best_higher else (cur < best_value)
                if improved:
                    best_value = cur
                    if self._is_main:
                        self._save_joint(output_dir, "best_joint.pt",
                                         optimizer, epoch, val_stats)
                        logger.info(f"  * 新最佳 {best_metric}={best_value:.4f}")

            if epoch % cfg.save_every == 0 and self._is_main:
                self._save_joint(output_dir, f"joint_epoch_{epoch}.pt",
                                 optimizer, epoch, val_stats)
            barrier()   # 等主进程写完再进入下一 epoch

        writer.close()
        if self._is_main:
            with open(os.path.join(output_dir, "joint_train_log.json"), "w") as f:
                json.dump(log_history, f, indent=2)
            logger.info(f"Joint Post-train 完成, 最佳 {best_metric}={best_value:.4f}")
            logger.info(f"TensorBoard: tensorboard --logdir {tb_dir}")
        barrier()

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

    # ── 进度条 ──

    def _progress(self, iterable, desc: str, total: Optional[int] = None):
        """
        包装迭代器为 tqdm 进度条 (仅主进程显示, 避免多卡日志交错)。

        tqdm 未安装或非主进程时原样返回, 不影响功能。
        """
        if not (_HAS_TQDM and self._is_main):
            return iterable
        return tqdm(
            iterable, desc=desc, total=total if total and total > 0 else None,
            leave=False, dynamic_ncols=True, unit="step",
        )

    # ── 单 epoch ──

    def _train_epoch(self, loader, optimizer, cvr_loss_fn, writer,
                     step_offset, scaler, epoch: int = 0):
        cfg = self.rcfg
        accum = max(cfg.grad_accum, 1)
        alpha = cfg.joint_alpha

        total_loss = total_cvr = total_flow = 0.0
        n = 0
        optimizer.zero_grad(set_to_none=True)

        # 所有可训练参数 (梯度裁剪用), 提前算好避免每步重复遍历
        clip_params = [
            p for p in (list(self.fm_model.parameters())
                        + list(self.ranker.parameters())
                        + list(self.cond_proj.parameters()))
            if p.requires_grad
        ]

        # 进度条总数: DDP 下用预算的 max_steps; 单卡时若能探测行数也显示
        bar_total = self._max_steps if self._max_steps > 0 else self._estimate_steps(loader)
        pbar = self._progress(loader, f"Epoch {epoch} train", bar_total)

        for batch_idx, batch in enumerate(pbar):
            # DDP: 截断到全局最小 step 数, 避免各 rank 步数不一致导致死锁
            if self._max_steps > 0 and n >= self._max_steps:
                break

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
                        nn.utils.clip_grad_norm_(clip_params, cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if cfg.grad_clip > 0:
                        nn.utils.clip_grad_norm_(clip_params, cfg.grad_clip)
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

            # 进度条实时显示滑动平均损失
            if hasattr(pbar, "set_postfix") and n % 10 == 0:
                pbar.set_postfix(
                    loss=f"{total_loss / n:.4f}",
                    cvr=f"{total_cvr / n:.4f}",
                    flow=f"{total_flow / n:.4f}",
                    refresh=False,
                )

        if hasattr(pbar, "close"):
            pbar.close()

        d = max(n, 1)
        # DDP: 跨卡聚合平均损失, 让日志反映全局而非单卡
        return {
            "avg_loss": reduce_value(total_loss / d, device=self.device),
            "avg_cvr_loss": reduce_value(total_cvr / d, device=self.device),
            "avg_flow_loss": reduce_value(total_flow / d, device=self.device),
        }, n

    def _estimate_steps(self, loader) -> Optional[int]:
        """
        估算 loader 的 step 数 (仅用于进度条总数显示, 失败返回 None)。

        用 loader.batch_size 而非 cfg.batch_size —
        验证集的 batch_size 是训练的 2 倍。
        """
        try:
            import math
            n_rows = self._probe_num_rows(loader)
            if n_rows:
                bs = getattr(loader, "batch_size", None) or self.rcfg.batch_size
                return math.ceil(n_rows / bs)
        except Exception:
            pass
        return None

    def _compute_joint_loss(
        self, sparse_ids, seq_id_lists, mm_feats, labels, cvr_loss_fn, alpha
    ) -> dict:
        """
        联合损失:
            1. Ranker 提取 embedding → CVR 预测
            2. cond_proj 投影到 fm.data_dim
            3. FM 以该 embedding 为 condition 计算 flow matching loss

        注意: 三个模块的前向都必须走 DDP 包装后的引用 (self._*_ddp),
        否则 DDP 的梯度同步 hook 不会触发, 各卡梯度不同步。
        FM 侧调用 __call__ 而非 compute_loss() —
        FlowMatchingModel.forward() 就是 loss 计算 (专为 DDP 设计)。
        """
        # Ranker 前向: 提取 embedding 和 CVR logit
        ranker_out = self._ranker_ddp(sparse_ids, seq_id_lists, mm_feats, self.vocab)
        ranker_emb = ranker_out["embedding"]   # (B, hidden_dim)
        cvr_logit = ranker_out["logit"]        # (B,)

        # CVR 监督损失
        l_cvr = cvr_loss_fn(cvr_logit, labels)

        # condition = ranker_emb 投影到 fm.data_dim (cond_proj 已在 build_ranker 创建)
        condition = self._cond_proj_ddp(ranker_emb)   # (B, fm_data_dim)

        # delta_t: 传入配置的训练时间跨度。
        # 两个作用:
        #   1. 语义对齐 — CF-OPD 推理时会用 delta_t 预测未来 embedding,
        #      这里传相同值让 FM 在该跨度上得到训练
        #   2. DDP 必需 — 若不传, delta_t_embed 的参数永不参与 loss,
        #      DDP 会报 "Expected to have finished reduction in the prior iteration"
        #      (虽可用 find_unused_parameters=True 绕过, 但有性能损失)
        delta_t = None
        if getattr(self.fcfg, "use_delta_t_condition", True):
            dt = float(getattr(self.rcfg, "train_delta_t", 0.5))
            delta_t = torch.full(
                (condition.size(0),), dt,
                device=condition.device, dtype=condition.dtype,
            )

        # FM Flow Matching 损失
        # x_clean = condition 本身 (让 FM 学习 identity + 残差变换)
        # 走 __call__ → forward() → compute_loss(), 保证 DDP hook 生效
        fm_loss_dict = self._fm_ddp(
            condition.detach(),          # x_clean: target 不传梯度给 ranker
            condition,                   # condition
            self.fcfg.time_weight_mode,
            self.fcfg.time_weight_scale,
            delta_t,
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

        DDP 说明: 验证集不按 rank 切分, 各 rank 读完整验证集独立算指标,
        再 all_reduce 取平均。这样各 rank 得到一致的 best 判据 (保存决策不分叉),
        且避免切分后某 rank 拿到全负样本导致 AUC 无定义。
        前向用未包装的原模型 (eval 无需梯度同步, 走 DDP 反而有额外开销)。
        """
        self.fm_model.eval()
        self.ranker.eval()
        self.cond_proj.eval()

        all_probs, all_labels, all_uids = [], [], []
        total_loss = 0.0
        n = 0

        vbar = self._progress(loader, "  validate", self._estimate_steps(loader))
        for batch in vbar:
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

        if hasattr(vbar, "close"):
            vbar.close()

        d = max(n, 1)
        stats = {"avg_loss": total_loss / d}

        if not all_labels:
            self._log("验证集为空, 跳过指标计算", level="warning")
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

        # DDP: 跨卡聚合取平均。
        # 各 rank 读的是同一份完整验证集, 理论上指标本就一致;
        # 聚合可消除浮点累加顺序差异, 确保各 rank 的 best 判据完全相同
        # (否则可能出现部分 rank 判定"改进"、部分不改进 → barrier 处不同步)
        if self.distributed:
            stats = {
                k: reduce_value(v, device=self.device) for k, v in stats.items()
            }

        return stats

    # ── 保存 ──

    def _save_joint(self, output_dir, filename, optimizer, epoch, val_stats):
        """
        保存 joint checkpoint (只应在主进程调用)。

        用 unwrap_model 剥离可能的 DDP / compile 包装 —
        否则 state_dict 的 key 会带 'module.' 前缀, 单卡加载时全部 missing。
        """
        path = os.path.join(output_dir, filename)

        fm = unwrap_model(self.fm_model)
        ranker = unwrap_model(self.ranker)
        cond_proj = unwrap_model(self.cond_proj) if self.cond_proj is not None else None

        torch.save({
            "epoch": epoch,
            "fm_state_dict": fm.state_dict(),
            "ranker_state_dict": ranker.state_dict(),
            "cond_proj_state_dict": cond_proj.state_dict() if cond_proj is not None else None,
            "optimizer_state_dict": optimizer.state_dict(),
            "val_stats": val_stats,
            # 重建 Ranker 所需的元信息 (CF-OPD 阶段使用)
            "ranker_type": getattr(self, "_ranker_type", None),
            "mm_feat_dim": getattr(self, "_mm_feat_dim", None),
            "feature_info": getattr(self, "_feature_info", None),
            "label_mode": getattr(self, "_label_mode", "binary"),
            "world_size": self.world_size,
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

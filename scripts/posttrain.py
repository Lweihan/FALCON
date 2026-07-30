#!/usr/bin/env python
"""
posttrain.py — FALCON 后训练主脚本
====================================
后训练分两个阶段:

  Stage 1 (joint_train):
    将预训练 FM + MACRanker 在 alimamaTech/MAC 广告数据集上联合微调。
    FM 学习 "在 CVR 信号约束下" 优化用户 embedding 的演变方向。
    Loss = alpha * L_CVR + (1-alpha) * L_flow

  Stage 2 (cf_opd):
    用 Stage 1 产出的 joint checkpoint 为 FM Teacher + Ranker,
    让 LLM (Student) 生成候选用户画像,
    FM+Ranker 对候选打 CVR 分,
    高分候选作为正例 / 低分作为负例,
    DPO 学习使 LLM 生成更高 CVR 潜力的画像。

使用示例:

  # 完整后训练 (joint_train → cf_opd)
  python scripts/posttrain.py \\
      --config configs/falcon_config.yaml \\
      --fm_checkpoint ./output/fm_checkpoints/best_fm.pt \\
      --mac_data_path /path/to/MAC \\
      --mac_label last

  # 仅 joint 训练 (多卡)
  torchrun --nproc_per_node=4 scripts/posttrain.py \\
      --stage joint_train \\
      --fm_checkpoint ./output/fm_checkpoints/best_fm.pt \\
      --mac_data_path /path/to/MAC \\
      --ranker_type deepfm \\
      --epochs 3 --batch_size 1024

  # 调试 (只读 1000 条)
  python scripts/posttrain.py --stage joint_train \\
      --mac_data_path /path/to/MAC --mac_max_samples 1000

  # 仅 CF-OPD (已有 joint checkpoint)
  python scripts/posttrain.py \\
      --stage cf_opd \\
      --joint_checkpoint ./output/joint_checkpoints/best_joint.pt \\
      --mac_data_path /path/to/MAC \\
      --embedding_dir ./output/embeddings \\
      --student_model Qwen/Qwen3-7B

MAC 目录结构 (local_dir 模式, 默认):
  MAC/
  ├── train/   *.parquet
  ├── val/     *.parquet
  └── vocabs/  {field}.json
"""

import argparse
import os
import sys

# 添加项目根目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.config import FALCONConfig, RankerConfig, FMConfig
from common.logger import get_logger
from common.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="FALCON 后训练脚本")

    # ── 通用 ──
    parser.add_argument("--config", default="", help="YAML 配置文件路径")
    parser.add_argument(
        "--stage",
        choices=["joint_train", "cf_opd", "all"],
        default="all",
        help="后训练阶段: joint_train / cf_opd / all (默认)",
    )
    parser.add_argument("--output_dir", default="./output", help="输出根目录")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto / cuda:0 / cpu")

    # ── Joint Post-train ──
    parser.add_argument("--fm_checkpoint", default="", help="预训练 FM checkpoint 路径")
    parser.add_argument(
        "--ranker_type",
        default="deepfm",
        choices=["dnn", "deepfm", "dcn", "autoint", "token_mixer"],
        help="Ranker 类型",
    )
    parser.add_argument("--epochs", type=int, default=None, help="训练 epoch 数")
    parser.add_argument("--batch_size", type=int, default=None, help="每 GPU batch 大小")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--joint_alpha", type=float, default=None, help="CVR loss 权重 α")
    parser.add_argument("--grad_accum", type=int, default=None, help="梯度累积步数")
    parser.add_argument("--no_amp", action="store_true", help="关闭混合精度")
    parser.add_argument("--freeze_fm", action="store_true", help="冻结 FM 全部参数")
    parser.add_argument("--freeze_fm_backbone", action="store_true", help="冻结 FM backbone")

    # ── MAC 数据集 ──
    parser.add_argument(
        "--mac_load_mode",
        default="local_dir",
        choices=["local_dir", "hf_streaming", "hf_download", "local_parquet", "local_csv"],
        help="MAC 数据集加载模式 (local_dir 推荐: 需 train/ val/ vocabs/ 子目录)",
    )
    parser.add_argument("--mac_data_path", default="",
                        help="local_dir: MAC 根目录; local_parquet/csv: 单文件路径")
    parser.add_argument("--mac_train_subdir", default="train", help="训练数据子目录")
    parser.add_argument("--mac_val_subdir", default="val", help="验证数据子目录")
    parser.add_argument("--mac_vocab_subdir", default="vocabs", help="词表子目录")
    parser.add_argument("--mac_parquet_batch_size", type=int, default=None,
                        help="parquet 流式读取行组大小 (默认 10000)")
    parser.add_argument(
        "--mac_label",
        default="last",
        choices=["linear", "mta", "first", "last"],
        help="CVR 标签字段",
    )
    parser.add_argument("--mac_vocab_path", default="",
                        help="独立词表 json 路径 (非 local_dir 模式)")
    parser.add_argument("--mac_max_samples", type=int, default=-1, help="-1=全量; 正数=限制(调试)")
    parser.add_argument("--mac_val_max_samples", type=int, default=None,
                        help="val 最多读取样本数 (默认 100000)")
    parser.add_argument("--mac_val_ratio", type=float, default=0.01,
                        help="val 比例 (仅非 local_dir 模式生效)")

    # ── CF-OPD ──
    parser.add_argument("--joint_checkpoint", default="", help="Joint 后训练的 checkpoint 路径")
    parser.add_argument("--embedding_dir", default="", help="用户 embedding 目录")
    parser.add_argument("--student_model", default="", help="LLM Student 模型路径/名称")
    parser.add_argument("--num_candidates", type=int, default=None, help="每步候选数量")

    return parser.parse_args()


def build_configs(args) -> FALCONConfig:
    """从 YAML + 命令行参数构建配置。"""
    if args.config and os.path.isfile(args.config):
        cfg = FALCONConfig.from_yaml(args.config)
    else:
        cfg = FALCONConfig()

    # 覆写通用字段
    cfg.output_dir = args.output_dir
    cfg.seed = args.seed
    if args.device != "auto":
        cfg.device = args.device

    # 覆写 RankerConfig (MAC 相关)
    rc = cfg.ranker
    if args.epochs is not None:
        rc.epochs = args.epochs
    if args.batch_size is not None:
        rc.batch_size = args.batch_size
    if args.lr is not None:
        rc.lr = args.lr
    if args.joint_alpha is not None:
        rc.joint_alpha = args.joint_alpha
    if args.grad_accum is not None:
        rc.grad_accum = args.grad_accum
    if args.no_amp:
        rc.use_amp = False
    if args.freeze_fm:
        rc.freeze_fm = True
    if args.freeze_fm_backbone:
        rc.freeze_fm_backbone = True

    rc.mac_load_mode = args.mac_load_mode
    if args.mac_data_path:
        rc.mac_data_path = args.mac_data_path
    rc.mac_train_subdir = args.mac_train_subdir
    rc.mac_val_subdir = args.mac_val_subdir
    rc.mac_vocab_subdir = args.mac_vocab_subdir
    if args.mac_parquet_batch_size is not None:
        rc.mac_parquet_batch_size = args.mac_parquet_batch_size
    rc.mac_label = args.mac_label
    if args.mac_vocab_path:
        rc.mac_vocab_path = args.mac_vocab_path
    if args.mac_max_samples != -1:
        rc.mac_max_samples = args.mac_max_samples
    if args.mac_val_max_samples is not None:
        rc.mac_val_max_samples = args.mac_val_max_samples
    rc.mac_val_ratio = args.mac_val_ratio

    # PosttrainConfig
    pc = cfg.posttrain
    if args.fm_checkpoint:
        pc.fm_checkpoint = args.fm_checkpoint
    if args.joint_checkpoint:
        pc.joint_checkpoint = args.joint_checkpoint
    if args.embedding_dir:
        pc.embedding_dir = args.embedding_dir

    # 设置输出目录
    if not rc.joint_output_dir:
        rc.joint_output_dir = os.path.join(cfg.output_dir, "joint_checkpoints")
    if not pc.cf_opd_output_dir:
        pc.cf_opd_output_dir = os.path.join(cfg.output_dir, "cf_opd_checkpoints")
    if not pc.embedding_dir:
        pc.embedding_dir = os.path.join(cfg.output_dir, "embeddings")

    # CFOPDConfig
    if args.num_candidates is not None:
        cfg.cf_opd.num_candidates = args.num_candidates
    if args.student_model:
        cfg.opd.student_model_name = args.student_model

    return cfg


# ════════════════════════════════════════════════════════════
# Stage 1: Joint Post-train
# ════════════════════════════════════════════════════════════

def run_joint_train(cfg: FALCONConfig, ranker_type: str, device: str) -> str:
    """
    在 MAC 数据集上对 FM + Ranker 进行联合后训练。

    Returns:
        joint_checkpoint_path: best_joint.pt 路径
    """
    from ranker.mac_data import create_mac_dataloaders
    from ranker.joint_posttrain import JointPostTrainer

    rc = cfg.ranker
    fc = cfg.fm
    output_dir = rc.joint_output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("Stage 1: Joint Post-train (FM + MACRanker on MAC dataset)")
    print("=" * 60)
    print(f"  MAC 加载模式: {rc.mac_load_mode}")
    print(f"  CVR 标签:    {rc.mac_label}")
    print(f"  Ranker 类型: {ranker_type}")
    print(f"  联合 loss:   α={rc.joint_alpha} * L_CVR + {1-rc.joint_alpha:.2f} * L_flow")
    print(f"  输出目录:    {output_dir}")

    # 加载数据
    print("\n[1/4] 准备 MAC 数据...")
    train_loader, val_loader, vocab, feature_info = create_mac_dataloaders(rc)
    print(f"  词表大小: {sum(feature_info['vocab_sizes'].values()):,}")
    print(f"  mm_feat 维度: {feature_info['mm_feat_dim']}")

    # 构建 Trainer
    print("\n[2/4] 初始化 FM + Ranker...")
    trainer = JointPostTrainer(rc, fc, device=device)
    trainer.load_fm(cfg.posttrain.fm_checkpoint or None)
    trainer.build_ranker(ranker_type, feature_info)

    # 训练
    print(f"\n[3/4] 开始联合训练 ({rc.epochs} epochs)...")
    joint_ckpt_path = trainer.train(train_loader, val_loader, vocab, output_dir)

    print(f"\n[4/4] Stage 1 完成!")
    print(f"  Joint checkpoint: {joint_ckpt_path}")
    return joint_ckpt_path


# ════════════════════════════════════════════════════════════
# Stage 2: CF-OPD
# ════════════════════════════════════════════════════════════

def run_cf_opd(cfg: FALCONConfig, device: str):
    """
    使用 joint post-trained FM + Ranker 对 LLM 做 DPO 学习。
    """
    from ranker.joint_posttrain import JointPostTrainer
    from ranker.models import create_mac_ranker

    pc = cfg.posttrain
    oc = cfg.opd
    cf = cfg.cf_opd
    fc = cfg.fm

    output_dir = pc.cf_opd_output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("Stage 2: CF-OPD (LLM DPO via joint FM+Ranker scoring)")
    print("=" * 60)

    # 加载 joint checkpoint (FM + Ranker)
    joint_ckpt_path = pc.joint_checkpoint
    if not joint_ckpt_path or not os.path.isfile(joint_ckpt_path):
        raise FileNotFoundError(
            f"CF-OPD 需要 Stage 1 的 joint checkpoint, 未找到: {joint_ckpt_path}\n"
            "请先运行 Stage 1 (joint_train), 或通过 --joint_checkpoint 指定路径"
        )

    print(f"[1/5] 加载 joint checkpoint: {joint_ckpt_path}")
    ckpt = JointPostTrainer.load_joint_checkpoint(joint_ckpt_path, device=device)

    # 重建 FM Teacher
    from fm.model import FlowMatchingModel
    fm_model = FlowMatchingModel.from_config(fc).to(device)
    fm_model.load_state_dict(ckpt["fm_state_dict"])
    fm_model.eval()

    # 重建 MACRanker
    # 需要 feature_info — 尝试从 vocab 文件恢复
    ranker_cfg_dict = ckpt.get("ranker_config", {})
    rc = cfg.ranker
    # 用 checkpoint 里的 ranker_config 覆盖当前 rc (保证结构匹配)
    for k, v in ranker_cfg_dict.items():
        if hasattr(rc, k):
            setattr(rc, k, v)

    # 词表 (必须有): 优先从 MAC/vocabs/ 目录加载, 否则读独立 json
    from ranker.mac_data import (
        FeatureVocab, SPARSE_FIELDS, SEQ_ID_FIELDS, SEQ_FLOAT_FIELDS,
    )

    vocab = None
    vocab_dir = (os.path.join(rc.mac_data_path, rc.mac_vocab_subdir)
                 if rc.mac_data_path else "")
    if vocab_dir and os.path.isdir(vocab_dir):
        vocab = FeatureVocab.load_from_dir(vocab_dir)
    elif rc.mac_vocab_path and os.path.isfile(rc.mac_vocab_path):
        vocab = FeatureVocab.load(rc.mac_vocab_path)
    else:
        raise FileNotFoundError(
            "CF-OPD 需要词表来重建 Ranker。请确保以下之一:\n"
            f"  1. --mac_data_path 指向含 {rc.mac_vocab_subdir}/ 子目录的 MAC 根目录 "
            f"(当前尝试: {vocab_dir or 'N/A'})\n"
            f"  2. --mac_vocab_path 指向词表 json 文件 (当前: {rc.mac_vocab_path or 'N/A'})"
        )

    feature_info = {
        "vocab_sizes": vocab.vocab_sizes,
        "sparse_fields": SPARSE_FIELDS,
        "seq_id_fields": SEQ_ID_FIELDS,
        "seq_float_fields": SEQ_FLOAT_FIELDS,
        "mm_feat_dim": ckpt.get("mm_feat_dim") or ranker_cfg_dict.get("mm_feat_dim", 2),
        "label_field": rc.mac_label,
    }

    ranker_type = (ckpt.get("ranker_type")
                   or ranker_cfg_dict.get("ranker_type")
                   or "deepfm")
    ranker = create_mac_ranker(ranker_type, feature_info, rc).to(device)
    ranker.load_state_dict(ckpt["ranker_state_dict"])
    ranker.eval()
    print(f"[2/5] MACRanker ({ranker_type}) 加载完成")

    # 加载 LLM Student
    print(f"[3/5] 加载 LLM Student: {oc.student_model_name}")
    from opd.student import LLMStudent
    student = LLMStudent(oc, device=device)

    # 加载 LatentPipeline
    from latent.pipeline import LatentPipeline
    latent_pipeline = LatentPipeline(cfg.latent, device=device)

    # FMTeacher 包装
    from opd.teacher import FMTeacher
    teacher = FMTeacher(fm_model, cfg.fm, device=device)

    print(f"[4/5] 初始化 CF-OPD Engine...")
    from opd.cf_opd import CFOPDEngine
    engine = CFOPDEngine(
        student=student,
        teacher=teacher,
        latent_pipeline=latent_pipeline,
        ranker=ranker,
        cf_config=cf,
        device=device,
    )

    # 加载训练数据 (用户 prompt + 对应 condition embedding)
    print(f"[5/5] 开始 CF-OPD 训练 ...")
    _run_cf_opd_loop(engine, student, cfg, output_dir, device)


def _run_cf_opd_loop(engine, student, cfg, output_dir: str, device: str):
    """
    CF-OPD 训练循环。

    数据来源: embedding_dir 下的用户 embedding (预训练阶段产出)
    每轮:
      - 从 embedding_dir 加载用户 condition embedding
      - 用 LLM Student 生成 N 条候选
      - CF-OPD Engine 打分 → DPO loss
      - 更新 LLM LoRA 参数
    """
    import json
    import glob
    from torch.utils.tensorboard import SummaryWriter
    from torch.optim import AdamW

    cf = cfg.cf_opd
    pc = cfg.posttrain

    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))
    optimizer = AdamW(
        student.trainable_parameters(),
        lr=cfg.opd.student_lr,
        weight_decay=1e-5,
    )

    # 保存 reference model (DPO 需要)
    engine.save_reference()

    # 读取 condition embedding 文件列表
    emb_dir = pc.embedding_dir
    emb_files = sorted(glob.glob(os.path.join(emb_dir, "*.pt")))
    if not emb_files:
        emb_files = sorted(glob.glob(os.path.join(emb_dir, "*.npy")))
    if not emb_files:
        raise FileNotFoundError(
            f"未找到用户 embedding 文件: {emb_dir}\n"
            "请先运行预训练阶段生成 embedding, 或通过 --embedding_dir 指定目录"
        )

    print(f"  找到 {len(emb_files)} 个 embedding 文件")
    print(f"  每步候选数: {cf.num_candidates}, 训练模式: {cf.learning_mode}")

    global_step = 0
    log_history = []

    for epoch in range(1, cfg.opd.epochs + 1):
        for emb_file in emb_files:
            # 加载 condition embedding
            if emb_file.endswith(".pt"):
                data = torch.load(emb_file, map_location=device)
                # 支持 dict 格式 {"condition": tensor} 或直接是 tensor
                if isinstance(data, dict):
                    cond_emb = data.get("condition", data.get("embedding", None))
                else:
                    cond_emb = data
            else:
                import numpy as np
                cond_emb = torch.from_numpy(np.load(emb_file)).to(device)

            if cond_emb is None:
                continue
            if cond_emb.dim() == 1:
                cond_emb = cond_emb.unsqueeze(0)   # (1, dim)

            # 构建 prompt
            prompt = _build_cf_opd_prompt(emb_file)

            # CF-OPD 单步
            optimizer.zero_grad(set_to_none=True)
            step_result = engine.cf_opd_step(
                prompt_text=prompt,
                condition_emb=cond_emb,
                optimizer=optimizer,
            )

            if step_result.get("skipped"):
                continue

            global_step += 1
            loss = step_result.get("loss", 0.0)
            n_pos = step_result.get("n_positive", 0)
            n_neg = step_result.get("n_negative", 0)

            writer.add_scalar("step/dpo_loss", loss, global_step)
            writer.add_scalar("step/n_positive", n_pos, global_step)
            writer.add_scalar("step/n_negative", n_neg, global_step)

            if global_step % 100 == 0:
                print(f"  Epoch {epoch} Step {global_step}: "
                      f"loss={loss:.4f}, pos={n_pos}, neg={n_neg}")

            log_history.append({
                "epoch": epoch, "step": global_step,
                "loss": loss, "n_pos": n_pos, "n_neg": n_neg,
            })

        # 保存每 epoch checkpoint
        student.save(os.path.join(output_dir, f"student_epoch_{epoch}.pt"))
        print(f"Epoch {epoch} 完成, checkpoint 已保存")

    # 最终保存
    student.save(os.path.join(output_dir, "final_student.pt"))
    writer.close()
    with open(os.path.join(output_dir, "cf_opd_log.json"), "w") as f:
        json.dump(log_history, f, indent=2)
    print(f"CF-OPD 训练完成: {output_dir}")


def _build_cf_opd_prompt(emb_file: str) -> str:
    """
    根据 embedding 文件名构建 prompt (简单版本)。
    实际使用时可传入对应的 jsonl 用户历史文本。
    """
    basename = os.path.splitext(os.path.basename(emb_file))[0]
    return (
        f"请根据用户 {basename} 的历史行为和偏好，"
        "生成一份全面的用户画像，包含静态信息、兴趣偏好、消费特征和需求意向。"
    )


# ════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    cfg = build_configs(args)

    logger = get_logger("posttrain")
    set_seed(cfg.seed)

    device = cfg.resolve_device()
    print(f"设备: {device}")
    print(f"阶段: {args.stage}")

    stages = []
    if args.stage == "all":
        stages = ["joint_train", "cf_opd"]
    elif args.stage == "joint_train":
        stages = ["joint_train"]
    elif args.stage == "cf_opd":
        stages = ["cf_opd"]

    joint_ckpt = cfg.posttrain.joint_checkpoint

    if "joint_train" in stages:
        joint_ckpt = run_joint_train(cfg, args.ranker_type, device)
        # 自动把 joint_ckpt 传给 cf_opd
        cfg.posttrain.joint_checkpoint = joint_ckpt

    if "cf_opd" in stages:
        if joint_ckpt:
            cfg.posttrain.joint_checkpoint = joint_ckpt
        run_cf_opd(cfg, device)

    print("\n后训练全部完成!")


if __name__ == "__main__":
    main()

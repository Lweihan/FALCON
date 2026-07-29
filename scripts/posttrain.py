#!/usr/bin/env python3
"""
posttrain.py — FALCON 后训练脚本
==================================
后训练阶段完成两件事:

  Stage 1 (opd):    OPD 蒸馏 — FM Teacher → LLM Student
      Teacher 预测 target embedding
      Student (LLM + LoRA) 生成画像文本 → 编码
      L = L_gen + α·L_emb + β·L_section + γ·L_kl
      → final_lora/

  Stage 2 (ranker): 下游 Ranker 训练
      加载预训练 FM 作为 embedding 优化器
      Ranker (DNN/DeepFM/DCN/AutoInt/TokenMixer) 联合训练
      对比: 直接预测 vs FALCON 优化后预测
      → AUC/GAUC/LogLoss/NDCG 对比报告

Usage:
  # 完整后训练
  python scripts/posttrain.py \
      --config configs/falcon_config.yaml \
      --fm_checkpoint ./output/fm_checkpoints/best_fm.pt \
      --embedding_dir ./output/embeddings

  # 仅 OPD 蒸馏
  python scripts/posttrain.py --stage opd \
      --fm_checkpoint ./output/fm_checkpoints/best_fm.pt \
      --embedding_dir ./output/embeddings

  # 仅下游 Ranker (合成数据快速验证)
  python scripts/posttrain.py --stage ranker \
      --fm_checkpoint ./output/fm_checkpoints/best_fm.pt \
      --synthetic --max_samples 20000

  # Ranker 消融实验
  python scripts/posttrain.py --stage ranker \
      --rankers dnn deepfm token_mixer \
      --fm_weights 0.1 0.5 1.0
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.config import FALCONConfig
from common.logger import get_logger
from common.utils import set_seed

logger = get_logger("posttrain")


# ════════════════════════════════════════════════════════════
# Stage 1: OPD 蒸馏
# ════════════════════════════════════════════════════════════

def run_opd_stage(config: FALCONConfig, args, fm_checkpoint: str) -> str:
    """
    FM Teacher → LLM Student 蒸馏。

    Returns:
        opd_output_dir
    """
    from opd.trainer import OPDTrainer

    embedding_dir = (
        args.embedding_dir
        or config.posttrain.embedding_dir
        or os.path.join(config.output_dir, "embeddings")
    )
    opd_output_dir = (
        args.opd_output_dir
        or config.posttrain.opd_output_dir
        or os.path.join(config.output_dir, "opd_checkpoints")
    )

    if not os.path.isfile(fm_checkpoint):
        raise FileNotFoundError(f"FM checkpoint 不存在: {fm_checkpoint}")
    if not os.path.isdir(embedding_dir):
        raise FileNotFoundError(f"Embedding 目录不存在: {embedding_dir}")

    logger.info("=" * 70)
    logger.info("Stage 1: OPD 蒸馏 (FM Teacher → LLM Student)")
    logger.info("=" * 70)
    logger.info(f"  FM checkpoint:   {fm_checkpoint}")
    logger.info(f"  Embedding 目录:  {embedding_dir}")
    logger.info(f"  Student 模型:    {config.opd.student_model_path}")
    logger.info(f"  LoRA rank:       {config.opd.lora_r}")
    logger.info(f"  蒸馏权重:        α={config.opd.distill_alpha} "
                f"β={config.opd.distill_beta} γ={config.opd.distill_gamma}")
    logger.info(f"  输出目录:        {opd_output_dir}")

    device = config.resolve_device()
    trainer = OPDTrainer(
        opd_config=config.opd,
        fm_config=config.fm,
        latent_config=config.latent,
        fm_checkpoint=fm_checkpoint,
        device=device,
    )
    trainer.train_offline(data_dir=embedding_dir, output_dir=opd_output_dir)

    return opd_output_dir


# ════════════════════════════════════════════════════════════
# Stage 2: 下游 Ranker 训练
# ════════════════════════════════════════════════════════════

def run_ranker_stage(config: FALCONConfig, args, fm_checkpoint: str) -> str:
    """
    FALCON + Ranker 联合训练与对比评估。

    Returns:
        ranker_output_dir
    """
    from ranker.data import create_ranker_dataloaders
    from ranker.trainer import RankerTrainer

    ranker_output_dir = (
        args.ranker_output_dir
        or config.posttrain.ranker_output_dir
        or os.path.join(config.output_dir, "ranker_checkpoints")
    )

    logger.info("=" * 70)
    logger.info("Stage 2: 下游 Ranker 训练 (FALCON Embedding 优化对比)")
    logger.info("=" * 70)
    logger.info(f"  FM checkpoint:   {fm_checkpoint or '(随机初始化)'}")
    logger.info(f"  Ranker 模型:     {config.ranker.ranker_types}")
    logger.info(f"  FM 权重消融:     {config.ranker.fm_weights}")
    logger.info(f"  joint_alpha:     {config.ranker.joint_alpha}")
    logger.info(f"  freeze_fm:       {config.ranker.freeze_fm}")
    logger.info(f"  数据源:          "
                f"{'合成数据' if args.synthetic else config.ranker.data_path or '合成数据'}")
    logger.info(f"  输出目录:        {ranker_output_dir}")

    device = config.resolve_device()

    train_loader, val_loader, voc_sizes, sparse_features = create_ranker_dataloaders(
        config.ranker,
        sparse_features=args.sparse_features,
        use_synthetic=args.synthetic,
    )

    trainer = RankerTrainer(
        ranker_config=config.ranker,
        fm_config=config.fm,
        fm_checkpoint=fm_checkpoint if fm_checkpoint and os.path.isfile(fm_checkpoint) else None,
        device=device,
    )
    trainer.run_all(
        train_loader=train_loader,
        val_loader=val_loader,
        feature_voc_sizes=voc_sizes,
        sparse_features=sparse_features,
        output_dir=ranker_output_dir,
    )

    return ranker_output_dir


# ════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════

def apply_cli_overrides(config: FALCONConfig, args):
    """命令行参数覆盖配置"""
    if args.output_dir:
        config.output_dir = args.output_dir
    # OPD
    if args.opd_epochs is not None:
        config.opd.epochs = args.opd_epochs
    # Ranker
    if args.rankers:
        config.ranker.ranker_types = args.rankers
    if args.fm_weights:
        config.ranker.fm_weights = args.fm_weights
    if args.joint_alpha is not None:
        config.ranker.joint_alpha = args.joint_alpha
    if args.freeze_fm:
        config.ranker.freeze_fm = True
    if args.ranker_epochs is not None:
        config.ranker.epochs = args.ranker_epochs
    if args.batch_size is not None:
        config.ranker.batch_size = args.batch_size
    if args.max_samples is not None:
        config.ranker.max_samples = args.max_samples
    if args.data_path:
        config.ranker.data_path = args.data_path
    if args.no_fast_train:
        config.ranker.fast_train = False


def main():
    parser = argparse.ArgumentParser(
        description="FALCON 后训练 (OPD 蒸馏 + 下游 Ranker)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/falcon_config.yaml")
    parser.add_argument(
        "--stage", type=str, default="all", choices=["opd", "ranker", "all"],
        help="执行阶段: opd(蒸馏) / ranker(下游) / all(全部)",
    )
    # 路径
    parser.add_argument("--fm_checkpoint", type=str, default=None, help="预训练 FM checkpoint")
    parser.add_argument("--embedding_dir", type=str, default=None)
    parser.add_argument("--opd_output_dir", type=str, default=None)
    parser.add_argument("--ranker_output_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    # OPD
    parser.add_argument("--opd_epochs", type=int, default=None)
    # Ranker
    parser.add_argument(
        "--rankers", type=str, nargs="+", default=None,
        choices=["dnn", "deepfm", "dcn", "autoint", "token_mixer"],
        help="要训练的 ranker 类型",
    )
    parser.add_argument(
        "--fm_weights", type=float, nargs="+", default=None,
        help="FM 损失权重消融列表, 如 0.1 0.5 1.0",
    )
    parser.add_argument("--joint_alpha", type=float, default=None, help="直接预测损失权重")
    parser.add_argument("--freeze_fm", action="store_true", help="冻结 FM backbone")
    parser.add_argument("--ranker_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--no_fast_train", action="store_true", help="训练时也走完整 ODE 求解")
    # 数据
    parser.add_argument("--data_path", type=str, default=None, help="排序数据集路径")
    parser.add_argument(
        "--sparse_features", type=str, nargs="+", default=None,
        help="稀疏特征列名 (使用真实数据时必填)",
    )
    parser.add_argument("--synthetic", action="store_true", help="使用合成数据快速验证")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    config = FALCONConfig.from_yaml(args.config)
    apply_cli_overrides(config, args)
    set_seed(config.seed)

    fm_checkpoint = (
        args.fm_checkpoint
        or config.posttrain.fm_checkpoint
        or os.path.join(config.output_dir, "fm_checkpoints", "best_fm.pt")
    )

    logger.info("=" * 70)
    logger.info("FALCON 后训练")
    logger.info("=" * 70)
    logger.info(f"  设备:     {config.resolve_device()}")
    logger.info(f"  随机种子: {config.seed}")
    logger.info(f"  执行阶段: {args.stage}")
    logger.info(f"  Backbone: {config.fm.backbone_type}"
                + (f" (preset={config.fm.dit_preset})" if config.fm.backbone_type == "dit" else ""))

    stages = ["opd", "ranker"] if args.stage == "all" else [args.stage]

    if "opd" in stages:
        opd_dir = run_opd_stage(config, args, fm_checkpoint)
        logger.info(f"OPD 蒸馏完成: {opd_dir}")

    if "ranker" in stages:
        ranker_dir = run_ranker_stage(config, args, fm_checkpoint)
        logger.info(f"Ranker 训练完成: {ranker_dir}")

    logger.info("")
    logger.info("后训练全部完成")


if __name__ == "__main__":
    main()

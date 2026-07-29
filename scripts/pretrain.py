#!/usr/bin/env python3
"""
pretrain.py — FALCON 预训练脚本
=================================
预训练阶段完成两件事:

  Stage 1 (latent): Latent Clue 提取
      JSONL 用户小传 → XML 结构化 → qwen3-embedding 分部分编码
                    → Attention Fusion → 融合 embedding 序列 (.pt)

  Stage 2 (fm):     FM Teacher 预训练
      融合 embedding 序列 → DiT 速度场网络
                        → OT-CFM 损失 w(t)·||v_θ - v_t||²
                        → best_fm.pt

Usage:
  # 完整预训练
  python scripts/pretrain.py \
      --config configs/falcon_config.yaml \
      --jsonl_path /path/to/output_cleaned.jsonl

  # 仅提取 latent
  python scripts/pretrain.py --stage latent --jsonl_path /path/to/data.jsonl

  # 仅训练 FM (已有 embedding)
  python scripts/pretrain.py --stage fm --embedding_dir ./output/embeddings

  # 切换 DiT 规模
  python scripts/pretrain.py --dit_preset large --use_amp --grad_ckpt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.config import FALCONConfig
from common.logger import get_logger
from common.utils import set_seed

logger = get_logger("pretrain")


# ════════════════════════════════════════════════════════════
# Stage 1: Latent Clue 提取
# ════════════════════════════════════════════════════════════

def run_latent_stage(config: FALCONConfig, args) -> str:
    """
    JSONL → 融合 embedding 序列。

    Returns:
        embedding_dir: 输出目录
    """
    import torch
    from latent.pipeline import LatentPipeline

    jsonl_path = args.jsonl_path or config.pretrain.jsonl_path
    if not jsonl_path:
        raise ValueError("必须指定 --jsonl_path 或在配置中设置 pretrain.jsonl_path")

    embedding_dir = (
        args.embedding_dir
        or config.pretrain.embedding_dir
        or os.path.join(config.output_dir, "embeddings")
    )

    logger.info("=" * 70)
    logger.info("Stage 1: Latent Clue 提取")
    logger.info("=" * 70)
    logger.info(f"  输入 JSONL:      {jsonl_path}")
    logger.info(f"  输出目录:        {embedding_dir}")
    logger.info(f"  Embedding 模型:  {config.latent.embed_model_path}")
    logger.info(f"  Embedding 维度:  {config.latent.embed_dim}")
    logger.info(f"  融合输出维度:    {config.latent.fusion_output_dim}")

    device = config.resolve_device()
    pipeline = LatentPipeline(config.latent, device=device)
    pipeline.preprocess_jsonl(
        jsonl_path=jsonl_path,
        output_dir=embedding_dir,
        max_users=args.max_users,
    )

    # 保存 fusion 权重 (OPD / Ranker / 推理阶段需要)
    fusion_path = os.path.join(embedding_dir, "fusion_weights.pt")
    torch.save(pipeline.fusion.state_dict(), fusion_path)
    logger.info(f"Fusion 权重已保存: {fusion_path}")

    return embedding_dir


# ════════════════════════════════════════════════════════════
# Stage 2: FM Teacher 预训练
# ════════════════════════════════════════════════════════════

def run_fm_stage(config: FALCONConfig, args, embedding_dir: str) -> str:
    """
    训练 Flow Matching Teacher。

    Returns:
        fm_output_dir: checkpoint 目录
    """
    from fm.trainer import FMTrainer

    fm_output_dir = (
        args.fm_output_dir
        or config.pretrain.fm_output_dir
        or os.path.join(config.output_dir, "fm_checkpoints")
    )

    if not os.path.isdir(embedding_dir):
        raise FileNotFoundError(
            f"Embedding 目录不存在: {embedding_dir}\n"
            f"请先运行 --stage latent 或指定 --embedding_dir"
        )

    logger.info("=" * 70)
    logger.info("Stage 2: FM Teacher 预训练")
    logger.info("=" * 70)
    logger.info(f"  Embedding 目录:  {embedding_dir}")
    logger.info(f"  输出目录:        {fm_output_dir}")

    device = config.resolve_device()
    trainer = FMTrainer(config.fm, device=device)
    trainer.train(
        data_dir=embedding_dir,
        output_dir=fm_output_dir,
        train_ratio=config.scenario.train_ratio,
    )

    return fm_output_dir


# ════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════

def apply_cli_overrides(config: FALCONConfig, args):
    """命令行参数覆盖配置"""
    if args.dit_preset:
        config.fm.dit_preset = args.dit_preset
        config.fm.backbone_type = "dit"
    if args.backbone:
        config.fm.backbone_type = args.backbone
    if args.epochs is not None:
        config.fm.epochs = args.epochs
    if args.batch_size is not None:
        config.fm.batch_size = args.batch_size
    if args.lr is not None:
        config.fm.lr = args.lr
    if args.use_amp:
        config.fm.use_amp = True
    if args.grad_ckpt:
        config.fm.use_gradient_checkpointing = True
    if args.output_dir:
        config.output_dir = args.output_dir


def main():
    parser = argparse.ArgumentParser(
        description="FALCON 预训练 (Latent Clue 提取 + FM Teacher)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/falcon_config.yaml")
    parser.add_argument(
        "--stage", type=str, default="all", choices=["latent", "fm", "all"],
        help="执行阶段: latent(提取) / fm(训练) / all(全部)",
    )
    # 数据路径
    parser.add_argument("--jsonl_path", type=str, default=None, help="用户小传 JSONL 路径")
    parser.add_argument("--embedding_dir", type=str, default=None, help="Embedding 输出/输入目录")
    parser.add_argument("--fm_output_dir", type=str, default=None, help="FM checkpoint 目录")
    parser.add_argument("--output_dir", type=str, default=None, help="总输出目录")
    parser.add_argument("--max_users", type=int, default=None, help="限制用户数 (调试)")
    # 模型规模
    parser.add_argument(
        "--dit_preset", type=str, default=None,
        choices=["tiny", "small", "base", "large", "xl"],
        help="DiT 规模预设",
    )
    parser.add_argument(
        "--backbone", type=str, default=None, choices=["dit", "transformer", "mlp"],
        help="Backbone 类型",
    )
    # 训练超参
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--use_amp", action="store_true", help="启用混合精度")
    parser.add_argument("--grad_ckpt", action="store_true", help="启用梯度检查点")
    args = parser.parse_args()

    config = FALCONConfig.from_yaml(args.config)
    apply_cli_overrides(config, args)
    set_seed(config.seed)

    logger.info("=" * 70)
    logger.info("FALCON 预训练")
    logger.info("  Flow matching in the Advertising scenarios")
    logger.info("  based on Latent Clue Online policy distillatioN")
    logger.info("=" * 70)
    logger.info(f"  设备:     {config.resolve_device()}")
    logger.info(f"  随机种子: {config.seed}")
    logger.info(f"  执行阶段: {args.stage}")

    stages = ["latent", "fm"] if args.stage == "all" else [args.stage]

    embedding_dir = (
        args.embedding_dir
        or config.pretrain.embedding_dir
        or os.path.join(config.output_dir, "embeddings")
    )

    if "latent" in stages:
        embedding_dir = run_latent_stage(config, args)

    if "fm" in stages:
        fm_dir = run_fm_stage(config, args, embedding_dir)
        logger.info("")
        logger.info("预训练完成，后续步骤:")
        logger.info(f"  python scripts/posttrain.py \\")
        logger.info(f"      --config {args.config} \\")
        logger.info(f"      --fm_checkpoint {os.path.join(fm_dir, 'best_fm.pt')} \\")
        logger.info(f"      --embedding_dir {embedding_dir}")


if __name__ == "__main__":
    main()

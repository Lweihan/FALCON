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
    JSONL / Parquet 目录 → 融合 embedding 序列。

    若 args.data_path 不为空，则进入 Parquet 目录模式:
      data_path 为含 day_xx.parquet 的目录，配合 start_time 推算各文件实际日期。
      xml 支持两种 tag 格式: static_information/… 或 static_info/…

    否则进入原有 JSONL 模式:
      jsonl_path 为 JSONL 文件。

    Returns:
        embedding_dir: 输出目录
    """
    import torch
    from latent.pipeline import LatentPipeline

    embedding_dir = (
        args.embedding_dir
        or config.pretrain.embedding_dir
        or os.path.join(config.output_dir, "embeddings")
    )

    logger.info("=" * 70)
    logger.info("Stage 1: Latent Clue 提取")
    logger.info("=" * 70)
    logger.info(f"  输出目录:        {embedding_dir}")
    logger.info(f"  Embedding 模型:  {config.latent.embed_model_path}")
    logger.info(f"  Embedding 维度:  {config.latent.embed_dim}")
    logger.info(f"  融合输出维度:    {config.latent.fusion_output_dim}")

    device = config.resolve_device()
    pipeline = LatentPipeline(config.latent, device=device)

    # ── Parquet 目录模式 ──────────────────────────────────────────
    if args.data_path:
        if not os.path.isdir(args.data_path):
            raise NotADirectoryError(
                f"--data_path 必须是目录（含 day_xx.parquet 文件）: {args.data_path}"
            )
        if not args.start_time:
            raise ValueError(
                "--data_path 模式下必须指定 --start_time（day_00 对应的日期，格式 YYYYMMDD）"
            )

        logger.info(f"  模式:            Parquet 目录")
        logger.info(f"  输入目录:        {args.data_path}")
        logger.info(f"  start_time:      {args.start_time}（day_00 对应日期）")

        pipeline.preprocess_parquet(
            data_dir=args.data_path,
            output_dir=embedding_dir,
            start_time=args.start_time,
            uid_col=args.uid_col,
            xml_col=args.xml_col,
            max_users=args.max_users,
            shard_size=args.shard_size,
        )

    # ── JSONL 模式（原有）──────────────────────────────────────────
    else:
        jsonl_path = args.jsonl_path or config.pretrain.jsonl_path
        if not jsonl_path:
            raise ValueError("必须指定 --jsonl_path 或在配置中设置 pretrain.jsonl_path")

        logger.info(f"  模式:            JSONL（原有流程）")
        logger.info(f"  输入 JSONL:      {jsonl_path}")

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

def run_fm_stage(config: FALCONConfig, args, embedding_dir: str, device: str = None,
                 distributed: bool = False) -> str:
    """
    训练 Flow Matching Teacher。

    Returns:
        fm_output_dir: checkpoint 目录
    """
    from common.distributed import is_main_process
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

    # 外部 val 目录：优先读 CLI 参数，再读配置，再 fallback 到 None
    val_embedding_dir = getattr(args, "val_embedding_dir", None) or \
                        getattr(config.pretrain, "val_embedding_dir", None) or None
    if val_embedding_dir and not os.path.isdir(val_embedding_dir):
        raise FileNotFoundError(
            f"验证 Embedding 目录不存在: {val_embedding_dir}"
        )

    if is_main_process():
        logger.info("=" * 70)
        logger.info("Stage 2: FM Teacher 预训练")
        logger.info("=" * 70)
        logger.info(f"  Embedding 目录:  {embedding_dir}")
        if val_embedding_dir:
            logger.info(f"  验证集目录:      {val_embedding_dir}")
        logger.info(f"  输出目录:        {fm_output_dir}")

    device = device or config.resolve_device()
    trainer = FMTrainer(config.fm, device=device, distributed=distributed)
    trainer.train(
        data_dir=embedding_dir,
        output_dir=fm_output_dir,
        train_ratio=config.scenario.train_ratio,
        val_data_dir=val_embedding_dir,
    )

    return fm_output_dir


# ════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════

def apply_cli_overrides(config: FALCONConfig, args, world_size: int = 1):
    """命令行参数覆盖配置"""
    # 模型规模
    if args.dit_preset:
        config.fm.dit_preset = args.dit_preset
        config.fm.backbone_type = "dit"
    if args.backbone:
        config.fm.backbone_type = args.backbone
    # 训练超参
    if args.epochs is not None:
        config.fm.epochs = args.epochs
    if args.lr is not None:
        config.fm.lr = args.lr
    if args.grad_accum is not None:
        config.fm.grad_accum_steps = args.grad_accum

    # batch_size: --global_batch_size 优先, 自动换算 per-GPU
    if args.global_batch_size is not None:
        accum = config.fm.grad_accum_steps
        per_gpu = max(args.global_batch_size // (world_size * accum), 1)
        config.fm.batch_size = per_gpu
    elif args.batch_size is not None:
        config.fm.batch_size = args.batch_size

    # 配对策略 / delta_t
    if args.pair_mode:
        config.fm.pair_mode = args.pair_mode
    if args.no_delta_t_condition:
        config.fm.use_delta_t_condition = False

    # 数据加载
    if args.num_workers is not None:
        config.fm.num_workers = args.num_workers
    if args.no_lazy_load:
        config.fm.lazy_load = False

    # 验证优化
    if args.val_every is not None:
        config.fm.val_every = args.val_every
    if args.val_ode_steps is not None:
        config.fm.val_ode_steps = args.val_ode_steps
    if args.val_max_batches is not None:
        config.fm.val_max_batches = args.val_max_batches
    if args.no_val_cos_sim:
        config.fm.val_compute_cos_sim = False

    # 性能
    if args.use_amp:
        config.fm.use_amp = True
    if args.grad_ckpt:
        config.fm.use_gradient_checkpointing = True
    if args.use_compile:
        config.fm.use_compile = True
    if args.no_tf32:
        config.fm.use_tf32 = False

    if args.output_dir:
        config.output_dir = args.output_dir


def main():
    parser = argparse.ArgumentParser(
        description="FALCON 预训练 (Latent Clue 提取 + FM Teacher)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
多卡训练示例:
  torchrun --nproc_per_node=4 scripts/pretrain.py --stage fm \\
      --embedding_dir ./output/embeddings --batch_size 64

GPU 利用率优化示例:
  torchrun --nproc_per_node=4 scripts/pretrain.py --stage fm \\
      --embedding_dir ./output/embeddings \\
      --batch_size 128 --num_workers 8 --use_amp \\
      --val_every 5 --val_ode_steps 10
        """,
    )
    parser.add_argument("--config", type=str, default="configs/falcon_config.yaml")
    parser.add_argument(
        "--stage", type=str, default="all", choices=["latent", "fm", "all"],
        help="执行阶段: latent(提取) / fm(训练) / all(全部)",
    )
    # 数据路径
    parser.add_argument("--jsonl_path", type=str, default=None, help="用户小传 JSONL 路径（JSONL 模式）")
    parser.add_argument("--embedding_dir", type=str, default=None, help="Embedding 输出/输入目录 (train)")
    parser.add_argument("--val_embedding_dir", type=str, default=None,
                        help="独立验证集 Embedding 目录 (已按时间预分好的 test 目录)；"
                             "指定后忽略 train_ratio，直接以此目录作为验证集")
    parser.add_argument("--fm_output_dir", type=str, default=None, help="FM checkpoint 目录")
    parser.add_argument("--output_dir", type=str, default=None, help="总输出目录")
    parser.add_argument("--max_users", type=int, default=None, help="限制用户数 (调试)")

    # ── Parquet 目录模式（--data_path 触发）────────────────────────
    parquet_group = parser.add_argument_group(
        "Parquet 目录模式 (--data_path 触发)",
        "当指定 --data_path 时进入 Parquet 目录模式：\n"
        "  data_path 为含 day_xx.parquet 的目录，每文件代表一天的用户 XML 快照\n"
        "  --start_time 指定 day_00 对应的日期，day_N = start_time - N 天\n"
        "  xml tag 支持完整格式(static_information/…)和简化格式(static_info/…)\n"
        "示例:\n"
        "  python scripts/pretrain.py --stage latent \\\n"
        "      --data_path /data/user_xml_dir \\\n"
        "      --start_time 20261106 \\\n"
        "      --embedding_dir ./output/embeddings",
    )
    parquet_group.add_argument(
        "--data_path", type=str, default=None,
        help="[Parquet 模式] 含 day_xx.parquet 的目录路径（指定后触发 Parquet 模式）",
    )
    parquet_group.add_argument(
        "--start_time", type=str, default=None,
        help="[Parquet 模式] day_00 对应的日期（格式 YYYYMMDD / YYYY-MM-DD），"
             "day_N 对应 start_time - N 天",
    )
    parquet_group.add_argument(
        "--uid_col", type=str, default="uid",
        help="[Parquet 模式] uid 列名（默认 'uid'）",
    )
    parquet_group.add_argument(
        "--xml_col", type=str, default="xml",
        help="[Parquet 模式] xml 列名（默认 'xml'）",
    )
    parquet_group.add_argument(
        "--shard_size", type=int, default=50000,
        help="[Parquet 模式] 每个 shard 文件最大用户数（默认 50000）",
    )

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
    # 配对策略
    parser.add_argument(
        "--pair_mode", type=str, default=None,
        choices=["last_target", "all_pairs", "adjacent"],
        help="配对策略: last_target(默认) / all_pairs(全配对) / adjacent(相邻)",
    )
    parser.add_argument("--no_delta_t_condition", action="store_true",
                        help="关闭 delta_t 条件注入 (退化为旧行为)")
    # 训练超参
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="per-GPU batch size")
    parser.add_argument("--global_batch_size", type=int, default=None,
                        help="全局 batch size, 自动换算 per-GPU (优先于 --batch_size)")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--grad_accum", type=int, default=None, help="梯度累积步数")
    # 数据加载
    parser.add_argument("--num_workers", type=int, default=None,
                        help="DataLoader 并行进程数 (建议 4-8)")
    parser.add_argument("--no_lazy_load", action="store_true",
                        help="关闭懒加载 (数据全部预载入内存)")
    # 验证优化
    parser.add_argument("--val_every", type=int, default=None, help="每 N epoch 验证一次")
    parser.add_argument("--val_ode_steps", type=int, default=None,
                        help="验证 ODE 步数 (越小越快, 默认 10)")
    parser.add_argument("--val_max_batches", type=int, default=None,
                        help="验证最多跑 N 个 batch (-1 全量)")
    parser.add_argument("--no_val_cos_sim", action="store_true",
                        help="跳过验证时的 ODE 求解 (验证极快)")
    # 性能
    parser.add_argument("--use_amp", action="store_true", help="启用混合精度")
    parser.add_argument("--grad_ckpt", action="store_true", help="启用梯度检查点")
    parser.add_argument("--use_compile", action="store_true", help="启用 torch.compile")
    parser.add_argument("--no_tf32", action="store_true", help="关闭 TF32")
    args = parser.parse_args()

    # ── 分布式初始化 ──
    from common.distributed import (
        cleanup_distributed, is_main_process, setup_distributed,
    )
    is_dist, rank, world_size, local_rank, device = setup_distributed()

    config = FALCONConfig.from_yaml(args.config)
    apply_cli_overrides(config, args, world_size=world_size)
    config.fm.distributed = is_dist
    set_seed(config.seed + rank)   # 各卡不同种子, 避免噪声采样完全相同

    if is_main_process():
        logger.info("=" * 70)
        logger.info("FALCON 预训练")
        logger.info("  Flow matching in the Advertising scenarios")
        logger.info("  based on Latent Clue Online policy distillatioN")
        logger.info("=" * 70)
        logger.info(f"  设备:     {device}")
        logger.info(f"  随机种子: {config.seed}")
        logger.info(f"  执行阶段: {args.stage}")
        if is_dist:
            logger.info(f"  分布式:   DDP, world_size={world_size}")

    try:
        stages = ["latent", "fm"] if args.stage == "all" else [args.stage]

        embedding_dir = (
            args.embedding_dir
            or config.pretrain.embedding_dir
            or os.path.join(config.output_dir, "embeddings")
        )

        # Latent 阶段不做多卡 (embedding 模型推理，仅主进程执行)
        if "latent" in stages:
            if is_main_process():
                embedding_dir = run_latent_stage(config, args)
            if is_dist:
                from common.distributed import barrier
                barrier()

        if "fm" in stages:
            fm_dir = run_fm_stage(config, args, embedding_dir,
                                  device=device, distributed=is_dist)
            if is_main_process():
                logger.info("")
                logger.info("预训练完成，后续步骤:")
                logger.info(f"  python scripts/posttrain.py \\")
                logger.info(f"      --config {args.config} \\")
                logger.info(f"      --fm_checkpoint {os.path.join(fm_dir, 'best_fm.pt')} \\")
                logger.info(f"      --embedding_dir {embedding_dir}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()

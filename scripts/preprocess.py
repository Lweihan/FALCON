#!/usr/bin/env python3
"""
preprocess.py — 数据预处理脚本
=================================
JSONL → Latent Pipeline → Embedding 序列 .pt 文件

Usage:
  python scripts/preprocess.py \
      --config configs/falcon_config.yaml \
      --jsonl_path /path/to/output_cleaned.jsonl
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
from common.config import FALCONConfig
from common.utils import set_seed
from common.logger import get_logger
from latent.pipeline import LatentPipeline

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description='FALCON 数据预处理')
    parser.add_argument('--config', type=str, default='configs/falcon_config.yaml')
    parser.add_argument('--jsonl_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--max_users', type=int, default=None)
    args = parser.parse_args()

    config = FALCONConfig.from_yaml(args.config)
    set_seed(config.seed)
    device = config.resolve_device()

    output_dir = args.output_dir or os.path.join(config.output_dir, 'embeddings')

    pipeline = LatentPipeline(config.latent, device=device)
    pipeline.preprocess_jsonl(
        jsonl_path=args.jsonl_path,
        output_dir=output_dir,
        max_users=args.max_users,
    )

    # 同时保存 fusion 权重 (OPD 和推理时需要)
    fusion_path = os.path.join(output_dir, 'fusion_weights.pt')
    import torch
    torch.save(pipeline.fusion.state_dict(), fusion_path)
    logger.info(f"Fusion 权重保存: {fusion_path}")


if __name__ == '__main__':
    main()

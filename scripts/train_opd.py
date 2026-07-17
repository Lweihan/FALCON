#!/usr/bin/env python3
"""
train_opd.py — 训练 OPD (Online Policy Distillation)
======================================================
将 FM Teacher 的知识蒸馏到 LLM Student。

Usage:
  python scripts/train_opd.py \
      --config configs/falcon_config.yaml \
      --fm_checkpoint ./output/fm_checkpoints/best_fm.pt \
      --data_dir ./output/embeddings
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
from common.config import FALCONConfig
from common.utils import set_seed
from common.logger import get_logger
from opd.trainer import OPDTrainer

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description='FALCON OPD 训练')
    parser.add_argument('--config', type=str, default='configs/falcon_config.yaml')
    parser.add_argument('--fm_checkpoint', type=str, required=True,
                        help='FM Teacher checkpoint 路径')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='预处理 embedding 数据目录')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    config = FALCONConfig.from_yaml(args.config)
    set_seed(config.seed)
    device = config.resolve_device()

    output_dir = args.output_dir or os.path.join(config.output_dir, 'opd_checkpoints')

    trainer = OPDTrainer(
        opd_config=config.opd,
        fm_config=config.fm,
        latent_config=config.latent,
        fm_checkpoint=args.fm_checkpoint,
        device=device,
    )

    trainer.train_offline(
        data_dir=args.data_dir,
        output_dir=output_dir,
    )


if __name__ == '__main__':
    main()

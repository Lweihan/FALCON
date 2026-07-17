#!/usr/bin/env python3
"""
train_fm.py — 训练 Flow Matching Teacher
==========================================

Usage:
  python scripts/train_fm.py \
      --config configs/falcon_config.yaml \
      --data_dir ./output/embeddings
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
from common.config import FALCONConfig
from common.utils import set_seed
from common.logger import get_logger
from fm.trainer import FMTrainer

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description='FALCON FM Teacher 训练')
    parser.add_argument('--config', type=str, default='configs/falcon_config.yaml')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    config = FALCONConfig.from_yaml(args.config)
    set_seed(config.seed)
    device = config.resolve_device()

    output_dir = args.output_dir or os.path.join(config.output_dir, 'fm_checkpoints')

    trainer = FMTrainer(config.fm, device=device)
    trainer.train(
        data_dir=args.data_dir,
        output_dir=output_dir,
        train_ratio=config.scenario.train_ratio,
    )


if __name__ == '__main__':
    main()

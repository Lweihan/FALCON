#!/usr/bin/env python3
"""
evaluate.py — 模型评估
========================

Usage:
  python scripts/evaluate.py \
      --config configs/falcon_config.yaml \
      --fm_checkpoint ./output/fm_checkpoints/best_fm.pt \
      --data_dir ./output/embeddings
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
from common.config import FALCONConfig
from common.utils import set_seed
from common.logger import get_logger
from scenario.evaluator import FALCONEvaluator
from scenario.data_loader import create_dataloader
from fm.model import FlowMatchingModel
from common.utils import load_checkpoint

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description='FALCON 评估')
    parser.add_argument('--config', type=str, default='configs/falcon_config.yaml')
    parser.add_argument('--fm_checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./eval_output')
    args = parser.parse_args()

    config = FALCONConfig.from_yaml(args.config)
    set_seed(config.seed)
    device = config.resolve_device()

    # 加载模型
    model = FlowMatchingModel(
        data_dim=config.fm.data_dim, time_dim=config.fm.time_dim,
        cond_dim=config.fm.cond_dim, hidden_dim=config.fm.hidden_dim,
        num_layers=config.fm.num_layers, dropout=0.0,
    ).to(device)
    load_checkpoint(args.fm_checkpoint, model, device=device)
    model.eval()

    # 加载验证数据
    val_loader = create_dataloader(
        args.data_dir, batch_size=32, split='val', train_ratio=config.scenario.train_ratio
    )

    # 评估
    results = FALCONEvaluator.evaluate_fm_model(model, val_loader, device)

    logger.info("=" * 50)
    logger.info("FM Teacher 评估结果:")
    logger.info(f"  Cosine Similarity: {results['cosine_sim_mean']:.4f} ± {results['cosine_sim_std']:.4f}")
    logger.info(f"  MSE:               {results['mse_mean']:.6f} ± {results['mse_std']:.6f}")
    logger.info(f"  样本数:             {results['n_samples']}")
    logger.info("=" * 50)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'eval_results.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()

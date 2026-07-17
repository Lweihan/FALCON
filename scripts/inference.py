#!/usr/bin/env python3
"""
inference.py — FALCON 端到端推理
===================================

Usage:
  # 仅 Teacher 推理
  python scripts/inference.py \
      --config configs/falcon_config.yaml \
      --fm_checkpoint ./output/fm_checkpoints/best_fm.pt \
      --embedding_file ./output/embeddings/user_000000.pt

  # Teacher + Student 推理
  python scripts/inference.py \
      --config configs/falcon_config.yaml \
      --fm_checkpoint ./output/fm_checkpoints/best_fm.pt \
      --lora_path ./output/opd_checkpoints/final_lora \
      --jsonl_path /path/to/output_cleaned.jsonl \
      --num_users 10
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np

from common.config import FALCONConfig
from common.utils import set_seed
from common.logger import get_logger

logger = get_logger(__name__)


def fm_inference(args):
    """FM Teacher 单独推理"""
    from fm.model import FlowMatchingModel
    from scenario.data_loader import UserEmbeddingDataset

    config = FALCONConfig.from_yaml(args.config)
    device = config.resolve_device()

    model = FlowMatchingModel(
        data_dim=config.fm.data_dim,
        time_dim=config.fm.time_dim,
        cond_dim=config.fm.cond_dim,
        hidden_dim=config.fm.hidden_dim,
        num_layers=config.fm.num_layers,
        dropout=0.0,
        backbone_type=config.fm.backbone_type,
        transformer_dim=config.fm.transformer_dim,
        transformer_depth=config.fm.transformer_depth,
        transformer_heads=config.fm.transformer_heads,
        transformer_mlp_ratio=config.fm.transformer_mlp_ratio,
        transformer_dropout=0.0,
    ).to(device)

    ckpt = torch.load(args.fm_checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    logger.info(f"FM 模型加载: {args.fm_checkpoint}")

    if args.embedding_file:
        data = torch.load(args.embedding_file, map_location='cpu')
        embeddings = data['embeddings']
        times = data['times']
        T = embeddings.size(0)

        logger.info(f"加载: {T} 个时间步, times={times}")

        results = []
        for i in range(T - 1):
            condition = embeddings[i].unsqueeze(0).to(device)
            with torch.no_grad():
                future_emb = model.predict_future_embedding(
                    condition, delta_t=args.delta_t, num_steps=args.num_steps
                )
            real_next = embeddings[i + 1].unsqueeze(0).to(device)
            sim = F.cosine_similarity(future_emb, real_next, dim=-1).item()

            results.append({
                'cond_step': i,
                'cond_time': times[i],
                'next_time': times[i + 1],
                'cosine_sim': sim,
            })
            logger.info(f"  Step {i} ({times[i]}): cos_sim={sim:.4f}")

        output_path = os.path.join(args.output_dir or '.', 'fm_inference.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"结果保存: {output_path}")


def full_inference(args):
    """FALCON 端到端推理 (Teacher + Student)"""
    from scenario.serving import FALCONServing
    from latent.xml_structured import load_jsonl

    config = FALCONConfig.from_yaml(args.config)
    set_seed(config.seed)

    serving = FALCONServing(
        config=config,
        fm_checkpoint=args.fm_checkpoint,
        lora_path=args.lora_path,
        fusion_checkpoint=args.fusion_checkpoint,
    )

    if args.jsonl_path:
        users = load_jsonl(args.jsonl_path, max_lines=args.num_users)
        logger.info(f"推理 {len(users)} 个用户")

        results = []
        for uid, user_data in enumerate(users):
            result = serving.predict(user_data, delta_t=args.delta_t, use_student=bool(args.lora_path))
            result['user_idx'] = uid
            results.append(result)

            if 'cosine_sim' in result:
                logger.info(f"  User {uid}: cos_sim={result['cosine_sim']:.4f}")
            elif 'teacher_target_emb' in result:
                logger.info(f"  User {uid}: teacher_emb computed")

        output_path = os.path.join(args.output_dir or '.', 'falcon_inference.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            # Convert tensors to lists for JSON serialization
            def convert(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert(i) for i in obj]
                return obj
            json.dump(convert(results), f, indent=2, ensure_ascii=False)
        logger.info(f"结果保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='FALCON 推理')
    parser.add_argument('--config', type=str, default='configs/falcon_config.yaml')
    parser.add_argument('--fm_checkpoint', type=str, required=True)
    parser.add_argument('--lora_path', type=str, default=None)
    parser.add_argument('--fusion_checkpoint', type=str, default=None)
    parser.add_argument('--embedding_file', type=str, default=None)
    parser.add_argument('--jsonl_path', type=str, default=None)
    parser.add_argument('--delta_t', type=float, default=0.5)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--num_users', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='./inference_output')
    args = parser.parse_args()

    if args.lora_path or args.jsonl_path:
        full_inference(args)
    else:
        fm_inference(args)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
inference.py — FALCON 端到端推理
===================================

模式一：生产模式（--prod）[推荐]
  输入: parquet / csv / pkl，仅需两列: user_id, emb0（今天的 embedding）
  输出: user_id + pred_future_emb（预测的未来 embedding）→ parquet / json / npy

  流程: emb0(今天) → predict_from_source(delta_t=1.0) → 预测未来 embedding
  即模型从 emb0 出发，沿 ODE 积分，预测 28 天后的 embedding 走向。

模式二：单文件模式（已转换 .pt，兼容）
  --embedding_file  单个 user_XXXXXX.pt 文件推理

模式三：Teacher + Student 端到端推理
  --lora_path / --jsonl_path

Usage:
  # 生产模式（最简）
  python scripts/inference.py --prod \\
      --fm_checkpoint ./output/fm_checkpoints/best_fm.pt \\
      --input /data/user_emb_20261001.parquet \\
      --output_dir ./inference_output

  # 指定 ODE 步数和输出格式
  python scripts/inference.py --prod \\
      --fm_checkpoint ./output/fm_checkpoints/best_fm.pt \\
      --input /data/user_emb_20261001.parquet \\
      --ode_steps 50 --output_format parquet

  # 原有 .pt 文件模式（兼容）
  python scripts/inference.py \\
      --fm_checkpoint ./output/fm_checkpoints/best_fm.pt \\
      --embedding_file ./output/embeddings/user_000000.pt
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import ast
import json
import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from common.config import FALCONConfig, FMConfig
from common.logger import get_logger

logger = get_logger(__name__)


# ════════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════════

def _parse_emb(raw) -> Optional[torch.Tensor]:
    """解析单个 embedding 值为 1D float32 Tensor，失败返回 None"""
    try:
        if raw is None:
            return None
        if isinstance(raw, float) and np.isnan(raw):
            return None
        if isinstance(raw, torch.Tensor):
            return raw.float()
        if isinstance(raw, np.ndarray):
            return torch.from_numpy(np.array(raw, dtype=np.float32))
        if isinstance(raw, list):
            return torch.tensor(raw, dtype=torch.float32)
        if isinstance(raw, str):
            return torch.tensor(ast.literal_eval(raw), dtype=torch.float32)
    except Exception:
        return None
    return None


def _load_file(path: str):
    """加载 parquet / csv / pkl → pandas DataFrame"""
    import pandas as pd
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    elif ext in (".pkl", ".pickle"):
        return pd.read_pickle(path)
    elif ext == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}，支持 parquet / csv / pkl")


def _load_model(ckpt_path: str, config_fm: FMConfig, device: str):
    """加载 FM checkpoint，优先使用内嵌配置"""
    from fm.model import FlowMatchingModel

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_cfg = ckpt.get("extra", {}).get("config", None)
    if saved_cfg is not None:
        try:
            fm_cfg = FMConfig(**{k: v for k, v in saved_cfg.items()
                                 if hasattr(FMConfig, k)})
            logger.info(f"使用 checkpoint 内嵌配置: backbone={fm_cfg.backbone_type}, "
                        f"dim={fm_cfg.data_dim}")
        except Exception:
            fm_cfg = config_fm
    else:
        fm_cfg = config_fm

    model = FlowMatchingModel.from_config(fm_cfg, dropout_override=0.0).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    val_cos = ckpt.get("extra", {}).get("val_cos_sim", None)
    logger.info(f"已加载 checkpoint: epoch={epoch}"
                + (f", val_cos={val_cos:.4f}" if val_cos else ""))
    return model, fm_cfg


# ════════════════════════════════════════════════════════════════
# 生产模式
# ════════════════════════════════════════════════════════════════

def prod_inference(args):
    """
    生产模式：输入仅需 user_id + emb0（今天的 embedding），
    预测未来 embedding 并输出。

    训练约定：
      序列顺序（旧→新）= [emb4(-28d), emb3(-21d), emb2(-14d), emb1(-7d), emb0(今天)]
      ODE 方向：x_1=旧 → x_0=新；delta_t=1.0 对应跨越全部 4 步（28天）

    推理约定：
      输入 emb0（今天）作为 source（训练时序列末端），
      delta_t=1.0 让模型预测"从当前状态出发、28天跨度的未来走向"。
      输出即为预测的 pred_future_emb。
    """
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"输入文件不存在: {args.input}")

    config = FALCONConfig.from_yaml(args.config) if os.path.exists(args.config) \
        else FALCONConfig()
    device = args.device or config.resolve_device()
    logger.info(f"设备: {device}")

    # ── 加载模型 ──
    model, fm_cfg = _load_model(args.fm_checkpoint, config.fm, device)
    data_dim = fm_cfg.data_dim
    inject_dt = fm_cfg.use_delta_t_condition

    logger.info(f"推理配置: ode_steps={args.ode_steps}, method={args.ode_method}, "
                f"delta_t=1.0 (固定), inject_delta_t={inject_dt}")

    # ── 加载数据 ──
    logger.info(f"加载数据: {args.input}")
    df = _load_file(args.input)
    logger.info(f"  {len(df)} 用户，列: {list(df.columns)}")

    if "user_id" not in df.columns:
        raise ValueError("输入文件缺少 user_id 列")
    if "emb0" not in df.columns:
        raise ValueError("输入文件缺少 emb0 列（今天的 embedding）")

    if args.max_users:
        df = df.iloc[:args.max_users]
        logger.info(f"  限制到 {len(df)} 用户（--max_users）")

    # ── 批量推理 ──
    os.makedirs(args.output_dir, exist_ok=True)
    n = len(df)
    all_user_ids: List[str] = []
    all_pred_embs: List[torch.Tensor] = []

    t0 = time.time()
    n_ok = n_skip = 0

    from tqdm import tqdm

    for batch_start in tqdm(range(0, n, args.batch_size), desc="推理", unit="batch"):
        batch_df = df.iloc[batch_start: batch_start + args.batch_size]

        batch_uids: List[str] = []
        batch_emb0: List[torch.Tensor] = []

        for _, row in batch_df.iterrows():
            emb = _parse_emb(row["emb0"])
            if emb is None:
                n_skip += 1
                continue
            if emb.shape[0] != data_dim:
                logger.debug(f"用户 {row['user_id']} 维度不匹配: "
                             f"{emb.shape[0]} != {data_dim}，跳过")
                n_skip += 1
                continue
            batch_uids.append(str(row["user_id"]))
            batch_emb0.append(emb)

        if not batch_uids:
            continue

        src_tensor = torch.stack(batch_emb0, dim=0).to(device)   # (B, dim)

        with torch.no_grad():
            pred = model.predict_from_source(
                src_tensor,
                delta_t=1.0,          # emb0 → 未来，跨越完整时间跨度
                num_steps=args.ode_steps,
                method=args.ode_method,
                inject_delta_t=inject_dt,
            )   # (B, dim)

        all_user_ids.extend(batch_uids)
        all_pred_embs.extend(pred.cpu().unbind(dim=0))
        n_ok += len(batch_uids)

    elapsed = time.time() - t0
    logger.info(f"推理完成: 成功={n_ok}, 跳过={n_skip}, 耗时={elapsed:.1f}s "
                f"({n_ok / max(elapsed, 1e-3):.0f} 用户/s)")

    if not all_pred_embs:
        logger.error("无有效输出，退出")
        return

    # ── 保存结果 ──
    pred_matrix = torch.stack(all_pred_embs, dim=0).numpy()   # (N, dim)
    _save_results(all_user_ids, pred_matrix, args.output_dir, args.output_format, data_dim)


def _save_results(
    user_ids: List[str],
    pred_matrix: np.ndarray,
    output_dir: str,
    fmt: str,
    data_dim: int,
):
    """保存推理结果到指定格式"""
    if fmt == "parquet":
        import pandas as pd
        out_path = os.path.join(output_dir, "pred_future_emb.parquet")
        out_df = pd.DataFrame({
            "user_id":        user_ids,
            "pred_future_emb": [pred_matrix[i].tolist() for i in range(len(user_ids))],
        })
        out_df.to_parquet(out_path, index=False)
        logger.info(f"输出 parquet: {out_path}  ({len(out_df)} 行, dim={data_dim})")

    elif fmt == "json":
        out_path = os.path.join(output_dir, "pred_future_emb.json")
        records = [
            {"user_id": uid, "pred_future_emb": pred_matrix[i].tolist()}
            for i, uid in enumerate(user_ids)
        ]
        with open(out_path, "w") as f:
            json.dump(records, f)
        logger.info(f"输出 JSON: {out_path}  ({len(records)} 条)")

    elif fmt == "npy":
        uid_path = os.path.join(output_dir, "pred_future_emb_uids.json")
        emb_path = os.path.join(output_dir, "pred_future_emb.npy")
        np.save(emb_path, pred_matrix)
        with open(uid_path, "w") as f:
            json.dump(user_ids, f)
        logger.info(f"输出 npy: {emb_path}  shape={pred_matrix.shape}")
        logger.info(f"对应 uid: {uid_path}")


# ════════════════════════════════════════════════════════════════
# 原有模式（兼容保留）
# ════════════════════════════════════════════════════════════════

def fm_inference(args):
    """FM Teacher 单 .pt 文件推理（原有模式）"""
    config = FALCONConfig.from_yaml(args.config) if os.path.exists(args.config) \
        else FALCONConfig()
    device = config.resolve_device()

    model, _ = _load_model(args.fm_checkpoint, config.fm, device)

    if args.embedding_file:
        data = torch.load(args.embedding_file, map_location="cpu", weights_only=False)
        embeddings = data["embeddings"]
        times = data["times"]
        T = embeddings.size(0)
        logger.info(f"加载: T={T}, times={times}")

        results = []
        for i in range(T - 1):
            condition = embeddings[i].unsqueeze(0).to(device)
            with torch.no_grad():
                if getattr(model, "use_data_as_noise", False):
                    from scenario.data_loader import compute_delta_t
                    dt = compute_delta_t(times, i, T - 1)
                    future_emb = model.predict_from_source(
                        condition, delta_t=dt, num_steps=args.num_steps
                    )
                else:
                    future_emb = model.predict_future_embedding(
                        condition, delta_t=args.delta_t, num_steps=args.num_steps
                    )
            real_next = embeddings[T - 1].unsqueeze(0).to(device)
            sim = F.cosine_similarity(future_emb, real_next, dim=-1).item()
            results.append({
                "cond_step": i, "cond_time": times[i],
                "target_time": times[T - 1], "cosine_sim": sim,
            })
            logger.info(f"  Step {i} ({times[i]} → {times[T-1]}): cos_sim={sim:.4f}")

        os.makedirs(args.output_dir or ".", exist_ok=True)
        output_path = os.path.join(args.output_dir or ".", "fm_inference.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"结果保存: {output_path}")


def full_inference(args):
    """FALCON 端到端推理 (Teacher + Student)"""
    from scenario.serving import FALCONServing
    from latent.xml_structured import load_jsonl

    config = FALCONConfig.from_yaml(args.config)
    from common.utils import set_seed
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
            result = serving.predict(
                user_data, delta_t=args.delta_t, use_student=bool(args.lora_path)
            )
            result["user_idx"] = uid
            results.append(result)
            if "cosine_sim" in result:
                logger.info(f"  User {uid}: cos_sim={result['cosine_sim']:.4f}")

        os.makedirs(args.output_dir or ".", exist_ok=True)
        output_path = os.path.join(args.output_dir or ".", "falcon_inference.json")
        with open(output_path, "w", encoding="utf-8") as f:
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


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="FALCON 推理（生产模式 / 单文件模式 / 端到端模式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── 公共参数 ──
    parser.add_argument("--config", type=str, default="configs/falcon_config.yaml")
    parser.add_argument("--fm_checkpoint", type=str, required=True,
                        help="FM Teacher checkpoint 路径")
    parser.add_argument("--output_dir", type=str, default="./inference_output",
                        help="输出目录（默认 ./inference_output）")
    parser.add_argument("--device", type=str, default=None,
                        help="推理设备（默认自动选择 cuda:0 / cpu）")

    # ── 生产模式 ──
    prod = parser.add_argument_group("生产模式 (--prod)  [推荐]")
    prod.add_argument(
        "--prod", action="store_true",
        help="启用生产模式：输入仅需 user_id + emb0，预测未来 embedding",
    )
    prod.add_argument(
        "--input", type=str, default=None,
        help="[生产模式] 输入文件路径（parquet / csv / pkl），需含 user_id, emb0 列",
    )
    prod.add_argument(
        "--output_format", type=str, default="parquet",
        choices=["parquet", "json", "npy"],
        help="[生产模式] 输出格式（默认 parquet）",
    )
    prod.add_argument(
        "--ode_steps", type=int, default=50,
        help="[生产模式] ODE 积分步数（默认 50；越多越精确，线性增加耗时）",
    )
    prod.add_argument(
        "--ode_method", type=str, default="euler", choices=["euler", "rk4"],
        help="[生产模式] ODE 方法（默认 euler；rk4 更精确但慢 4x）",
    )
    prod.add_argument(
        "--batch_size", type=int, default=512,
        help="[生产模式] 推理 batch size（默认 512）",
    )
    prod.add_argument(
        "--max_users", type=int, default=None,
        help="[生产模式] 最多处理用户数（调试用）",
    )

    # ── 原有单文件模式 ──
    legacy = parser.add_argument_group("单文件模式（原有，兼容）")
    legacy.add_argument("--embedding_file", type=str, default=None,
                        help="已转换的 .pt embedding 文件路径")
    legacy.add_argument("--delta_t", type=float, default=0.5)
    legacy.add_argument("--num_steps", type=int, default=100)

    # ── 端到端模式 ──
    e2e = parser.add_argument_group("端到端模式（Teacher + Student）")
    e2e.add_argument("--lora_path", type=str, default=None)
    e2e.add_argument("--fusion_checkpoint", type=str, default=None)
    e2e.add_argument("--jsonl_path", type=str, default=None)
    e2e.add_argument("--num_users", type=int, default=None)

    args = parser.parse_args()

    # ── 路由 ──
    if args.prod:
        if not args.input:
            parser.error("--prod 模式需要指定 --input 文件路径")
        prod_inference(args)
    elif args.lora_path or args.jsonl_path:
        full_inference(args)
    else:
        fm_inference(args)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
inference.py — FALCON 端到端推理
===================================

模式一：生产模式（--prod）[推荐]
  输入: parquet / csv / pkl，仅需两列: user_id, emb0（今天的 embedding）
  输出: user_id + pred_future_emb（预测的未来 embedding）→ parquet / json / npy

  流程: emb0(今天) → predict_from_source(delta_t=1.0) → 预测未来 embedding

  特性:
    - 多卡推理: torchrun --nproc_per_node=N 自动启用，数据按 rank 分片
    - 断点续推: --resume 跳过已完成的 part 文件，重启后接着跑
    - 周期保存: 每 --save_every 个 batch 落盘一个 part 文件
    - tqdm 进度: 仅主进程（rank=0）显示

模式二：单文件模式（已转换 .pt，兼容）
模式三：Teacher + Student 端到端推理

Usage:
  # 单卡
  python scripts/inference.py --prod \\
      --fm_checkpoint ./output/fm_checkpoints/best_fm.pt \\
      --input /data/user_emb_20261001.parquet \\
      --output_dir ./inference_output

  # 多卡（4 GPU）
  torchrun --nproc_per_node=4 scripts/inference.py --prod \\
      --fm_checkpoint ./output/fm_checkpoints/best_fm.pt \\
      --input /data/user_emb_20261001.parquet \\
      --output_dir ./inference_output

  # 断点续推（上次中断后继续）
  torchrun --nproc_per_node=4 scripts/inference.py --prod \\
      --fm_checkpoint ./output/fm_checkpoints/best_fm.pt \\
      --input /data/user_emb_20261001.parquet \\
      --output_dir ./inference_output --resume
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import ast
import glob
import json
import math
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from common.config import FALCONConfig, FMConfig
from common.logger import get_logger

logger = get_logger(__name__)


# ════════════════════════════════════════════════════════════════
# 分布式工具
# ════════════════════════════════════════════════════════════════

def _get_dist_info() -> Tuple[int, int, int]:
    """读取 torchrun 注入的环境变量，单进程时默认 rank=0/world_size=1"""
    rank       = int(os.environ.get("RANK",       0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return rank, world_size, local_rank


def _init_dist(world_size: int):
    """初始化 NCCL/Gloo 进程组（多卡时）"""
    if world_size <= 1:
        return
    import torch.distributed as dist
    if dist.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)


def _barrier(world_size: int):
    """全局 barrier（多卡时同步）"""
    if world_size <= 1:
        return
    import torch.distributed as dist
    if dist.is_initialized():
        dist.barrier()


def _cleanup_dist():
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════
# 通用工具
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
        except Exception:
            fm_cfg = config_fm
    else:
        fm_cfg = config_fm

    model = FlowMatchingModel.from_config(fm_cfg, dropout_override=0.0).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    epoch   = ckpt.get("epoch", "?")
    val_cos = ckpt.get("extra", {}).get("val_cos_sim", None)
    return model, fm_cfg, epoch, val_cos


# ════════════════════════════════════════════════════════════════
# Part 文件 I/O（用于断点续推）
# ════════════════════════════════════════════════════════════════

def _part_pattern(output_dir: str, rank: int) -> str:
    return os.path.join(output_dir, f"_part_rank{rank:04d}_*.parquet")


def _part_path(output_dir: str, rank: int, part_idx: int) -> str:
    return os.path.join(output_dir, f"_part_rank{rank:04d}_{part_idx:06d}.parquet")


def _count_resumed_rows(output_dir: str, rank: int) -> Tuple[int, int]:
    """
    统计该 rank 已完成的 part 文件行数，用于断点续推。

    Returns:
        (total_rows_done, next_part_idx)
    """
    import pyarrow.parquet as pq
    parts = sorted(glob.glob(_part_pattern(output_dir, rank)))
    total = 0
    for p in parts:
        try:
            total += pq.read_metadata(p).num_rows
        except Exception:
            pass
    return total, len(parts)


def _flush_part(output_dir: str, rank: int, part_idx: int,
                user_ids: List[str], pred_matrix: np.ndarray):
    """将 buffer 写为一个 part parquet 文件"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({
        "user_id":         pa.array(user_ids, type=pa.string()),
        "pred_future_emb": pa.array(pred_matrix.tolist(),
                                    type=pa.list_(pa.float32())),
    })
    out = _part_path(output_dir, rank, part_idx)
    pq.write_table(table, out, compression="snappy")
    return out


def _merge_rank_parts(output_dir: str, rank: int) -> str:
    """合并该 rank 所有 part 文件为单一 shard 文件，删除 parts"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    parts = sorted(glob.glob(_part_pattern(output_dir, rank)))
    if not parts:
        return None

    shard_path = os.path.join(output_dir, f"_shard_rank{rank:04d}.parquet")
    tables = [pq.read_table(p) for p in parts]
    merged = pa.concat_tables(tables)
    pq.write_table(merged, shard_path, compression="snappy")

    for p in parts:
        os.remove(p)
    return shard_path


def _merge_all_shards(output_dir: str, world_size: int, fmt: str, data_dim: int):
    """rank 0: 合并所有 rank shard 文件 → 最终输出，删除 shard 临时文件"""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pandas as pd

    shard_files = sorted(
        glob.glob(os.path.join(output_dir, "_shard_rank*.parquet"))
    )
    if not shard_files:
        logger.warning("未找到 shard 文件，跳过合并")
        return

    tables = [pq.read_table(f) for f in shard_files]
    merged = pa.concat_tables(tables)
    n = merged.num_rows

    if fmt == "parquet":
        out_path = os.path.join(output_dir, "pred_future_emb.parquet")
        pq.write_table(merged, out_path, compression="snappy")
        logger.info(f"最终输出 parquet: {out_path}  ({n} 行, dim={data_dim})")

    elif fmt == "json":
        out_path = os.path.join(output_dir, "pred_future_emb.json")
        df = merged.to_pandas()
        records = df.to_dict(orient="records")
        with open(out_path, "w") as f:
            json.dump(records, f)
        logger.info(f"最终输出 JSON: {out_path}  ({n} 条)")

    elif fmt == "npy":
        df = merged.to_pandas()
        emb_arr = np.array(df["pred_future_emb"].tolist(), dtype=np.float32)
        uid_list = df["user_id"].tolist()
        emb_path = os.path.join(output_dir, "pred_future_emb.npy")
        uid_path = os.path.join(output_dir, "pred_future_emb_uids.json")
        np.save(emb_path, emb_arr)
        with open(uid_path, "w") as f:
            json.dump(uid_list, f)
        logger.info(f"最终输出 npy: {emb_path}  shape={emb_arr.shape}")
        logger.info(f"对应 uid: {uid_path}")

    # 清理 shard 临时文件
    for f in shard_files:
        os.remove(f)


# ════════════════════════════════════════════════════════════════
# 生产模式主函数
# ════════════════════════════════════════════════════════════════

def prod_inference(args):
    """
    生产模式：输入仅需 user_id + emb0，批量预测未来 embedding。

    支持：
      - 多卡（torchrun）：每 rank 处理数据的连续分片，最后 rank 0 合并
      - 断点续推（--resume）：统计已完成 part 文件行数，跳过对应行继续推理
      - 周期落盘（--save_every）：每 N 个 batch 写一个 part 文件
      - tqdm：仅 rank 0 显示，显示该 rank 的用户级别进度
    """
    from tqdm import tqdm

    rank, world_size, local_rank = _get_dist_info()
    is_main = (rank == 0)

    # ── 分布式初始化 ──
    _init_dist(world_size)

    # ── 设备 ──
    if world_size > 1 and torch.cuda.is_available():
        device = f"cuda:{local_rank}"
    else:
        cfg_tmp = FALCONConfig.from_yaml(args.config) if os.path.exists(args.config) \
            else FALCONConfig()
        device = args.device or cfg_tmp.resolve_device()

    config = FALCONConfig.from_yaml(args.config) if os.path.exists(args.config) \
        else FALCONConfig()

    if is_main:
        logger.info(f"生产推理启动: world_size={world_size}, device={device}")

    # ── 加载模型（每 rank 独立加载，无 DDP wrapper，推理不需要梯度同步）──
    model, fm_cfg, epoch, val_cos = _load_model(args.fm_checkpoint, config.fm, device)
    data_dim  = fm_cfg.data_dim
    inject_dt = fm_cfg.use_delta_t_condition

    if is_main:
        logger.info(f"已加载 checkpoint: epoch={epoch}"
                    + (f", val_cos={val_cos:.4f}" if val_cos else ""))
        logger.info(f"推理配置: ode_steps={args.ode_steps}, method={args.ode_method}, "
                    f"delta_t=1.0(固定), inject_delta_t={inject_dt}")

    # ── 加载数据 ──
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"输入文件不存在: {args.input}")

    if is_main:
        logger.info(f"加载数据: {args.input}")
    df = _load_file(args.input)

    if "user_id" not in df.columns:
        raise ValueError("输入文件缺少 user_id 列")
    if "emb0" not in df.columns:
        raise ValueError("输入文件缺少 emb0 列（今天的 embedding）")

    if args.max_users:
        df = df.iloc[:args.max_users]

    N = len(df)
    if is_main:
        logger.info(f"  共 {N} 用户，按 {world_size} 个 rank 分片")

    # ── 数据分片：每 rank 取连续子集 ──
    chunk     = math.ceil(N / world_size)
    r_start   = rank * chunk
    r_end     = min(r_start + chunk, N)
    df_rank   = df.iloc[r_start:r_end].reset_index(drop=True)
    n_rank    = len(df_rank)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 断点续推：统计已完成 part 文件 ──
    resume_row = 0
    next_part  = 0
    if args.resume:
        resume_row, next_part = _count_resumed_rows(args.output_dir, rank)
        if resume_row > 0:
            logger.info(f"[rank{rank}] 断点续推: 跳过 {resume_row} 行，"
                        f"已有 {next_part} 个 part 文件")
        if resume_row >= n_rank:
            logger.info(f"[rank{rank}] 全部行已完成，跳过推理")
            _barrier(world_size)
            if is_main:
                _finalize(args, world_size, data_dim)
            _cleanup_dist()
            return
    else:
        # 清理旧 part/shard 文件
        for old in glob.glob(_part_pattern(args.output_dir, rank)):
            os.remove(old)

    # ── 推理循环 ──
    df_todo   = df_rank.iloc[resume_row:].reset_index(drop=True)
    n_todo    = len(df_todo)
    batch_size = args.batch_size
    save_every = max(args.save_every, 1)

    buf_uids: List[str]          = []
    buf_embs: List[torch.Tensor] = []
    n_ok = n_skip = 0
    part_idx = next_part

    # tqdm 仅 rank 0 显示，单位为用户数
    pbar = tqdm(
        total=n_todo,
        desc=f"推理 [rank{rank}]" if world_size > 1 else "推理",
        unit="用户",
        disable=not is_main,
        dynamic_ncols=True,
        smoothing=0.05,
    )

    t0 = time.time()

    def _do_flush():
        nonlocal part_idx, buf_uids, buf_embs
        if not buf_uids:
            return
        mat = torch.stack(buf_embs, dim=0).numpy()
        part_out = _flush_part(args.output_dir, rank, part_idx, buf_uids, mat)
        logger.debug(f"[rank{rank}] 落盘 part {part_idx}: {len(buf_uids)} 用户 → {part_out}")
        part_idx  += 1
        buf_uids   = []
        buf_embs   = []

    batch_count = 0
    for batch_start in range(0, n_todo, batch_size):
        batch_df = df_todo.iloc[batch_start: batch_start + batch_size]

        # 解析 emb0
        batch_uids: List[str]          = []
        batch_embs: List[torch.Tensor] = []
        for _, row in batch_df.iterrows():
            emb = _parse_emb(row["emb0"])
            if emb is None or emb.shape[0] != data_dim:
                n_skip += 1
                continue
            batch_uids.append(str(row["user_id"]))
            batch_embs.append(emb)

        if batch_uids:
            src = torch.stack(batch_embs, dim=0).to(device)
            with torch.no_grad():
                pred = model.predict_from_source(
                    src,
                    delta_t=1.0,
                    num_steps=args.ode_steps,
                    method=args.ode_method,
                    inject_delta_t=inject_dt,
                )
            buf_uids.extend(batch_uids)
            buf_embs.extend(pred.cpu().unbind(dim=0))
            n_ok += len(batch_uids)

        pbar.update(len(batch_df))
        pbar.set_postfix(ok=n_ok, skip=n_skip, parts=part_idx,
                         spd=f"{n_ok / max(time.time() - t0, 1e-3):.0f}/s")

        batch_count += 1
        if batch_count % save_every == 0:
            _do_flush()

    # 最后一批落盘
    _do_flush()
    pbar.close()

    elapsed = time.time() - t0
    logger.info(
        f"[rank{rank}] 推理完成: 成功={n_ok}, 跳过={n_skip}, "
        f"耗时={elapsed:.1f}s ({n_ok / max(elapsed, 1e-3):.0f} 用户/s), "
        f"共 {part_idx} 个 part 文件"
    )

    # ── 合并本 rank 的 part 文件 ──
    shard = _merge_rank_parts(args.output_dir, rank)
    if shard:
        logger.debug(f"[rank{rank}] part 已合并: {shard}")

    # ── 等待所有 rank 完成，rank 0 合并最终输出 ──
    _barrier(world_size)

    if is_main:
        _finalize(args, world_size, data_dim)

    _cleanup_dist()


def _finalize(args, world_size: int, data_dim: int):
    """rank 0: 合并所有 rank 的 shard → 最终输出文件"""
    logger.info("合并所有 rank 结果...")
    _merge_all_shards(args.output_dir, world_size, args.output_format, data_dim)
    logger.info("全部完成 ✓")


# ════════════════════════════════════════════════════════════════
# 原有模式（兼容保留）
# ════════════════════════════════════════════════════════════════

def fm_inference(args):
    """FM Teacher 单 .pt 文件推理（原有模式）"""
    config = FALCONConfig.from_yaml(args.config) if os.path.exists(args.config) \
        else FALCONConfig()
    device = config.resolve_device()

    model, fm_cfg, epoch, val_cos = _load_model(args.fm_checkpoint, config.fm, device)
    logger.info(f"已加载 checkpoint: epoch={epoch}"
                + (f", val_cos={val_cos:.4f}" if val_cos else ""))

    if args.embedding_file:
        data = torch.load(args.embedding_file, map_location="cpu", weights_only=False)
        embeddings = data["embeddings"]
        times      = data["times"]
        T          = embeddings.size(0)
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
                        help="推理设备（单卡时生效；多卡由 torchrun 自动分配）")

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
        help="[生产模式] ODE 积分步数（默认 50）",
    )
    prod.add_argument(
        "--ode_method", type=str, default="euler", choices=["euler", "rk4"],
        help="[生产模式] ODE 方法（默认 euler）",
    )
    prod.add_argument(
        "--batch_size", type=int, default=512,
        help="[生产模式] 推理 batch size（默认 512）",
    )
    prod.add_argument(
        "--save_every", type=int, default=20,
        help="[生产模式] 每 N 个 batch 写一次 part 文件（断点保存粒度，默认 20）",
    )
    prod.add_argument(
        "--resume", action="store_true",
        help="[生产模式] 断点续推：跳过已落盘的 part 文件，从中断处继续",
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

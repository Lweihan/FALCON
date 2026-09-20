#!/usr/bin/env python3
"""
convert_emb_data.py — 将预计算 embedding 转换为 FM 训练所需的 .pt 文件
=======================================================================
支持两种输入模式:

  ① 文件夹模式 (--input_dir)  ← 推荐，适合「每天一个 parquet」的结构
    data_dir/
      20240101.parquet    ← 文件名即日期，列: user_id + 1 个 embedding 列
      20240102.parquet
      20240103.parquet
      20240104.parquet
      20240105.parquet
    脚本自动按文件名排序、提取日期作为 times、按 user_id join 拼出时间序列。

  ② 单文件模式 (--input)      ← 兼容旧格式
    列: user_id, emb0, emb1, emb2, emb3, emb4
    emb0–emb4 对应 5 个时间步的 embedding。

输出 (两种模式相同):
  output_dir/
    user_000000.pt   ← { "embeddings": Tensor(T, dim),
    user_000001.pt        "times":      ["20240101", ...],
    ...                   "user_id":    "147353755" }

Usage:
  # ① 文件夹模式
  python scripts/convert_emb_data.py \\
      --input_dir /data/weekly_emb/ \\
      --output_dir ./output/embeddings \\
      --config configs/falcon_config.yaml

  # ② 单文件模式
  python scripts/convert_emb_data.py \\
      --input /data/user_emb.parquet \\
      --output_dir ./output/embeddings

  # 转换后直接训练 FM
  python scripts/pretrain.py --stage fm \\
      --config configs/falcon_config.yaml \\
      --embedding_dir ./output/embeddings
"""

import argparse
import ast
import os
import re
import sys
from typing import List, Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.logger import get_logger

logger = get_logger("convert_emb_data")

# 单文件模式的默认 emb 列名
_SINGLE_FILE_EMB_COLS = ["emb0", "emb1", "emb2", "emb3", "emb4"]

# 文件夹模式支持的扩展名
_SUPPORTED_EXTS = (".parquet", ".pkl", ".pickle", ".csv")


# ═══════════════════════════════════════════════════════════════
# 通用工具
# ═══════════════════════════════════════════════════════════════

def parse_emb(raw) -> torch.Tensor:
    """
    将 embedding 解析为 1D float32 Tensor。

    支持:
      - list[float]   → 直接转换
      - str "[0.032, ...]" → ast.literal_eval 后转换
      - np.ndarray    → 直接转换
      - torch.Tensor  → 直接返回
    """
    if isinstance(raw, torch.Tensor):
        return raw.float()
    if isinstance(raw, np.ndarray):
        return torch.from_numpy(raw).float()
    if isinstance(raw, list):
        return torch.tensor(raw, dtype=torch.float32)
    if isinstance(raw, str):
        try:
            lst = ast.literal_eval(raw)
            return torch.tensor(lst, dtype=torch.float32)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"无法解析 embedding 字符串: {raw[:80]}...") from e
    raise TypeError(f"不支持的 embedding 类型: {type(raw)}")


def _load_file(path: str):
    """根据扩展名加载单个文件为 pandas DataFrame"""
    import pandas as pd
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    elif ext in (".pkl", ".pickle"):
        return pd.read_pickle(path)
    elif ext == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"不支持的文件扩展名: {ext}")


def check_dim_vs_config(emb_dim: int, config_path: Optional[str]):
    """检查 embedding 维度与 FALCON 配置是否一致"""
    if not config_path or not os.path.exists(config_path):
        return
    try:
        from common.config import FALCONConfig
        cfg = FALCONConfig.from_yaml(config_path)
        cfg_dim = cfg.fm.data_dim
        if emb_dim != cfg_dim:
            logger.warning(
                f"⚠️  维度不匹配: embedding dim={emb_dim} vs config.fm.data_dim={cfg_dim}\n"
                f"   请在配置文件中修改:\n"
                f"     fm:\n"
                f"       data_dim: {emb_dim}\n"
                f"       cond_dim: {emb_dim}"
            )
        else:
            logger.info(
                f"✅ 维度检查通过: embedding dim={emb_dim} == config.fm.data_dim={cfg_dim}"
            )
    except Exception as e:
        logger.debug(f"配置检查跳过: {e}")


def _save_user_pt(out_path: str, embeddings: torch.Tensor, times: List[str], uid: str):
    torch.save(
        {
            "embeddings": embeddings,    # Tensor(T, dim)
            "times": times,              # list[str] × T，与 delta_t 计算对齐
            "user_id": str(uid),
        },
        out_path,
    )


# ═══════════════════════════════════════════════════════════════
# 模式 ①：文件夹模式
# ═══════════════════════════════════════════════════════════════

def _scan_daily_files(
    input_dir: str,
    select_dates: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """
    扫描文件夹，返回 [(date_str, abs_path), ...] 按日期升序排列。

    date_str 取文件名 stem（去扩展名），例如 "20240101"。
    支持扩展名: .parquet / .pkl / .pickle / .csv
    可通过 select_dates 只保留指定日期。
    """
    entries = []
    for fname in os.listdir(input_dir):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in _SUPPORTED_EXTS:
            continue
        # 过滤非日期文件（至少 6 位数字）
        if not re.search(r"\d{6,}", stem):
            logger.debug(f"跳过非日期文件: {fname}")
            continue
        if select_dates and stem not in select_dates:
            continue
        entries.append((stem, os.path.join(input_dir, fname)))

    if not entries:
        raise FileNotFoundError(
            f"在 {input_dir} 中未找到可用的 daily parquet 文件。\n"
            f"  支持扩展名: {_SUPPORTED_EXTS}\n"
            f"  文件名须包含 6 位以上数字（如 20240101.parquet）"
        )

    entries.sort(key=lambda x: x[0])   # 按日期字符串升序
    logger.info(f"找到 {len(entries)} 个日期文件:")
    for d, p in entries:
        logger.info(f"  {d}  →  {p}")
    return entries


def _detect_emb_col(df, date_str: str, emb_col: Optional[str] = None) -> str:
    """
    在 daily 文件中自动检测 embedding 列名。

    规则:
      1. 若指定 --emb_col，直接使用
      2. 否则取除 user_id 外唯一的非 user_id 列
      3. 若有多列（如 user_id + emb），取第一个非 user_id 列
    """
    if emb_col:
        if emb_col not in df.columns:
            raise ValueError(
                f"[{date_str}] 指定的 --emb_col='{emb_col}' 不在列中: {list(df.columns)}"
            )
        return emb_col

    cols = [c for c in df.columns if c != "user_id"]
    if not cols:
        raise ValueError(f"[{date_str}] 文件除 user_id 外没有其他列: {list(df.columns)}")
    if len(cols) > 1:
        logger.warning(
            f"[{date_str}] 检测到多个非 user_id 列 {cols}，使用第一列 '{cols[0]}'。\n"
            f"  若需指定，请使用 --emb_col 参数。"
        )
    return cols[0]


def convert_from_dir(
    input_dir: str,
    output_dir: str,
    emb_col: Optional[str] = None,
    select_dates: Optional[List[str]] = None,
    min_days: int = 0,
    max_users: Optional[int] = None,
    skip_existing: bool = False,
) -> Tuple[int, int]:
    """
    文件夹模式主流程。

    Returns:
        (emb_dim, success_count)
    """
    import pandas as pd

    daily_files = _scan_daily_files(input_dir, select_dates)
    T = len(daily_files)
    times = [d for d, _ in daily_files]     # 日期字符串作为 times（供 delta_t 计算）

    # ── 逐日加载，按 user_id 拼接 ──
    logger.info(f"加载 {T} 个日期文件并按 user_id 合并...")
    merged: Optional[pd.DataFrame] = None
    detected_emb_col: Optional[str] = None

    for i, (date_str, path) in enumerate(daily_files):
        df_day = _load_file(path)
        if "user_id" not in df_day.columns:
            raise ValueError(f"[{date_str}] 文件缺少 user_id 列: {path}")

        col = _detect_emb_col(df_day, date_str, emb_col)
        if detected_emb_col is None:
            detected_emb_col = col
        elif col != detected_emb_col:
            logger.warning(
                f"[{date_str}] embedding 列名 '{col}' 与首日 '{detected_emb_col}' 不同，"
                f"以各自列名读取"
            )

        # 重命名 emb 列为 emb_{date_str} 避免冲突
        df_day = df_day[["user_id", col]].rename(columns={col: f"emb_{date_str}"})
        df_day = df_day.drop_duplicates(subset="user_id")   # 去重，保留首条

        if merged is None:
            merged = df_day
        else:
            merged = merged.merge(df_day, on="user_id", how="outer")

        logger.info(f"  [{i+1}/{T}] {date_str}: {len(df_day)} 用户，合并后 {len(merged)} 用户")

    if merged is None or len(merged) == 0:
        raise ValueError("合并后数据为空")

    emb_day_cols = [f"emb_{d}" for d, _ in daily_files]

    # ── min_days 过滤 ──
    if min_days > 0:
        non_null = merged[emb_day_cols].notna().sum(axis=1)
        before = len(merged)
        merged = merged[non_null >= min_days]
        logger.info(f"min_days={min_days} 过滤: {before} → {len(merged)} 用户")

    if len(merged) == 0:
        raise ValueError(f"min_days={min_days} 过滤后数据为空，请降低该阈值")

    # ── 检测 embedding 维度 ──
    emb_dim = None
    for col in emb_day_cols:
        for val in merged[col]:
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                emb_dim = parse_emb(val).shape[0]
                break
        if emb_dim:
            break
    if emb_dim is None:
        raise ValueError("无法从任何列中检测 embedding 维度")
    logger.info(f"检测到 embedding 维度: {emb_dim}")

    # ── 转换为 .pt ──
    os.makedirs(output_dir, exist_ok=True)
    total = len(merged) if max_users is None else min(max_users, len(merged))
    success = skip_count = error_count = 0

    pbar = tqdm(total=total, desc="转换 embedding", unit="用户")

    for row_idx in range(total):
        row = merged.iloc[row_idx]
        uid = row["user_id"]
        out_path = os.path.join(output_dir, f"user_{row_idx:06d}.pt")

        if skip_existing and os.path.exists(out_path):
            skip_count += 1
            pbar.update(1)
            continue

        try:
            embs = []
            for col in emb_day_cols:
                val = row[col]
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    # 缺失天：填零向量
                    embs.append(torch.zeros(emb_dim, dtype=torch.float32))
                else:
                    embs.append(parse_emb(val))

            embeddings = torch.stack(embs, dim=0)   # (T, dim)
            _save_user_pt(out_path, embeddings, times, uid)
            success += 1

        except Exception as e:
            error_count += 1
            logger.debug(f"跳过 row={row_idx} user_id={uid}: {e}")

        pbar.update(1)

    pbar.close()
    logger.info(
        f"转换完成: 成功={success}, 跳过(已存在)={skip_count}, "
        f"失败={error_count} / 共 {total} 用户"
    )
    return emb_dim, success


# ═══════════════════════════════════════════════════════════════
# 模式 ②：单文件模式（向后兼容）
# ═══════════════════════════════════════════════════════════════

def convert_from_file(
    input_path: str,
    output_dir: str,
    emb_cols: Optional[List[str]] = None,
    times: Optional[List[str]] = None,
    max_users: Optional[int] = None,
    skip_existing: bool = False,
) -> Tuple[int, int]:
    """
    单文件模式主流程（user_id + emb0…embN）。

    Returns:
        (emb_dim, success_count)
    """
    df = _load_file(input_path)
    logger.info(f"加载完成: {len(df)} 行, 列: {list(df.columns)}")

    # 自动推断 emb 列
    if emb_cols is None:
        emb_cols = [c for c in df.columns if c != "user_id"]
        if not emb_cols:
            raise ValueError("单文件中未找到 user_id 以外的列")
        logger.info(f"自动检测到 emb 列: {emb_cols}")

    missing = [c for c in ["user_id"] + emb_cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame 缺少必要列: {missing}")

    T = len(emb_cols)
    if times is None:
        times = [f"day{i}" for i in range(T)]
    if len(times) != T:
        raise ValueError(f"times 长度 ({len(times)}) 必须与 emb 列数 ({T}) 一致")

    # 检测 embedding 维度
    emb_dim = None
    for idx in range(min(5, len(df))):
        raw = df.iloc[idx][emb_cols[0]]
        if raw is not None:
            emb_dim = parse_emb(raw).shape[0]
            break
    if emb_dim is None:
        raise ValueError("无法检测 embedding 维度")
    logger.info(f"检测到 embedding 维度: {emb_dim}")

    os.makedirs(output_dir, exist_ok=True)
    total = len(df) if max_users is None else min(max_users, len(df))
    success = skip_count = error_count = 0

    pbar = tqdm(total=total, desc="转换 embedding", unit="用户")

    for row_idx in range(total):
        row = df.iloc[row_idx]
        uid = row["user_id"]
        out_path = os.path.join(output_dir, f"user_{row_idx:06d}.pt")

        if skip_existing and os.path.exists(out_path):
            skip_count += 1
            pbar.update(1)
            continue

        try:
            embs = []
            for col in emb_cols:
                raw = row[col]
                if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                    embs.append(torch.zeros(emb_dim, dtype=torch.float32))
                else:
                    embs.append(parse_emb(raw))

            dims = {e.shape[0] for e in embs}
            if len(dims) > 1:
                raise ValueError(f"同一用户不同时间步 embedding 维度不一致: {dims}")

            embeddings = torch.stack(embs, dim=0)   # (T, dim)
            _save_user_pt(out_path, embeddings, times, uid)
            success += 1

        except Exception as e:
            error_count += 1
            logger.debug(f"跳过 row={row_idx} user_id={uid}: {e}")

        pbar.update(1)

    pbar.close()
    logger.info(
        f"转换完成: 成功={success}, 跳过(已存在)={skip_count}, "
        f"失败={error_count} / 共 {total} 行"
    )
    return emb_dim, success


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def _print_next_steps(output_dir: str, config_path: Optional[str]):
    config_arg = (
        f"--config {config_path}" if config_path else "--config configs/falcon_config.yaml"
    )
    print()
    print("=" * 62)
    print("转换完成！后续步骤:")
    print()
    print("  # 直接训练 FM Teacher (跳过 latent stage)")
    print(f"  python scripts/pretrain.py --stage fm \\")
    print(f"      {config_arg} \\")
    print(f"      --embedding_dir {output_dir}")
    print()
    print("  # 多卡训练")
    print(f"  torchrun --nproc_per_node=4 scripts/pretrain.py --stage fm \\")
    print(f"      {config_arg} \\")
    print(f"      --embedding_dir {output_dir} \\")
    print(f"      --batch_size 64 --use_amp")
    print("=" * 62)


def main():
    parser = argparse.ArgumentParser(
        description="将预计算 embedding 转换为 FALCON FM 训练所需 .pt 文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
文件夹模式示例 (每天一个 parquet，文件名即日期):
  python scripts/convert_emb_data.py \\
      --input_dir /data/weekly_emb/ \\
      --output_dir ./output/embeddings \\
      --config configs/falcon_config.yaml

  # 只使用指定日期
  python scripts/convert_emb_data.py \\
      --input_dir /data/weekly_emb/ \\
      --output_dir ./output/embeddings \\
      --select_dates 20240101,20240102,20240103,20240104,20240105

  # 指定每个文件中 embedding 的列名（默认自动检测）
  python scripts/convert_emb_data.py \\
      --input_dir /data/weekly_emb/ \\
      --output_dir ./output/embeddings \\
      --emb_col user_embedding

单文件模式示例 (user_id + emb0…emb4):
  python scripts/convert_emb_data.py \\
      --input /data/user_emb.parquet \\
      --output_dir ./output/embeddings
        """,
    )

    # ── 输入源（二选一）──
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--input_dir", type=str, default=None,
        help="【文件夹模式】包含按日期命名 parquet 的目录 (如 /data/weekly_emb/)",
    )
    src.add_argument(
        "--input", type=str, default=None,
        help="【单文件模式】单个 parquet/pkl/csv 文件路径",
    )

    # ── 输出 ──
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出 .pt 文件目录")

    # ── 文件夹模式专属 ──
    parser.add_argument(
        "--emb_col", type=str, default=None,
        help=(
            "【文件夹模式】每个日期文件中 embedding 的列名。"
            "默认自动检测 (user_id 以外的第一列)"
        ),
    )
    parser.add_argument(
        "--select_dates", type=str, default=None,
        help="【文件夹模式】只处理指定日期，逗号分隔 (如 20240101,20240102,...)",
    )
    parser.add_argument(
        "--min_days", type=int, default=0,
        help=(
            "【文件夹模式】用户至少出现在多少天才保留 (默认 0 = 不过滤)。"
            "设为总天数可过滤出全天在线用户"
        ),
    )

    # ── 单文件模式专属 ──
    parser.add_argument(
        "--emb_cols", type=str, default=None,
        help=(
            "【单文件模式】embedding 列名，逗号分隔 (默认: 除 user_id 外所有列)。"
            "例如: emb0,emb1,emb2,emb3,emb4"
        ),
    )
    parser.add_argument(
        "--times", type=str, default=None,
        help=(
            "【单文件模式】时间戳列表，逗号分隔，与 emb_cols 等长。"
            "默认: day0,day1,...  若有实际日期可在此指定以获得更精确的 delta_t"
        ),
    )

    # ── 通用 ──
    parser.add_argument("--config", type=str, default=None,
                        help="FALCON 配置文件路径 (用于维度一致性检查)")
    parser.add_argument("--max_users", type=int, default=None,
                        help="最多处理用户数 (调试用)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="跳过已存在的 .pt 文件 (断点续转)")

    args = parser.parse_args()

    # ── 路由到对应模式 ──
    if args.input_dir:
        # 文件夹模式
        select_dates = (
            [d.strip() for d in args.select_dates.split(",")]
            if args.select_dates else None
        )
        emb_dim, _ = convert_from_dir(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            emb_col=args.emb_col,
            select_dates=select_dates,
            min_days=args.min_days,
            max_users=args.max_users,
            skip_existing=args.skip_existing,
        )

    else:
        # 单文件模式
        emb_cols = (
            [c.strip() for c in args.emb_cols.split(",")]
            if args.emb_cols else None
        )
        times = (
            [t.strip() for t in args.times.split(",")]
            if args.times else None
        )
        emb_dim, _ = convert_from_file(
            input_path=args.input,
            output_dir=args.output_dir,
            emb_cols=emb_cols,
            times=times,
            max_users=args.max_users,
            skip_existing=args.skip_existing,
        )

    check_dim_vs_config(emb_dim, args.config)
    _print_next_steps(args.output_dir, args.config)


if __name__ == "__main__":
    main()

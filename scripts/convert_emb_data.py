#!/usr/bin/env python3
"""
convert_emb_data.py — 将预计算 embedding 转换为 FM 训练所需的 .pt 文件
=======================================================================
支持两种输入模式:

  ① 文件夹模式 (--input_dir)  ← 推荐，适合「每天一个 parquet」的结构
    data_dir/
      20240101.parquet    ← 文件名即日期，列: user_id + 1 个 embedding 列
      20240102.parquet
      ...
    脚本自动按文件名升序排序（旧→新），提取日期作为 times，按 user_id join。

  ② 单文件模式 (--input)      ← 当天 + 历史多快照合并在一个文件
    列: user_id, emb0, emb1, emb2, emb3, emb4
    其中:
      emb0 = 当天 (day  0)
      emb1 = 7 天前 (day -7)
      emb2 = 14 天前 (day -14)
      emb3 = 21 天前 (day -21)
      emb4 = 28 天前 (day -28)
    ⚠️  列在 DataFrame 中是"从新到旧"，脚本会自动反转为时间升序（旧→新）
        保存到 .pt，times = ["-28d", "-21d", "-14d", "-7d", "0d"]。
        FM 训练要求序列从旧到新，condition=早期步，x_clean=较晚步。

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

# 单文件模式：DataFrame 列顺序（新→旧）
# emb0=今天, emb1=-7d, emb2=-14d, emb3=-21d, emb4=-28d
_SINGLE_FILE_EMB_COLS_RAW = ["emb0", "emb1", "emb2", "emb3", "emb4"]

# 反转后的时间序列顺序（旧→新），与 FM 训练语义一致
# condition = 早期 embedding，x_clean = 较晚 embedding
_SINGLE_FILE_EMB_COLS = list(reversed(_SINGLE_FILE_EMB_COLS_RAW))   # [emb4, emb3, emb2, emb1, emb0]
_SINGLE_FILE_TIMES    = ["-28d", "-21d", "-14d", "-7d", "0d"]        # 与上面对应

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


def convert_from_dir(
    input_dir: str,
    output_dir: str,
    emb_col: Optional[str] = None,
    select_dates: Optional[List[str]] = None,
    min_days: int = 0,
    shard_mode: bool = True,
    max_users: Optional[int] = None,
    skip_existing: bool = False,
) -> Tuple[int, int]:
    """
    文件夹模式主流程。自动识别两种子结构：

    子模式 A — 每文件已含完整时间序列（emb0–emb4），文件名为快照日期:
      20260801.parquet: user_id, emb0(今天), emb1(-7d), ..., emb4(-28d)
      20260808.parquet: user_id, emb0(今天), emb1(-7d), ..., emb4(-28d)
      → 每个文件独立转换，反转列顺序，times 用相对天数

    子模式 B — 每文件只含单天 embedding，跨文件 join 拼时间序列:
      20260801.parquet: user_id, emb
      20260802.parquet: user_id, emb
      → 跨文件 outer join，文件名日期作为 times

    Returns:
        (emb_dim, success_count)
    """
    daily_files = _scan_daily_files(input_dir, select_dates)

    # ── 自动检测子模式：读第一个文件，看列数 ──
    first_df = _load_file(daily_files[0][1])
    non_uid_cols = [c for c in first_df.columns if c != "user_id"]
    has_standard_emb_cols = all(c in first_df.columns for c in _SINGLE_FILE_EMB_COLS_RAW)

    if has_standard_emb_cols:
        logger.info(
            f"检测到子模式 A：每文件含完整 emb0–emb4 时间序列，"
            f"共 {len(daily_files)} 个快照文件，"
            + ("分片模式（推荐）" if shard_mode else "逐用户模式")
        )
        return _convert_dir_mode_a(
            daily_files=daily_files,
            output_dir=output_dir,
            shard_mode=shard_mode,
            max_users=max_users,
            skip_existing=skip_existing,
        )
    else:
        logger.info(
            f"检测到子模式 B：每文件含单天 embedding，"
            f"共 {len(daily_files)} 个日期文件，跨文件 join 拼接时间序列。"
        )
        return _convert_dir_mode_b(
            daily_files=daily_files,
            output_dir=output_dir,
            emb_col=emb_col,
            min_days=min_days,
            max_users=max_users,
            skip_existing=skip_existing,
        )


def _convert_dir_mode_a(
    daily_files: List[Tuple[str, str]],
    output_dir: str,
    shard_mode: bool = True,
    max_users: Optional[int] = None,
    skip_existing: bool = False,
) -> Tuple[int, int]:
    """
    子模式 A：每个文件自包含完整 emb0–emb4 时间序列，逐文件独立转换。
    列顺序反转（emb4→emb0），times 使用相对天数标签。

    shard_mode=True (默认，推荐):
      每个快照文件 → 一个 shard_XXXXXX.pt，内含该文件全部用户
      shard 格式: { "embeddings": Tensor(N, 5, dim), "times": [...], "user_ids": [...] }
      优点: 文件数 = 快照数（49个），Dataset 初始化极快

    shard_mode=False:
      每用户一个 user_XXXXXX.pt（旧格式，文件数 = 用户数，不推荐大数据量场景）
    """
    os.makedirs(output_dir, exist_ok=True)
    emb_dim = None
    total_success = total_error = 0
    global_user_idx = 0

    for snap_idx, (date_str, path) in enumerate(daily_files):
        shard_path = os.path.join(output_dir, f"shard_{snap_idx:06d}.pt")

        if shard_mode and skip_existing and os.path.exists(shard_path):
            # 跳过已存在的 shard，但要累加 global_user_idx
            try:
                existing = torch.load(shard_path, map_location="cpu", weights_only=False)
                n_skip = existing["embeddings"].size(0)
                global_user_idx += n_skip
                total_success += n_skip
                logger.info(f"  [{snap_idx+1}/{len(daily_files)}] {date_str}: 跳过已存在 shard ({n_skip} 用户)")
            except Exception:
                pass
            continue

        df = _load_file(path)
        logger.info(f"  [{snap_idx+1}/{len(daily_files)}] {date_str}: {len(df)} 用户")

        if max_users is not None and global_user_idx >= max_users:
            break

        n_this = len(df) if max_users is None else min(max_users - global_user_idx, len(df))
        df = df.iloc[:n_this]

        # 检测 embedding 维度（只需一次）
        if emb_dim is None:
            for raw in df[_SINGLE_FILE_EMB_COLS_RAW[0]]:
                if raw is not None:
                    emb_dim = parse_emb(raw).shape[0]
                    logger.info(f"检测到 embedding 维度: {emb_dim}")
                    break

        # ── 构建 embeddings Tensor 和 user_ids ──
        all_embs = []
        all_uids = []
        n_err = 0

        for row_idx in range(len(df)):
            row = df.iloc[row_idx]
            uid = row["user_id"]
            try:
                embs = []
                for col in _SINGLE_FILE_EMB_COLS:   # [emb4, emb3, emb2, emb1, emb0]
                    raw = row[col]
                    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                        embs.append(torch.zeros(emb_dim, dtype=torch.float32))
                    else:
                        embs.append(parse_emb(raw))
                all_embs.append(torch.stack(embs, dim=0))   # (5, dim)
                all_uids.append(str(uid))
            except Exception as e:
                n_err += 1
                logger.debug(f"跳过 date={date_str} row={row_idx} uid={uid}: {e}")

        if not all_embs:
            logger.warning(f"  [{date_str}] 无有效用户，跳过")
            continue

        if shard_mode:
            # ── 保存为单个 shard ──
            shard_data = {
                "embeddings": torch.stack(all_embs, dim=0),   # (N, 5, dim)
                "times":      _SINGLE_FILE_TIMES,              # ["-28d",...,"0d"]
                "user_ids":   all_uids,                        # list[str] × N
                "snap_date":  date_str,                        # 快照日期，供溯源
            }
            torch.save(shard_data, shard_path)
            n_saved = len(all_embs)
            logger.info(f"    → shard_{snap_idx:06d}.pt  ({n_saved} 用户, {n_err} 失败)")
        else:
            # ── 逐用户保存（旧格式）──
            for emb_tensor, uid in zip(all_embs, all_uids):
                out_path = os.path.join(output_dir, f"user_{global_user_idx:06d}.pt")
                _save_user_pt(out_path, emb_tensor, _SINGLE_FILE_TIMES, uid)
                global_user_idx += 1
            n_saved = len(all_embs)

        total_success += n_saved
        total_error   += n_err
        global_user_idx += n_saved if shard_mode else 0  # shard 模式下 global_user_idx 仅用于 max_users

    logger.info(
        f"子模式 A 转换完成: 成功={total_success}, 失败={total_error}"
        + (f"  → {len(daily_files)} 个 shard 文件" if shard_mode else "")
    )
    return emb_dim, total_success


def _convert_dir_mode_b(
    daily_files: List[Tuple[str, str]],
    output_dir: str,
    emb_col: Optional[str] = None,
    min_days: int = 0,
    max_users: Optional[int] = None,
    skip_existing: bool = False,
) -> Tuple[int, int]:
    """
    子模式 B：每文件单天 embedding，跨文件 outer join 拼出时间序列。
    文件名日期作为 times（可被 parse_timestamp 解析，delta_t 更精确）。
    """
    import pandas as pd

    T = len(daily_files)
    times = [d for d, _ in daily_files]

    logger.info(f"加载 {T} 个日期文件并按 user_id 合并...")
    merged: Optional[pd.DataFrame] = None
    detected_emb_col: Optional[str] = None

    for i, (date_str, path) in enumerate(daily_files):
        df_day = _load_file(path)
        if "user_id" not in df_day.columns:
            raise ValueError(f"[{date_str}] 文件缺少 user_id 列: {path}")

        # 单文件只应有 1 个非 user_id 列
        cols = [c for c in df_day.columns if c != "user_id"]
        if emb_col:
            col = emb_col
        elif len(cols) == 1:
            col = cols[0]
        else:
            raise ValueError(
                f"[{date_str}] 子模式 B 期望每文件只有 1 个 embedding 列，"
                f"实际有 {len(cols)} 列: {cols}。\n"
                f"  若每文件已含完整 emb0–emb4，请确认列名为标准格式；\n"
                f"  或通过 --emb_col 指定使用哪一列。"
            )

        if detected_emb_col is None:
            detected_emb_col = col

        df_day = df_day[["user_id", col]].rename(columns={col: f"emb_{date_str}"})
        df_day = df_day.drop_duplicates(subset="user_id")

        merged = df_day if merged is None else merged.merge(df_day, on="user_id", how="outer")
        logger.info(f"  [{i+1}/{T}] {date_str}: {len(df_day)} 用户，合并后 {len(merged)} 用户")

    if merged is None or len(merged) == 0:
        raise ValueError("合并后数据为空")

    emb_day_cols = [f"emb_{d}" for d, _ in daily_files]

    if min_days > 0:
        before = len(merged)
        merged = merged[merged[emb_day_cols].notna().sum(axis=1) >= min_days]
        logger.info(f"min_days={min_days} 过滤: {before} → {len(merged)} 用户")
        if len(merged) == 0:
            raise ValueError(f"min_days={min_days} 过滤后数据为空")

    # 检测 embedding 维度
    emb_dim = None
    for col in emb_day_cols:
        for val in merged[col]:
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                emb_dim = parse_emb(val).shape[0]
                break
        if emb_dim:
            break
    if emb_dim is None:
        raise ValueError("无法检测 embedding 维度")
    logger.info(f"检测到 embedding 维度: {emb_dim}")

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
                    embs.append(torch.zeros(emb_dim, dtype=torch.float32))
                else:
                    embs.append(parse_emb(val))

            embeddings = torch.stack(embs, dim=0)
            _save_user_pt(out_path, embeddings, times, uid)
            success += 1

        except Exception as e:
            error_count += 1
            logger.debug(f"跳过 row={row_idx} uid={uid}: {e}")

        pbar.update(1)

    pbar.close()
    logger.info(
        f"子模式 B 转换完成: 成功={success}, 跳过={skip_count}, "
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
    shard_mode: bool = True,
    shard_size: int = 50000,
    max_users: Optional[int] = None,
    skip_existing: bool = False,
) -> Tuple[int, int]:
    """
    单文件模式主流程（user_id + emb0…embN）。

    shard_mode=True: 按 shard_size 分片存储，每片约 shard_size 个用户
    shard_mode=False: 每用户一个 user_XXXXXX.pt（旧格式）
    """
    df = _load_file(input_path)
    logger.info(f"加载完成: {len(df)} 行, 列: {list(df.columns)}")

    # 默认：使用反转后的列顺序（旧→新）
    if emb_cols is None:
        # 若文件包含标准 emb0–emb4 列，按约定反转；否则取 user_id 之外全部列（不反转）
        has_standard = all(c in df.columns for c in _SINGLE_FILE_EMB_COLS_RAW)
        if has_standard:
            emb_cols = _SINGLE_FILE_EMB_COLS   # [emb4, emb3, emb2, emb1, emb0]
            logger.info(
                f"检测到标准 emb0–emb4 列，已自动反转为时间升序: {emb_cols}"
            )
        else:
            emb_cols = [c for c in df.columns if c != "user_id"]
            if not emb_cols:
                raise ValueError("单文件中未找到 user_id 以外的列")
            logger.info(f"自动检测到 emb 列: {emb_cols}")

    missing = [c for c in ["user_id"] + emb_cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame 缺少必要列: {missing}")

    T = len(emb_cols)

    # times：优先外部传入，其次用标准相对天数，最后 fallback 到 day0...dayN
    if times is None:
        if emb_cols == _SINGLE_FILE_EMB_COLS:
            times = _SINGLE_FILE_TIMES   # ["-28d", "-21d", "-14d", "-7d", "0d"]
            logger.info(f"使用标准相对日期 times: {times}")
        else:
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
    success = error_count = 0

    pbar = tqdm(total=total, desc="转换 embedding", unit="用户")

    if shard_mode:
        shard_idx = 0
        shard_embs: List[torch.Tensor] = []
        shard_uids: List[str] = []

        def _flush_shard():
            nonlocal shard_idx
            if not shard_embs:
                return
            shard_path = os.path.join(output_dir, f"shard_{shard_idx:06d}.pt")
            torch.save({
                "embeddings": torch.stack(shard_embs, dim=0),
                "times":      times,
                "user_ids":   list(shard_uids),
            }, shard_path)
            shard_embs.clear()
            shard_uids.clear()
            shard_idx += 1

        for row_idx in range(total):
            row = df.iloc[row_idx]
            uid = row["user_id"]
            try:
                embs = []
                for col in emb_cols:
                    raw = row[col]
                    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                        embs.append(torch.zeros(emb_dim, dtype=torch.float32))
                    else:
                        embs.append(parse_emb(raw))
                shard_embs.append(torch.stack(embs, dim=0))
                shard_uids.append(str(uid))
                success += 1
                if len(shard_embs) >= shard_size:
                    _flush_shard()
            except Exception as e:
                error_count += 1
                logger.debug(f"跳过 row={row_idx} user_id={uid}: {e}")
            pbar.update(1)

        _flush_shard()   # 写最后一个不满 shard_size 的分片

    else:
        skip_count = 0
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

                embeddings = torch.stack(embs, dim=0)
                _save_user_pt(out_path, embeddings, times, uid)
                success += 1
            except Exception as e:
                error_count += 1
                logger.debug(f"跳过 row={row_idx} user_id={uid}: {e}")
            pbar.update(1)

    pbar.close()
    logger.info(
        f"转换完成: 成功={success}, 失败={error_count} / 共 {total} 行"
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

    # ── shard 控制 ──
    shard_group = parser.add_mutually_exclusive_group()
    shard_group.add_argument(
        "--shard", dest="shard_mode", action="store_true", default=True,
        help="启用 shard 模式（默认）：多用户合并为单个 shard_XXXXXX.pt，大幅减少文件数量",
    )
    shard_group.add_argument(
        "--no_shard", dest="shard_mode", action="store_false",
        help="禁用 shard 模式：每用户一个 user_XXXXXX.pt（旧格式，适合小规模调试）",
    )
    parser.add_argument(
        "--shard_size", type=int, default=50000,
        help="单文件模式下每个 shard 包含的用户数（默认 50000；文件夹模式下每个快照=一个 shard，忽略此参数）",
    )

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
            shard_mode=args.shard_mode,
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
            shard_mode=args.shard_mode,
            shard_size=args.shard_size,
            max_users=args.max_users,
            skip_existing=args.skip_existing,
        )

    check_dim_vs_config(emb_dim, args.config)
    _print_next_steps(args.output_dir, args.config)


if __name__ == "__main__":
    main()

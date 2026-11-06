#!/usr/bin/env python3
"""
eval_checkpoints.py — 批量评估多个预训练权重预测 emb0 的能力
=============================================================

对每个 checkpoint，分别用 emb1/emb2/emb3/emb4 作为 source，
调用 predict_from_source() 预测 emb0，计算：
  - cosine_sim:  预测 emb0 与真实 emb0 的余弦相似度
  - mse:         均方误差
  - l1:          L1 误差

结果同时打印到终端（表格形式）并保存为 JSON / CSV。

Usage:
  # 单个 checkpoint
  python scripts/eval_checkpoints.py \\
      --checkpoints ./output/fm_checkpoints/best_fm.pt \\
      --data_dir    ./output/test_embeddings \\
      --config      configs/falcon_config.yaml

  # 目录下所有 .pt（自动展开）
  python scripts/eval_checkpoints.py \\
      --checkpoints ./output/fm_checkpoints/ \\
      --data_dir    ./output/test_embeddings \\
      --ode_steps   50 \\
      --batch_size  256 \\
      --output_dir  ./eval_results

  # 多个路径混合（用空格分隔）
  python scripts/eval_checkpoints.py \\
      --checkpoints ./ckpts/best_fm.pt ./ckpts/fm_epoch_50.pt \\
      --data_dir    ./output/test_embeddings
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.config import FALCONConfig, FMConfig
from common.logger import get_logger
from scenario.data_loader import UserEmbeddingDataset, collate_fn

logger = get_logger("eval_checkpoints")


# ════════════════════════════════════════════════════════════════
# Checkpoint 加载
# ════════════════════════════════════════════════════════════════

def _infer_dit_params_from_state(state: dict) -> Optional[dict]:
    """
    从权重 state_dict 的 shape 反推 DiT 超参数（用于旧版无内嵌 config 的 checkpoint）。

    关键 key:
      velocity_net.blocks.0.attn.qkv.weight  → shape (3*dim, dim)  → dim
      velocity_net.blocks.*                  → depth（block 数量）
      velocity_net.patch_embed.weight        → shape (dim, patch_dim) → patch_dim → num_patches
      velocity_net.cond_embed.0.weight       → shape (dim, cond_dim) → cond_dim / data_dim
    """
    try:
        # 取 dim：从 qkv 权重第一维 / 3
        qkv_key = "velocity_net.blocks.0.attn.qkv.weight"
        if qkv_key not in state:
            return None
        dit_dim = state[qkv_key].shape[1]          # (3*dim, dim) → dim

        # 取 depth：收集所有 blocks.{idx}.attn.qkv.weight 中的 idx，取最大值+1
        import re
        block_indices = set()
        for k in state:
            m = re.match(r"velocity_net\.blocks\.(\d+)\.attn\.qkv\.weight", k)
            if m:
                block_indices.add(int(m.group(1)))
        depth = max(block_indices) + 1 if block_indices else 1

        # 取 patch_dim：从 patch_embed 权重
        pe_key = "velocity_net.patch_embed.weight"
        patch_dim = state[pe_key].shape[1] if pe_key in state else None  # (dim, patch_dim)

        # 取 data_dim / cond_dim：从 cond_embed 的第一层输入维度
        ce_key = "velocity_net.cond_embed.0.weight"
        cond_dim = state[ce_key].shape[1] if ce_key in state else 256    # (dim, cond_dim)
        data_dim = cond_dim  # 通常 data_dim == cond_dim

        # 取 num_patches：从 patch_type_embed shape (1, num_patches, dim)
        pt_key = "velocity_net.patch_type_embed"
        num_patches = state[pt_key].shape[1] if pt_key in state else (
            data_dim // patch_dim if patch_dim else 16
        )

        # 取 num_heads：从 ada_ln 输出维度 6*dim 无法直接读，改从 q_norm 推断
        # q_norm 的 weight shape = (head_dim,)，head_dim = dim // num_heads
        qn_key = "velocity_net.blocks.0.attn.q_norm.weight"
        if qn_key in state:
            head_dim = state[qn_key].shape[0]
            num_heads = dit_dim // head_dim
        else:
            num_heads = 16  # 默认

        return {
            "dit_dim":        dit_dim,
            "dit_depth":      depth,
            "dit_heads":      num_heads,
            "dit_num_patches": num_patches,
            "data_dim":       data_dim,
            "cond_dim":       cond_dim,
        }
    except Exception as e:
        logger.debug(f"  反推 DiT 超参失败: {e}")
        return None


def _load_model(ckpt_path: str, config_fm: FMConfig, device: str):
    """
    加载 checkpoint，自动从 extra['config'] 恢复模型超参（可覆盖外部 config）。

    Config 来源优先级（高→低）:
      1. checkpoint 内嵌的 extra['config']（best_fm.pt 和修复后的 fm_epoch_*.pt 均有）
      2. 从权重 shape 自动反推 DiT 超参（兼容旧版无内嵌 config 的 checkpoint）
      3. 外部 --config / FALCONConfig 默认值（最后兜底，可能不匹配）
    """
    from fm.model import FlowMatchingModel

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()
             if not k.startswith("_orig_mod.")}

    # ── 1. 优先用 checkpoint 内嵌配置 ──
    saved_cfg = ckpt.get("extra", {}).get("config", None)
    if saved_cfg is not None:
        try:
            fm_cfg = FMConfig(**{k: v for k, v in saved_cfg.items()
                                 if hasattr(FMConfig, k)})
            logger.info(f"  使用 checkpoint 内嵌配置: backbone={fm_cfg.backbone_type}, "
                        f"data_dim={fm_cfg.data_dim}, dit_preset={fm_cfg.dit_preset}")
        except Exception as e:
            logger.warning(f"  读取内嵌配置失败 ({e})，尝试从权重 shape 反推")
            saved_cfg = None

    # ── 2. 无内嵌 config：从权重 shape 反推 DiT 超参 ──
    if saved_cfg is None:
        inferred = _infer_dit_params_from_state(state)
        if inferred is not None:
            import copy
            fm_cfg = copy.copy(config_fm)
            fm_cfg.dit_preset = "custom"       # 禁用 preset，使用手动参数
            fm_cfg.dit_dim        = inferred["dit_dim"]
            fm_cfg.dit_depth      = inferred["dit_depth"]
            fm_cfg.dit_heads      = inferred["dit_heads"]
            fm_cfg.dit_num_patches = inferred["dit_num_patches"]
            fm_cfg.data_dim       = inferred["data_dim"]
            fm_cfg.cond_dim       = inferred["cond_dim"]
            logger.info(
                f"  从权重 shape 自动反推配置: "
                f"dim={fm_cfg.dit_dim}, depth={fm_cfg.dit_depth}, "
                f"heads={fm_cfg.dit_heads}, patches={fm_cfg.dit_num_patches}, "
                f"data_dim={fm_cfg.data_dim}"
            )
        else:
            fm_cfg = config_fm
            logger.warning(
                f"  无法反推超参，使用外部 config（preset={fm_cfg.dit_preset}）。"
                f"  如出现维度错误，请用 --dit_preset large 等参数手动指定。"
            )

    model = FlowMatchingModel.from_config(fm_cfg, dropout_override=0.0).to(device)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"  缺失 key: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        logger.warning(f"  多余 key: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    # 严格检查：关键层 shape 是否匹配，给出明确提示
    for key in ["velocity_net.blocks.0.attn.qkv.weight",
                "velocity_net.patch_embed.weight"]:
        if key in state:
            ckpt_shape = tuple(state[key].shape)
            model_param = dict(model.named_parameters()).get(key)
            if model_param is not None and tuple(model_param.shape) != ckpt_shape:
                raise RuntimeError(
                    f"维度不匹配: {key} checkpoint={ckpt_shape} vs model={tuple(model_param.shape)}\n"
                    f"请检查 --dit_preset / --config 是否与训练时一致。"
                )

    model.eval()
    epoch = ckpt.get("epoch", "?")
    val_cos = ckpt.get("extra", {}).get("val_cos_sim", None)
    logger.info(f"  已加载: epoch={epoch}"
                + (f", 保存时 val_cos={val_cos:.4f}" if val_cos else ""))
    return model


# ════════════════════════════════════════════════════════════════
# 单 checkpoint 评估
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_one_checkpoint(
    model,
    data_dir: str,
    device: str,
    ode_steps: int = 50,
    ode_method: str = "euler",
    batch_size: int = 128,
    num_workers: int = 0,
    max_batches: int = -1,
    inject_delta_t: bool = True,
) -> Dict:
    """
    对评估集做完整的 emb_k → emb0 预测评估（k=1,2,3,4）。

    数据集要求：已被 convert_emb_data.py 转换为 shard/user .pt 格式，
    times = ["-28d","-21d","-14d","-7d","0d"]，序列长度 T=5。

    返回字典包含:
      per_source:  {emb1→emb0: {cos_mean, cos_std, mse_mean, l1_mean}, ...}
      overall:     所有 source 汇总的平均指标
      n_samples:   总样本数
    """
    # 使用 all_pairs 加载全量数据，__getitem__ 里有 (i,j) 信息
    # 但这里我们需要按固定 source 索引评估，所以直接遍历 shard/user 文件
    dataset = UserEmbeddingDataset(
        data_dir,
        pair_mode="last_target",
        use_all=True,
        lazy_load=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory="cuda" in device,
    )

    # 按 step_gap / 真实 delta_t 分组统计
    # step_gap=1 → emb(T-2)→emb(T-1), ..., step_gap=4 → emb0→emb4
    # 这里我们只关心目标是最后一步（emb0, idx=T-1=4）的配对
    # 筛选条件：j == T-1（x_clean 是最新 emb0）
    # 可从 step_gap 和 num_timesteps 推断：j = i + step_gap，j == T-1

    buckets: Dict[int, Dict[str, List[float]]] = {}  # step_gap → metrics list

    t0 = time.time()
    n_total = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="  评估中", leave=False)):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        x_clean   = batch["x_clean"].to(device)    # (B, dim)  — 较新步 embedding
        cond      = batch["condition"].to(device)   # (B, dim)  — 较旧步 embedding
        step_gap  = batch["step_gap"]               # (B,) LongTensor
        delta_t   = batch["delta_t"].to(device)     # (B,)
        num_ts    = batch["num_timesteps"]          # list[int]

        # 筛选 j == T-1（x_clean 是序列最后一步，即最新的 emb0）
        # j = (T-1) ⟺ step_gap + i == T-1 且 i ≥ 0
        # 简化：(T-1) - step_gap == i ≥ 0
        T_arr = torch.tensor(num_ts, dtype=torch.long)
        j_idx = T_arr - 1
        i_idx = j_idx - step_gap
        mask = i_idx >= 0   # 正常对；last_target 占位时 i=-1 已被 Dataset 替换
        # 额外确认 j 是最后一步
        # 注意：all_pairs 包含所有 (i,j)，这里只取 j == T-1 的对
        # 用 step_gap + i_idx == T-1 再验证（i_idx 已从上面推出，必然满足）
        if not mask.any():
            continue

        idx = mask.nonzero(as_tuple=True)[0]
        x0     = x_clean[idx]      # 真实 emb0
        src    = cond[idx]         # 历史 emb（source）
        sg     = step_gap[idx]     # step_gap，对应 delta_t 信息
        dt     = delta_t[idx]      # 归一化时间跨度

        # 逐 step_gap 分组推理
        for sg_val in sg.unique():
            g_mask = (sg == sg_val)
            src_g  = src[g_mask]
            x0_g   = x0[g_mask]
            dt_g   = dt[g_mask]

            if inject_delta_t:
                # 每个样本用自己的 delta_t；batch 内可能有混合值
                # 逐样本推理（仅在 batch 内 delta_t 有差异时）
                dt_mean = float(dt_g.mean().item())
                pred = model.predict_from_source(
                    src_g, delta_t=dt_mean,
                    num_steps=ode_steps, method=ode_method,
                    inject_delta_t=True,
                )
            else:
                pred = model.predict_from_source(
                    src_g, delta_t=1.0,
                    num_steps=ode_steps, method=ode_method,
                    inject_delta_t=False,
                )

            cos = F.cosine_similarity(pred, x0_g, dim=-1).cpu()    # (n,)
            mse = F.mse_loss(pred, x0_g, reduction="none").mean(-1).cpu()
            l1  = F.l1_loss(pred, x0_g, reduction="none").mean(-1).cpu()

            sg_key = int(sg_val.item())
            if sg_key not in buckets:
                buckets[sg_key] = {"cos": [], "mse": [], "l1": []}
            buckets[sg_key]["cos"].extend(cos.tolist())
            buckets[sg_key]["mse"].extend(mse.tolist())
            buckets[sg_key]["l1"].extend(l1.tolist())
            n_total += int(g_mask.sum())

    elapsed = time.time() - t0

    # ── 汇总结果 ──
    def _stats(vals):
        t = torch.tensor(vals)
        return {
            "mean": float(t.mean()),
            "std":  float(t.std()),
            "min":  float(t.min()),
            "max":  float(t.max()),
            "n":    len(vals),
        }

    # step_gap → source 名（T=5时: gap=1→emb3→emb0, gap=4→emb4→emb0？）
    # times=["-28d","-21d","-14d","-7d","0d"], idx=4 是 emb0
    # source idx = 4 - step_gap
    # gap=1: idx=3 → "-7d"  (emb1 in raw naming)
    # gap=2: idx=2 → "-14d" (emb2)
    # gap=3: idx=1 → "-21d" (emb3)
    # gap=4: idx=0 → "-28d" (emb4)
    _gap_to_label = {1: "emb1(−7d)→emb0", 2: "emb2(−14d)→emb0",
                     3: "emb3(−21d)→emb0", 4: "emb4(−28d)→emb0"}

    per_source = {}
    all_cos, all_mse, all_l1 = [], [], []
    for sg_key in sorted(buckets):
        b = buckets[sg_key]
        label = _gap_to_label.get(sg_key, f"gap={sg_key}→emb0")
        per_source[label] = {
            "cos_sim":  _stats(b["cos"]),
            "mse":      _stats(b["mse"]),
            "l1":       _stats(b["l1"]),
        }
        all_cos.extend(b["cos"])
        all_mse.extend(b["mse"])
        all_l1.extend(b["l1"])

    overall = {
        "cos_sim": _stats(all_cos) if all_cos else {},
        "mse":     _stats(all_mse) if all_mse else {},
        "l1":      _stats(all_l1) if all_l1 else {},
    }

    return {
        "per_source": per_source,
        "overall":    overall,
        "n_samples":  n_total,
        "elapsed_s":  round(elapsed, 2),
        "ode_steps":  ode_steps,
    }


# ════════════════════════════════════════════════════════════════
# 多 checkpoint 批量评估
# ════════════════════════════════════════════════════════════════

def _expand_checkpoints(paths: List[str]) -> List[str]:
    """展开路径列表：目录 → 目录内全部 .pt；文件 → 直接加入"""
    result = []
    for p in paths:
        if os.path.isdir(p):
            pts = sorted(
                os.path.join(p, f)
                for f in os.listdir(p)
                if f.endswith(".pt") and not f.startswith("fusion")
            )
            if not pts:
                logger.warning(f"目录 {p} 中未找到 .pt 文件")
            result.extend(pts)
        elif os.path.isfile(p):
            result.append(p)
        else:
            logger.warning(f"路径不存在，跳过: {p}")
    return result


def _print_table(all_results: Dict[str, Dict]):
    """打印对比表格"""
    # 收集所有 source 标签
    source_labels = []
    for res in all_results.values():
        for lbl in res.get("per_source", {}):
            if lbl not in source_labels:
                source_labels.append(lbl)
    source_labels.sort()

    print()
    print("=" * 100)
    print("  Checkpoint 对比评估结果 (按 val cosine_sim 排序)")
    print("=" * 100)

    # 按 overall cos_sim 降序排列
    sorted_items = sorted(
        all_results.items(),
        key=lambda x: x[1].get("overall", {}).get("cos_sim", {}).get("mean", -1),
        reverse=True,
    )

    # 表头
    col_w = 18
    ckpt_w = 38
    header = f"{'Checkpoint':<{ckpt_w}}"
    for lbl in source_labels:
        short = lbl.split("→")[0].strip()
        header += f"  {short:>{col_w}}"
    header += f"  {'Overall cos':>{col_w}}  {'MSE':>{col_w}}"
    print(header)
    print("-" * len(header))

    for ckpt_path, res in sorted_items:
        ckpt_name = os.path.basename(ckpt_path)
        row = f"{ckpt_name:<{ckpt_w}}"
        for lbl in source_labels:
            ps = res.get("per_source", {}).get(lbl, {})
            cos = ps.get("cos_sim", {}).get("mean", float("nan"))
            row += f"  {cos:>{col_w}.4f}"
        ov_cos = res.get("overall", {}).get("cos_sim", {}).get("mean", float("nan"))
        ov_mse = res.get("overall", {}).get("mse", {}).get("mean", float("nan"))
        row += f"  {ov_cos:>{col_w}.4f}  {ov_mse:>{col_w}.6f}"
        print(row)

    print("=" * 100)
    print()


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="批量评估多个 FM checkpoint 预测 emb0 的能力",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 评估单个 checkpoint
  python scripts/eval_checkpoints.py \\
      --checkpoints ./output/fm_checkpoints/best_fm.pt \\
      --data_dir    ./output/test_embeddings

  # 评估整个目录下所有 checkpoint
  python scripts/eval_checkpoints.py \\
      --checkpoints ./output/fm_checkpoints/ \\
      --data_dir    ./output/test_embeddings \\
      --ode_steps   50 --batch_size 256

  # 多个路径混合
  python scripts/eval_checkpoints.py \\
      --checkpoints ./ckpts/best_fm.pt ./ckpts/fm_epoch_50.pt ./ckpts/fm_epoch_100.pt \\
      --data_dir    ./output/test_embeddings \\
      --output_dir  ./eval_results
        """,
    )

    # 必要参数
    parser.add_argument(
        "--checkpoints", nargs="+", required=True,
        help="checkpoint 路径（.pt 文件 或 包含 .pt 的目录），可传多个",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="评估集目录（已被 convert_emb_data.py 转换为 .pt 格式）",
    )

    # 推理参数
    parser.add_argument("--ode_steps", type=int, default=50,
                        help="ODE 积分步数（默认 50；越多越精确但越慢）")
    parser.add_argument("--ode_method", type=str, default="euler",
                        choices=["euler", "rk4"],
                        help="ODE 求解方法（默认 euler）")
    parser.add_argument("--no_inject_delta_t", action="store_true",
                        help="关闭 delta_t 条件注入（需与训练配置一致）")

    # 数据加载
    parser.add_argument("--batch_size", type=int, default=128,
                        help="推理 batch size（默认 128）")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader worker 数（默认 0）")
    parser.add_argument("--max_batches", type=int, default=-1,
                        help="每个 checkpoint 最多评估多少 batch（-1=全量，调试用）")

    # 配置与输出
    parser.add_argument("--config", type=str, default="configs/falcon_config.yaml",
                        help="FALCON 配置文件路径（模型超参 fallback 来源）")
    parser.add_argument("--device", type=str, default=None,
                        help="推理设备（默认自动选择 cuda:0 / cpu）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="评估结果保存目录（JSON + CSV），默认不保存")

    # 模型超参手动覆盖（用于旧版无内嵌 config 的 checkpoint）
    override_group = parser.add_argument_group(
        "模型超参覆盖（旧版 checkpoint 无内嵌 config 时使用）"
    )
    override_group.add_argument(
        "--dit_preset", type=str, default=None,
        choices=["tiny", "small", "base", "large", "xl"],
        help="手动指定 DiT preset（覆盖 --config 和自动反推，用于无 config 的旧 checkpoint）",
    )
    override_group.add_argument(
        "--data_dim", type=int, default=None,
        help="手动指定 data_dim（覆盖 --config）",
    )

    args = parser.parse_args()

    # ── 配置 ──
    if os.path.exists(args.config):
        config = FALCONConfig.from_yaml(args.config)
    else:
        logger.warning(f"配置文件不存在: {args.config}，使用默认配置")
        config = FALCONConfig()

    # ── 手动覆盖 dit_preset / data_dim（旧 checkpoint 无内嵌 config 时使用）──
    if args.dit_preset is not None:
        config.fm.dit_preset = args.dit_preset
        logger.info(f"手动覆盖 dit_preset={args.dit_preset}")
    if args.data_dim is not None:
        config.fm.data_dim = args.data_dim
        config.fm.cond_dim = args.data_dim
        logger.info(f"手动覆盖 data_dim={args.data_dim}")

    device = args.device or config.resolve_device()
    logger.info(f"推理设备: {device}")

    # ── 展开 checkpoint 列表 ──
    ckpt_paths = _expand_checkpoints(args.checkpoints)
    if not ckpt_paths:
        logger.error("未找到任何 checkpoint 文件，退出")
        sys.exit(1)
    logger.info(f"待评估 checkpoint 数: {len(ckpt_paths)}")
    for p in ckpt_paths:
        logger.info(f"  {p}")

    # ── 验证 data_dir ──
    if not os.path.isdir(args.data_dir):
        logger.error(f"评估集目录不存在: {args.data_dir}")
        sys.exit(1)

    inject_delta_t = not args.no_inject_delta_t

    # ── 逐 checkpoint 评估 ──
    all_results = {}

    for i, ckpt_path in enumerate(ckpt_paths):
        ckpt_name = os.path.basename(ckpt_path)
        logger.info(f"\n[{i+1}/{len(ckpt_paths)}] 评估: {ckpt_name}")

        try:
            model = _load_model(ckpt_path, config.fm, device)
        except Exception as e:
            logger.error(f"  加载失败: {e}")
            all_results[ckpt_path] = {"error": str(e)}
            continue

        try:
            result = eval_one_checkpoint(
                model=model,
                data_dir=args.data_dir,
                device=device,
                ode_steps=args.ode_steps,
                ode_method=args.ode_method,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_batches=args.max_batches,
                inject_delta_t=inject_delta_t,
            )
            result["ckpt_path"] = ckpt_path
            all_results[ckpt_path] = result

            # 单个 checkpoint 简要结果
            logger.info(f"  样本数: {result['n_samples']}, 耗时: {result['elapsed_s']}s")
            for src_lbl, metrics in result["per_source"].items():
                cos = metrics["cos_sim"]["mean"]
                mse = metrics["mse"]["mean"]
                logger.info(f"    {src_lbl:30s}  cos={cos:.4f}  mse={mse:.6f}")
            ov = result["overall"]
            logger.info(f"    {'Overall':30s}  cos={ov['cos_sim']['mean']:.4f}"
                        f"  mse={ov['mse']['mean']:.6f}")
        except Exception as e:
            logger.error(f"  评估失败: {e}")
            import traceback
            traceback.print_exc()
            all_results[ckpt_path] = {"error": str(e)}

        # 释放显存
        del model
        if "cuda" in device:
            torch.cuda.empty_cache()

    # ── 打印对比表格 ──
    valid_results = {k: v for k, v in all_results.items() if "error" not in v}
    if len(valid_results) > 1:
        _print_table(valid_results)
    elif len(valid_results) == 1:
        ckpt_path, res = next(iter(valid_results.items()))
        print()
        print(f"评估结果: {os.path.basename(ckpt_path)}")
        for src_lbl, metrics in res["per_source"].items():
            print(f"  {src_lbl:30s}  cos={metrics['cos_sim']['mean']:.4f}"
                  f"  mse={metrics['mse']['mean']:.6f}")
        ov = res["overall"]
        print(f"  {'Overall':30s}  cos={ov['cos_sim']['mean']:.4f}"
              f"  mse={ov['mse']['mean']:.6f}")

    # ── 保存结果 ──
    if args.output_dir and all_results:
        os.makedirs(args.output_dir, exist_ok=True)

        # JSON — 完整结果
        json_path = os.path.join(args.output_dir, "eval_results.json")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"完整结果已保存: {json_path}")

        # CSV — 每行一个 (checkpoint × source) 条目，方便 Excel / pandas 分析
        csv_path = os.path.join(args.output_dir, "eval_results.csv")
        rows = []
        for ckpt_path, res in all_results.items():
            if "error" in res:
                continue
            ckpt_name = os.path.basename(ckpt_path)
            for src_lbl, metrics in res["per_source"].items():
                rows.append({
                    "checkpoint":  ckpt_name,
                    "source":      src_lbl,
                    "cos_mean":    round(metrics["cos_sim"]["mean"], 6),
                    "cos_std":     round(metrics["cos_sim"]["std"],  6),
                    "mse_mean":    round(metrics["mse"]["mean"],     8),
                    "l1_mean":     round(metrics["l1"]["mean"],      8),
                    "n_samples":   metrics["cos_sim"]["n"],
                })
            ov = res["overall"]
            rows.append({
                "checkpoint": ckpt_name,
                "source":     "Overall",
                "cos_mean":   round(ov["cos_sim"]["mean"], 6),
                "cos_std":    round(ov["cos_sim"]["std"],  6),
                "mse_mean":   round(ov["mse"]["mean"],     8),
                "l1_mean":    round(ov["l1"]["mean"],      8),
                "n_samples":  ov["cos_sim"]["n"],
            })

        try:
            import csv
            if rows:
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)
                logger.info(f"CSV 已保存: {csv_path}")
        except Exception as e:
            logger.warning(f"CSV 保存失败: {e}")

    # ── 失败汇报 ──
    failed = [k for k, v in all_results.items() if "error" in v]
    if failed:
        logger.warning(f"\n以下 {len(failed)} 个 checkpoint 失败:")
        for p in failed:
            logger.warning(f"  {p}: {all_results[p]['error']}")


if __name__ == "__main__":
    main()

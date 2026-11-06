"""
pipeline.py — 潜在线索提取完整管线
====================================
封装 XML 结构化 → 分部分 Embedding → 注意力融合的端到端流程。

支持两种输入格式:
  1. JSONL (原始格式): 每行一个用户，含多时间步 records
  2. Parquet (新格式):  两列 uid + xml，每行一个用户快照，
     xml tag 支持完整格式(static_information/…)和简化格式(static_info/…)
"""

import os
import torch
from collections import defaultdict
from tqdm import tqdm
from typing import Optional

from common.logger import get_logger
from common.config import LatentConfig
from .xml_structured import load_jsonl, user_to_xml_sections, parse_profile_xml
from .embedder import ProfileEmbedder, create_embedder
from .attention_fusion import AttentionFusion

logger = get_logger(__name__)


class LatentPipeline:
    """
    潜在线索提取管线: JSONL/Parquet → XML → Part Embeddings → Fused Embedding。

    支持:
    - encode_user(): 单用户编码，返回时间序列 embedding
    - preprocess_jsonl(): 批量预处理 JSONL → .pt 文件（原有接口）
    - preprocess_parquet(): 批量预处理 Parquet → shard .pt 文件（新接口）
    - encode_text_sections(): 对已分段的文本直接编码（OPD 蒸馏时使用）
    """

    def __init__(self, config: LatentConfig, device: str = 'cuda:0'):
        self.config = config
        self.device = device

        self.embedder = create_embedder(
            model_path=config.embed_model_path,
            device=device,
            max_length=config.max_length,
            embed_dim=config.embed_dim,
        )
        self.fusion = AttentionFusion(
            embed_dim=config.embed_dim,
            hidden_dim=config.fusion_hidden_dim,
            output_dim=config.fusion_output_dim,
            num_heads=config.fusion_num_heads,
            dropout=config.fusion_dropout,
        ).to(device)

    # ─────────────────────────────────────────────────────────────
    # 原有接口
    # ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_user(self, user_data: dict) -> dict:
        """
        编码单个用户，返回时间序列 embedding。

        Returns:
            {
                "embeddings": Tensor(T, output_dim),
                "times": list[str],
                "section_embs": list[dict],   # 每个时间步 4 部分原始 embedding
            }
        """
        sections = user_to_xml_sections(user_data)
        encoded = self.embedder.encode_sections(sections, show_progress=False)

        fused_embs = []
        section_embs = []
        times = []

        for enc in encoded:
            part_dict = {k: v.unsqueeze(0).float().to(self.device) for k, v in enc.items() if k.endswith('_emb')}
            fused = self.fusion.forward_from_dict(part_dict).squeeze(0).cpu()
            fused_embs.append(fused)
            section_embs.append({k: v.cpu().float() for k, v in part_dict.items()})
            times.append(enc['time'])

        return {
            "embeddings": torch.stack(fused_embs, dim=0),
            "times": times,
            "section_embs": section_embs,
        }

    def preprocess_jsonl(
        self,
        jsonl_path: str,
        output_dir: str,
        max_users: Optional[int] = None,
    ):
        """批量预处理 JSONL → .pt 文件"""
        os.makedirs(output_dir, exist_ok=True)
        users = load_jsonl(jsonl_path, max_lines=max_users)
        logger.info(f"预处理 {len(users)} 个用户 → {output_dir}")

        for uid, user_data in enumerate(tqdm(users, desc="预处理")):
            result = self.encode_user(user_data)
            if result["embeddings"].size(0) == 0:
                continue
            torch.save(result, os.path.join(output_dir, f"user_{uid:06d}.pt"))

        logger.info(f"预处理完成: {output_dir}")

    # ─────────────────────────────────────────────────────────────
    # 新接口：Parquet → shard .pt
    # ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def preprocess_parquet(
        self,
        parquet_path: str,
        output_dir: str,
        uid_col: str = 'uid',
        xml_col: str = 'xml',
        time_col: Optional[str] = None,
        max_users: Optional[int] = None,
        shard_size: int = 50000,
        batch_size: int = 64,
    ):
        """
        从 Parquet 文件批量提取 latent embeddings，输出 shard .pt 文件。

        Parquet 格式要求:
          - uid_col  (默认 'uid'):  用户 ID
          - xml_col  (默认 'xml'):  XML 字符串，每行一个用户快照
            XML 支持两种 tag 格式:
              完整格式: <static_information> <interest_profile>
                        <consumption_profile> <demand_targets>
              简化格式: <static_info> <interest> <consumption> <demand>
          - time_col (可选):  快照时间列，用于对同一用户的多行按时间排序；
            若未指定则自动尝试 snap_date / date / time / dt / ds；
            若均无则按行序处理。

        多快照处理:
          同一 uid 的多行被视为该用户不同时间步的快照，按 time_col 排序后
          拼接为 Tensor(T, dim)，T 由该用户的行数决定。

        输出:
          shard_XXXXXX.pt，格式与 convert_emb_data.py 一致:
          {
            "embeddings": Tensor(N, T, dim),
            "times":      [str, ...],   # 长度 T，取自第一位用户的时间标签
            "user_ids":   [str, ...],   # 长度 N
          }
          同一 shard 内所有用户具有相同 T，T 不同的用户分别写入不同 shard。

        Args:
            parquet_path:  输入 parquet 文件路径
            output_dir:    输出目录
            uid_col:       uid 列名
            xml_col:       xml 列名
            time_col:      时间列名（None 则自动检测）
            max_users:     最多处理的用户数（调试用）
            shard_size:    每个 shard 文件包含的最大用户数
            batch_size:    Embedding 模型的推理 batch size
        """
        import pandas as pd

        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"加载 Parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        logger.info(f"  共 {len(df)} 行, 列: {list(df.columns)}")

        # ── 自动检测时间列 ──
        if time_col is None:
            for candidate in ('snap_date', 'date', 'time', 'dt', 'ds'):
                if candidate in df.columns:
                    time_col = candidate
                    logger.info(f"  自动检测时间列: '{time_col}'")
                    break

        if time_col and time_col not in df.columns:
            logger.warning(f"  指定的时间列 '{time_col}' 不存在，改为按行序处理")
            time_col = None

        # ── 按 uid 分组 ──
        grouped = df.groupby(uid_col, sort=False)
        all_uids = list(grouped.groups.keys())
        if max_users:
            all_uids = all_uids[:max_users]
        logger.info(f"  共 {len(all_uids)} 个唯一用户 → {output_dir}")

        # ── shard 缓冲：按 T 分桶，保证同 shard 内 T 一致 ──
        # 结构: {T: {"embs": [...], "uids": [...], "times": [...], "shard_idx": N}}
        t_buffers: dict = {}
        shard_counters: dict = defaultdict(int)   # {T: shard 序号}
        total_shards = 0

        def _flush_t_buffer(T: int):
            nonlocal total_shards
            buf = t_buffers[T]
            if not buf["embs"]:
                return
            shard_path = os.path.join(
                output_dir, f"shard_{total_shards:06d}.pt"
            )
            torch.save(
                {
                    "embeddings": torch.stack(buf["embs"], dim=0),  # (N, T, dim)
                    "times":      buf["times"],
                    "user_ids":   buf["uids"],
                },
                shard_path,
            )
            logger.debug(
                f"  shard {total_shards:06d}: T={T}, "
                f"用户数={len(buf['uids'])} → {shard_path}"
            )
            total_shards += 1
            shard_counters[T] += 1
            t_buffers[T] = {"embs": [], "uids": [], "times": buf["times"]}

        # ── 逐用户处理 ──
        for uid in tqdm(all_uids, desc="提取 latent (parquet)", unit="用户"):
            user_df = grouped.get_group(uid)

            # 按时间列排序
            if time_col:
                user_df = user_df.sort_values(time_col)

            xml_list = user_df[xml_col].tolist()
            if time_col:
                raw_times = user_df[time_col].astype(str).tolist()
            else:
                raw_times = [str(i) for i in range(len(xml_list))]

            if not xml_list:
                continue

            # ── 解析 XML → sections ──
            sections = []
            for xml_str, t_label in zip(xml_list, raw_times):
                parsed = parse_profile_xml(str(xml_str))
                sections.append({
                    'time':                t_label,
                    'static_information':  parsed['static_information'],
                    'interest_profile':    parsed['interest_profile'],
                    'consumption_profile': parsed['consumption_profile'],
                    'demand_targets':      parsed['demand_targets'],
                })

            if not sections:
                continue

            # ── 编码（复用 ProfileEmbedder.encode_sections）──
            try:
                encoded = self.embedder.encode_sections(
                    sections, batch_size=batch_size, show_progress=False
                )
            except Exception as e:
                logger.warning(f"  uid={uid} 编码失败，跳过: {e}")
                continue

            # ── 融合 ──
            fused_list = []
            for enc in encoded:
                part_dict = {
                    k: v.unsqueeze(0).float().to(self.device)
                    for k, v in enc.items()
                    if k.endswith('_emb')
                }
                try:
                    fused = self.fusion.forward_from_dict(part_dict).squeeze(0).cpu()
                except Exception as e:
                    logger.warning(f"  uid={uid} 融合失败，跳过: {e}")
                    fused_list = []
                    break
                fused_list.append(fused)

            if not fused_list:
                continue

            user_emb = torch.stack(fused_list, dim=0)  # (T_user, dim)
            T_user = user_emb.size(0)
            times_for_user = [enc['time'] for enc in encoded]

            # ── 加入对应 T 的缓冲 ──
            if T_user not in t_buffers:
                t_buffers[T_user] = {
                    "embs":  [],
                    "uids":  [],
                    "times": times_for_user,   # 该 T 下第一位用户的时间标签
                }
            t_buffers[T_user]["embs"].append(user_emb)
            t_buffers[T_user]["uids"].append(str(uid))

            if len(t_buffers[T_user]["embs"]) >= shard_size:
                _flush_t_buffer(T_user)

        # ── 末尾 flush ──
        for T_val in list(t_buffers.keys()):
            _flush_t_buffer(T_val)

        # ── 汇总 ──
        t_summary = {T: cnt for T, cnt in shard_counters.items() if cnt > 0}
        logger.info(
            f"preprocess_parquet 完成: 共 {total_shards} 个 shard 文件\n"
            + "  ".join(f"T={T}: {cnt} shards" for T, cnt in sorted(t_summary.items()))
        )

    # ─────────────────────────────────────────────────────────────
    # OPD 辅助接口
    # ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_text_sections(
        self,
        sections: list[dict],
    ) -> torch.Tensor:
        """
        对已分段的文本 (OPD student 生成的) 进行编码并融合。
        每项 dict 包含 static_information, interest_profile, consumption_profile, demand_targets 文本。

        Returns:
            Tensor(batch, output_dim) 融合 embedding
        """
        # 收集各部分文本
        section_texts = {
            'static_information': [],
            'interest_profile': [],
            'consumption_profile': [],
            'demand_targets': [],
        }
        for sec in sections:
            for name in section_texts:
                section_texts[name].append(sec.get(name, ''))

        # 批量编码
        section_embs = {}
        for name in section_texts:
            embs = self.embedder.encode_texts(
                section_texts[name], show_progress=False, desc=f"[Emb] {name}"
            )
            section_embs[name] = embs.float().to(self.device)

        # 注意力融合
        part_dict = {
            'static_information_emb': section_embs['static_information'],
            'interest_profile_emb': section_embs['interest_profile'],
            'consumption_profile_emb': section_embs['consumption_profile'],
            'demand_targets_emb': section_embs['demand_targets'],
        }
        fused = self.fusion.forward_from_dict(part_dict)
        return fused.cpu()

    def load_fusion_state(self, path: str):
        """加载预训练的融合模块权重"""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.fusion.load_state_dict(state)
        logger.info(f"加载 fusion 权重: {path}")


class LatentPipeline:
    """
    潜在线索提取管线: JSONL → XML → Part Embeddings → Fused Embedding。

    支持:
    - encode_user(): 单用户编码，返回时间序列 embedding
    - preprocess_jsonl(): 批量预处理并保存 .pt
    - encode_text_sections(): 对已分段的文本直接编码（OPD 蒸馏时使用）
    """

    def __init__(self, config: LatentConfig, device: str = 'cuda:0'):
        self.config = config
        self.device = device

        self.embedder = create_embedder(
            model_path=config.embed_model_path,
            device=device,
            max_length=config.max_length,
            embed_dim=config.embed_dim,
        )
        self.fusion = AttentionFusion(
            embed_dim=config.embed_dim,
            hidden_dim=config.fusion_hidden_dim,
            output_dim=config.fusion_output_dim,
            num_heads=config.fusion_num_heads,
            dropout=config.fusion_dropout,
        ).to(device)

    @torch.no_grad()
    def encode_user(self, user_data: dict) -> dict:
        """
        编码单个用户，返回时间序列 embedding。

        Returns:
            {
                "embeddings": Tensor(T, output_dim),
                "times": list[str],
                "section_embs": list[dict],   # 每个时间步 4 部分原始 embedding
            }
        """
        sections = user_to_xml_sections(user_data)
        encoded = self.embedder.encode_sections(sections, show_progress=False)

        fused_embs = []
        section_embs = []
        times = []

        for enc in encoded:
            part_dict = {k: v.unsqueeze(0).float().to(self.device) for k, v in enc.items() if k.endswith('_emb')}
            fused = self.fusion.forward_from_dict(part_dict).squeeze(0).cpu()
            fused_embs.append(fused)
            section_embs.append({k: v.cpu().float() for k, v in part_dict.items()})
            times.append(enc['time'])

        return {
            "embeddings": torch.stack(fused_embs, dim=0),
            "times": times,
            "section_embs": section_embs,
        }

    def preprocess_jsonl(
        self,
        jsonl_path: str,
        output_dir: str,
        max_users: Optional[int] = None,
    ):
        """批量预处理 JSONL → .pt 文件"""
        os.makedirs(output_dir, exist_ok=True)
        users = load_jsonl(jsonl_path, max_lines=max_users)
        logger.info(f"预处理 {len(users)} 个用户 → {output_dir}")

        for uid, user_data in enumerate(tqdm(users, desc="预处理")):
            result = self.encode_user(user_data)
            if result["embeddings"].size(0) == 0:
                continue
            torch.save(result, os.path.join(output_dir, f"user_{uid:06d}.pt"))

        logger.info(f"预处理完成: {output_dir}")

    @torch.no_grad()
    def encode_text_sections(
        self,
        sections: list[dict],
    ) -> torch.Tensor:
        """
        对已分段的文本 (OPD student 生成的) 进行编码并融合。
        每项 dict 包含 static_information, interest_profile, consumption_profile, demand_targets 文本。

        Returns:
            Tensor(batch, output_dim) 融合 embedding
        """
        # 收集各部分文本
        section_texts = {
            'static_information': [],
            'interest_profile': [],
            'consumption_profile': [],
            'demand_targets': [],
        }
        for sec in sections:
            for name in section_texts:
                section_texts[name].append(sec.get(name, ''))

        # 批量编码
        section_embs = {}
        for name in section_texts:
            embs = self.embedder.encode_texts(
                section_texts[name], show_progress=False, desc=f"[Emb] {name}"
            )
            section_embs[name] = embs.float().to(self.device)

        # 注意力融合
        batch_size = len(sections)
        part_dict = {
            'static_information_emb': section_embs['static_information'],
            'interest_profile_emb': section_embs['interest_profile'],
            'consumption_profile_emb': section_embs['consumption_profile'],
            'demand_targets_emb': section_embs['demand_targets'],
        }
        fused = self.fusion.forward_from_dict(part_dict)
        return fused.cpu()

    def load_fusion_state(self, path: str):
        """加载预训练的融合模块权重"""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.fusion.load_state_dict(state)
        logger.info(f"加载 fusion 权重: {path}")

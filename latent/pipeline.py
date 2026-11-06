"""
pipeline.py — 潜在线索提取完整管线
====================================
封装 XML 结构化 → 分部分 Embedding → 注意力融合的端到端流程。

支持两种输入格式:
  1. JSONL (原始格式): 每行一个用户，含多时间步 records
  2. Parquet 目录 (新格式): 目录下含多个 day_xx.parquet，
     每个文件代表某天的用户 XML 快照（uid + xml 两列），
     xml tag 支持完整格式(static_information/…)和简化格式(static_info/…)
"""

import os
import torch
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
    - encode_user():         单用户编码，返回时间序列 embedding
    - preprocess_jsonl():    批量预处理 JSONL → user_*.pt（原有接口）
    - preprocess_parquet():  批量预处理 day_xx.parquet 目录 → shard_*.pt（新接口）
    - encode_text_sections(): 对已分段文本编码（OPD 蒸馏时使用）
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
            part_dict = {k: v.unsqueeze(0).float().to(self.device)
                         for k, v in enc.items() if k.endswith('_emb')}
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
        """批量预处理 JSONL → user_*.pt 文件"""
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
    # 新接口：day_xx.parquet 目录 → shard .pt
    # ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def preprocess_parquet(
        self,
        data_dir: str,
        output_dir: str,
        start_time: str,
        uid_col: str = 'uid',
        xml_col: str = 'xml',
        max_users: Optional[int] = None,
        shard_size: int = 50000,
        batch_size: int = 64,
    ):
        """
        从 day_xx.parquet 目录批量提取 latent embeddings，输出 shard .pt 文件。

        目录结构:
          data_dir/
            day_00.parquet   ← start_time 当天（最新）
            day_07.parquet   ← start_time - 7 天
            day_14.parquet   ← start_time - 14 天
            ...
            day_28.parquet   ← start_time - 28 天（最旧）

        每个 parquet 含两列:
          - uid_col (默认 'uid'):  用户 ID
          - xml_col (默认 'xml'):  XML 用户画像快照
            tag 支持: <static_info>/<static_information>,
                      <interest>/<interest_profile>,
                      <consumption>/<consumption_profile>,
                      <demand>/<demand_targets>

        时间标签:
          day_N 文件 → 实际日期 = start_time - N 天，格式 'YYYYMMDD'
          时间序列按实际日期升序排列（最旧在前，最新在后），与 FM 训练约定一致。
          生成的日期字符串可被 data_loader.parse_timestamp 正确解析，
          从而让 compute_delta_t 准确计算真实时间跨度。

        输出:
          shard_XXXXXX.pt，格式:
          {
            "embeddings": Tensor(N, T, dim),   # N 用户 × T 时间步 × 维度
            "times":      ["YYYYMMDD", ...],   # 长度 T，取自首位用户的时间标签
            "user_ids":   [str, ...],          # 长度 N
          }
          同一 shard 内所有用户 T 相同；T 不同的用户分别写入独立 shard 系列。

        Args:
            data_dir:    含 day_*.parquet 文件的目录
            output_dir:  输出 shard 目录
            start_time:  day_00 对应的日期，格式 'YYYYMMDD' / 'YYYY-MM-DD' / 'YYYY/MM/DD'
            uid_col:     uid 列名（默认 'uid'）
            xml_col:     xml 列名（默认 'xml'）
            max_users:   最多处理的用户数（调试用，None=全量）
            shard_size:  每个 shard 文件最大用户数（默认 50000）
            batch_size:  Embedding 模型推理 batch size（默认 64）
        """
        import re
        import pandas as pd
        from datetime import datetime, timedelta

        os.makedirs(output_dir, exist_ok=True)

        # ── 解析 start_time ──
        base_date = None
        for fmt in ('%Y%m%d', '%Y-%m-%d', '%Y/%m/%d'):
            try:
                base_date = datetime.strptime(start_time, fmt)
                break
            except ValueError:
                continue
        if base_date is None:
            raise ValueError(
                f"无法解析 start_time='{start_time}'，"
                f"支持格式: YYYYMMDD / YYYY-MM-DD / YYYY/MM/DD"
            )
        logger.info(f"start_time={start_time} → base_date={base_date.strftime('%Y-%m-%d')}")

        # ── 扫描 day_*.parquet 文件 ──
        day_files: dict[int, str] = {}
        for fname in sorted(os.listdir(data_dir)):
            m = re.match(r'day_(\d+)\.parquet$', fname)
            if m:
                day_files[int(m.group(1))] = os.path.join(data_dir, fname)

        if not day_files:
            raise FileNotFoundError(
                f"在 {data_dir} 下未找到 day_*.parquet 文件"
            )
        day_min, day_max = min(day_files), max(day_files)
        logger.info(
            f"发现 {len(day_files)} 个 day 文件: day_{day_min} ~ day_{day_max}"
        )

        # ── 横向加载所有 day 文件，统一拼接为一个 DataFrame ──
        # 每行携带: _uid, _xml, _date(YYYYMMDD), _day_num
        # _day_num 越大 = 越旧；后面按 _day_num 降序排序 = 旧→新
        frames = []
        for day_num in sorted(day_files):
            actual_date = (base_date - timedelta(days=day_num)).strftime('%Y%m%d')
            df_day = pd.read_parquet(day_files[day_num], columns=[uid_col, xml_col])
            df_day = df_day.rename(columns={uid_col: '_uid', xml_col: '_xml'})
            df_day['_date']    = actual_date
            df_day['_day_num'] = day_num
            frames.append(df_day)
            logger.info(f"  day_{day_num:02d} → {actual_date}  ({len(df_day):,} 行)")

        all_df = pd.concat(frames, ignore_index=True)
        logger.info(f"合并后共 {len(all_df):,} 行")

        # ── 按 uid 分组 ──
        grouped = all_df.groupby('_uid', sort=False)
        all_uids = list(grouped.groups.keys())
        if max_users:
            all_uids = all_uids[:max_users]
        logger.info(f"共 {len(all_uids):,} 个唯一用户 → {output_dir}")

        # ── shard 缓冲（按 T 分桶，同 shard 内 T 一致）──
        t_buffers: dict = {}
        total_shards = 0

        def _flush_t_buffer(T: int):
            nonlocal total_shards
            buf = t_buffers[T]
            if not buf["embs"]:
                return
            shard_path = os.path.join(output_dir, f"shard_{total_shards:06d}.pt")
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
                f"用户={len(buf['uids']):,} → {shard_path}"
            )
            total_shards += 1
            t_buffers[T] = {"embs": [], "uids": [], "times": buf["times"]}

        # ── 逐用户处理 ──
        for uid in tqdm(all_uids, desc="提取 latent", unit="用户"):
            # 按 _day_num 降序排序：最大 day_num（最旧）排在前，day_00（最新）在后
            user_df = grouped.get_group(uid).sort_values('_day_num', ascending=False)

            xml_list  = user_df['_xml'].tolist()
            date_list = user_df['_date'].tolist()

            if not xml_list:
                continue

            # 解析 XML → sections
            sections = []
            for xml_str, date_str in zip(xml_list, date_list):
                parsed = parse_profile_xml(str(xml_str))
                sections.append({
                    'time':                date_str,
                    'static_information':  parsed['static_information'],
                    'interest_profile':    parsed['interest_profile'],
                    'consumption_profile': parsed['consumption_profile'],
                    'demand_targets':      parsed['demand_targets'],
                })

            # 编码（批量）
            try:
                encoded = self.embedder.encode_sections(
                    sections, batch_size=batch_size, show_progress=False
                )
            except Exception as e:
                logger.warning(f"  uid={uid} 编码失败，跳过: {e}")
                continue

            # 融合
            fused_list = []
            ok = True
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
                    ok = False
                    break
                fused_list.append(fused)

            if not ok or not fused_list:
                continue

            user_emb       = torch.stack(fused_list, dim=0)   # (T, dim)
            T_user         = user_emb.size(0)
            times_for_user = [enc['time'] for enc in encoded]

            # 加入对应 T 的缓冲
            if T_user not in t_buffers:
                t_buffers[T_user] = {
                    "embs":  [],
                    "uids":  [],
                    "times": times_for_user,   # 该 T 下首位用户的时间标签
                }
            t_buffers[T_user]["embs"].append(user_emb)
            t_buffers[T_user]["uids"].append(str(uid))

            if len(t_buffers[T_user]["embs"]) >= shard_size:
                _flush_t_buffer(T_user)

        # 末尾 flush
        for T_val in list(t_buffers.keys()):
            _flush_t_buffer(T_val)

        logger.info(
            f"preprocess_parquet 完成: 共 {total_shards} 个 shard 文件  "
            f"[start_time={start_time}, day_{day_min}~day_{day_max}, "
            f"用户数={len(all_uids):,}]"
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
        每项 dict 包含 static_information, interest_profile,
        consumption_profile, demand_targets 文本。

        Returns:
            Tensor(batch, output_dim) 融合 embedding
        """
        section_texts = {
            'static_information': [],
            'interest_profile': [],
            'consumption_profile': [],
            'demand_targets': [],
        }
        for sec in sections:
            for name in section_texts:
                section_texts[name].append(sec.get(name, ''))

        section_embs = {}
        for name in section_texts:
            embs = self.embedder.encode_texts(
                section_texts[name], show_progress=False, desc=f"[Emb] {name}"
            )
            section_embs[name] = embs.float().to(self.device)

        part_dict = {
            'static_information_emb':  section_embs['static_information'],
            'interest_profile_emb':    section_embs['interest_profile'],
            'consumption_profile_emb': section_embs['consumption_profile'],
            'demand_targets_emb':      section_embs['demand_targets'],
        }
        fused = self.fusion.forward_from_dict(part_dict)
        return fused.cpu()

    def load_fusion_state(self, path: str):
        """加载预训练的融合模块权重"""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.fusion.load_state_dict(state)
        logger.info(f"加载 fusion 权重: {path}")

"""
pipeline.py — 潜在线索提取完整管线
====================================
封装 XML 结构化 → 分部分 Embedding → 注意力融合的端到端流程。
"""

import os
import torch
from tqdm import tqdm
from typing import Optional

from common.logger import get_logger
from common.config import LatentConfig
from .xml_structured import load_jsonl, user_to_xml_sections
from .embedder import ProfileEmbedder, create_embedder
from .attention_fusion import AttentionFusion

logger = get_logger(__name__)


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
            part_dict = {k: v.unsqueeze(0).to(self.device) for k, v in enc.items() if k.endswith('_emb')}
            fused = self.fusion.forward_from_dict(part_dict).squeeze(0).cpu()
            fused_embs.append(fused)
            section_embs.append({k: v.cpu() for k, v in part_dict.items()})
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
            section_embs[name] = embs.to(self.device)

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
        state = torch.load(path, map_location=self.device)
        self.fusion.load_state_dict(state)
        logger.info(f"加载 fusion 权重: {path}")

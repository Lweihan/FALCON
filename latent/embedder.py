"""
embedder.py — 分部分 Embedding 编码器
=======================================
使用 qwen3-embedding 模型对用户画像 XML 的四个部分分别编码。
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from typing import Optional

from common.logger import get_logger

logger = get_logger(__name__)


class ProfileEmbedder:
    """
    用户画像分部分 Embedding 编码器。

    取 last hidden state 最后一个 token → L2 归一化 → 截取前 embed_dim 维。
    """

    SECTION_NAMES = ['static_information', 'interest_profile', 'consumption_profile', 'demand_targets']

    def __init__(
        self,
        model_path: str = 'Qwen/Qwen3-Embedding-0.6B',
        device: str = 'cuda:0',
        max_length: int = 512,
        embed_dim: int = 256,
    ):
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.device = device

        logger.info(f"加载 Embedding 模型: {model_path} → {device}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
            self.model = AutoModel.from_pretrained(
                model_path, torch_dtype=torch.float16, attn_implementation="flash_attention_2"
            ).to(device)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
            self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
            logger.warning("Flash Attention 2 不可用，使用默认 attention")
        self.model.eval()

    @torch.no_grad()
    def encode_texts(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = False,
        desc: str = "",
    ) -> torch.Tensor:
        """
        对一批文本编码，空文本对应全零向量。

        Returns:
            Tensor (len(texts), embed_dim)
        """
        n = len(texts)
        non_empty_idx = [i for i, t in enumerate(texts) if t.strip()]
        non_empty_txt = [texts[i] for i in non_empty_idx]

        all_embs = torch.zeros(n, self.embed_dim, dtype=torch.float16)

        if non_empty_txt:
            pbar = tqdm(total=len(non_empty_txt), desc=desc, disable=not show_progress, unit="条")
            for i in range(0, len(non_empty_txt), batch_size):
                batch = non_empty_txt[i:i + batch_size]
                inputs = self.tokenizer(batch, truncation=True, max_length=self.max_length,
                                        padding=True, return_tensors="pt").to(self.device)
                out = self.model(**inputs)
                last = F.normalize(out.last_hidden_state[:, -1], p=2, dim=1)
                embs = last[:, :self.embed_dim].cpu()
                idx_slice = non_empty_idx[i:i + batch_size]
                all_embs[idx_slice[:len(embs)]] = embs[:len(idx_slice)]
                pbar.update(len(batch))
            pbar.close()

        return all_embs

    def encode_sections(
        self,
        sections_list: list[dict],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> list[dict]:
        """
        对一个用户的多个时间步 sections 分部分批量编码。

        Returns:
            每项含 *_emb 四个 Tensor(embed_dim) + time
        """
        n = len(sections_list)
        section_texts = {name: [] for name in self.SECTION_NAMES}
        for sec in sections_list:
            for name in self.SECTION_NAMES:
                section_texts[name].append(sec.get(name, ''))

        section_embs = {}
        for name in self.SECTION_NAMES:
            section_embs[name] = self.encode_texts(
                section_texts[name], batch_size=batch_size,
                show_progress=show_progress, desc=f"[Emb] {name}"
            )

        results = []
        for i, sec in enumerate(sections_list):
            results.append({
                'time': sec['time'],
                'static_information_emb': section_embs['static_information'][i],
                'interest_profile_emb': section_embs['interest_profile'][i],
                'consumption_profile_emb': section_embs['consumption_profile'][i],
                'demand_targets_emb': section_embs['demand_targets'][i],
            })
        return results


def create_embedder(
    model_path: str = 'Qwen/Qwen3-Embedding-0.6B',
    device: Optional[str] = None,
    max_length: int = 512,
    embed_dim: int = 256,
) -> ProfileEmbedder:
    """工厂函数"""
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return ProfileEmbedder(model_path=model_path, device=device,
                           max_length=max_length, embed_dim=embed_dim)

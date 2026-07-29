"""
base_ranker.py — 精排模型基类
================================
所有下游精排模型继承此类，统一接口。

核心设计:
- 稀疏特征 Embedding 层 (支持分组 embed_dim, 大词表用小维度省参数)
- extract_embedding(): 提取中间表示，供 FALCON FM 优化
- 分组参数统计: Embedding 层 vs 网络层
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class BaseRanker(nn.Module):
    """
    精排模型基类。

    子类必须实现:
    - forward(sparse_values) -> logits
    - extract_embedding(sparse_values) -> embedding
    """

    def __init__(
        self,
        sparse_feature_names: List[str],
        feature_voc_sizes: Dict[str, int],
        embed_dim: int = 16,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        feature_embed_dims: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.sparse_feature_names = sparse_feature_names
        self.feature_voc_sizes = feature_voc_sizes
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.dropout_rate = dropout
        self.activation_name = activation

        # 分组 embed_dim: 未指定的特征统一用 embed_dim
        self.feature_embed_dims = feature_embed_dims or {}

        self.embeddings = nn.ModuleDict({
            feat: nn.Embedding(
                num_embeddings=feature_voc_sizes[feat],
                embedding_dim=self.feature_embed_dims.get(feat, embed_dim),
                padding_idx=0,
            )
            for feat in sparse_feature_names
        })

        self.concat_embed_dim = sum(
            self.feature_embed_dims.get(feat, embed_dim)
            for feat in sparse_feature_names
        )

        self.activation = self._get_activation(activation)

    # ── 工具方法 ──

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        table = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
        }
        if name not in table:
            raise ValueError(f"Unknown activation: {name}, choose from {list(table)}")
        return table[name]

    def count_parameters(self, only_trainable: bool = True) -> Dict[str, int]:
        """分组统计参数量: Embedding 层 vs 网络层"""
        def _count(params):
            return sum(p.numel() for p in params if (not only_trainable or p.requires_grad))

        emb = _count(self.embeddings.parameters())
        total = _count(self.parameters())
        return {"total": total, "embedding": emb, "network": total - emb}

    def build_mlp(self, in_dim: int, out_dim: int = 1) -> nn.Sequential:
        """按 hidden_dims 构建标准 MLP"""
        layers = []
        d = in_dim
        for h in self.hidden_dims:
            layers += [nn.Linear(d, h), self._get_activation(self.activation_name),
                       nn.Dropout(self.dropout_rate)]
            d = h
        layers.append(nn.Linear(d, out_dim))
        return nn.Sequential(*layers)

    # ── Embedding 处理 ──

    def embed_input(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """(B,) × N → (B, concat_embed_dim) 拼接"""
        return torch.cat(
            [self.embeddings[f](sparse_values[f]) for f in self.sparse_feature_names],
            dim=1,
        )

    def embed_input_stacked(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        (B,) × N → (B, N, embed_dim) 堆叠为序列。

        要求所有特征 embed_dim 相同 (未使用分组维度)，
        供 AutoInt / TokenMixer 等序列建模模型使用。
        """
        return torch.stack(
            [self.embeddings[f](sparse_values[f]) for f in self.sparse_feature_names],
            dim=1,
        )

    # ── 接口 ──

    def forward(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """返回 (batch,) logits (未过 sigmoid)"""
        raise NotImplementedError

    def predict(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """返回 (batch,) 概率"""
        return torch.sigmoid(self.forward(sparse_values))

    def extract_embedding(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        提取中间层 embedding，供 FALCON FM 优化。

        Returns:
            (batch, embed_output_dim)
        """
        raise NotImplementedError("子类必须实现 extract_embedding")

    @property
    def embed_output_dim(self) -> int:
        """extract_embedding 输出维度"""
        return self.hidden_dims[-1]

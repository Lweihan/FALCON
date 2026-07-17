"""
latent — 潜在线索提取模块 (Latent Clue)
===========================================
从大模型输出的用户小传中提取结构化潜在线索:
  XML 结构化 → 分部分 Embedding → 注意力融合 → 融合 Embedding
"""

__all__ = [
    'record_to_xml', 'parse_profile_xml', 'user_to_xml_sections', 'load_jsonl',
    'ProfileEmbedder', 'create_embedder',
    'AttentionFusion',
    'LatentPipeline',
]


def __getattr__(name):
    """Lazy loading: only import heavy modules when accessed."""
    if name in ('record_to_xml', 'parse_profile_xml', 'user_to_xml_sections', 'load_jsonl'):
        from .xml_structured import record_to_xml, parse_profile_xml, user_to_xml_sections, load_jsonl
        return locals()[name]
    elif name in ('ProfileEmbedder', 'create_embedder'):
        from .embedder import ProfileEmbedder, create_embedder
        return locals()[name]
    elif name == 'AttentionFusion':
        from .attention_fusion import AttentionFusion
        return locals()[name]
    elif name == 'LatentPipeline':
        from .pipeline import LatentPipeline
        return locals()[name]
    raise AttributeError(f"module 'latent' has no attribute {name!r}")

"""
scenario — 广告场景应用模块
=============================
"""

__all__ = ['UserEmbeddingDataset', 'create_dataloader', 'collate_fn',
           'FALCONEvaluator', 'FALCONServing']


def __getattr__(name):
    if name in ('UserEmbeddingDataset', 'create_dataloader', 'collate_fn'):
        from .data_loader import UserEmbeddingDataset, create_dataloader, collate_fn
        return locals()[name]
    elif name == 'FALCONEvaluator':
        from .evaluator import FALCONEvaluator
        return FALCONEvaluator
    elif name == 'FALCONServing':
        from .serving import FALCONServing
        return FALCONServing
    raise AttributeError(f"module 'scenario' has no attribute {name!r}")

"""
ranker — 下游精排模型模块
===========================
用于验证 FALCON 产出的 embedding 在真实广告排序任务上的效果。

包含:
- 5 种精排模型: DNN / DeepFM / DCN / AutoInt / TokenMixer
- JointFALCONRanker: FALCON FM + Ranker 端到端联合训练
- 评估指标: AUC / GAUC / LogLoss / NDCG@K
"""

__all__ = [
    'BaseRanker',
    'DNN', 'DeepFM', 'DCN', 'AutoInt', 'TokenMixer',
    'RANKER_REGISTRY', 'create_ranker',
    'JointFALCONRanker',
    'RankerTrainer',
    'SparseRankerDataset', 'SyntheticRankerDataset', 'create_ranker_dataloaders',
    'evaluate_predictions', 'compute_lift',
]


def __getattr__(name):
    if name == 'BaseRanker':
        from .base_ranker import BaseRanker
        return BaseRanker
    if name in ('DNN', 'DeepFM', 'DCN', 'AutoInt', 'TokenMixer',
                'RANKER_REGISTRY', 'create_ranker'):
        from . import models
        return getattr(models, name)
    if name == 'JointFALCONRanker':
        from .joint_falcon_ranker import JointFALCONRanker
        return JointFALCONRanker
    if name == 'RankerTrainer':
        from .trainer import RankerTrainer
        return RankerTrainer
    if name in ('SparseRankerDataset', 'SyntheticRankerDataset', 'create_ranker_dataloaders'):
        from . import data
        return getattr(data, name)
    if name in ('evaluate_predictions', 'compute_lift'):
        from . import metrics
        return getattr(metrics, name)
    raise AttributeError(f"module 'ranker' has no attribute {name!r}")

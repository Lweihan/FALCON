"""
fm — Flow Matching 模块 (Teacher)
===================================
基于 Flow Matching 的用户特征预测模型，作为 OPD 的 Teacher。

Backbone 选项:
- dit (推荐):   DiTVelocityNetwork — Patchify + AdaLN-Zero + QK-Norm + RoPE + SwiGLU
- transformer: TransformerVelocityNetwork — 旧版保留兼容
- mlp:         VelocityNetwork — 轻量
"""

__all__ = [
    'flow_match_forward',
    'VelocityNetwork', 'TransformerVelocityNetwork', 'DiTVelocityNetwork',
    'DIT_PRESETS', 'resolve_dit_preset',
    'FlowMatchingModel', 'FMTrainer',
]


def __getattr__(name):
    if name == 'flow_match_forward':
        from .flow_core import flow_match_forward
        return flow_match_forward
    elif name == 'VelocityNetwork':
        from .velocity_net import VelocityNetwork
        return VelocityNetwork
    elif name == 'TransformerVelocityNetwork':
        from .velocity_net import TransformerVelocityNetwork
        return TransformerVelocityNetwork
    elif name == 'DiTVelocityNetwork':
        from .dit_backbone import DiTVelocityNetwork
        return DiTVelocityNetwork
    elif name == 'DIT_PRESETS':
        from .dit_backbone import DIT_PRESETS
        return DIT_PRESETS
    elif name == 'resolve_dit_preset':
        from .dit_backbone import resolve_dit_preset
        return resolve_dit_preset
    elif name == 'FlowMatchingModel':
        from .model import FlowMatchingModel
        return FlowMatchingModel
    elif name == 'FMTrainer':
        from .trainer import FMTrainer
        return FMTrainer
    raise AttributeError(f"module 'fm' has no attribute {name!r}")

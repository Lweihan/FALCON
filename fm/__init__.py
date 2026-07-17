"""
fm — Flow Matching 模块 (Teacher)
===================================
基于 Flow Matching 的用户特征预测模型，作为 OPD 的 Teacher。
"""

__all__ = ['flow_match_forward', 'VelocityNetwork', 'TransformerVelocityNetwork',
           'FlowMatchingModel', 'FMTrainer']


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
    elif name == 'FlowMatchingModel':
        from .model import FlowMatchingModel
        return FlowMatchingModel
    elif name == 'FMTrainer':
        from .trainer import FMTrainer
        return FMTrainer
    raise AttributeError(f"module 'fm' has no attribute {name!r}")

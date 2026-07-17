"""
opd — Online Policy Distillation 模块 (Student = LLM)
=======================================================
"""

__all__ = ['FMTeacher', 'LLMStudent', 'OPDLoss', 'DistillationEngine',
           'DistillationReward', 'ReplayBuffer', 'OPDTrainer']


def __getattr__(name):
    if name == 'FMTeacher':
        from .teacher import FMTeacher
        return FMTeacher
    elif name == 'LLMStudent':
        from .student import LLMStudent
        return LLMStudent
    elif name in ('OPDLoss', 'DistillationEngine'):
        from .distillation import OPDLoss, DistillationEngine
        return locals()[name]
    elif name in ('DistillationReward', 'ReplayBuffer'):
        from .reward import DistillationReward, ReplayBuffer
        return locals()[name]
    elif name == 'OPDTrainer':
        from .trainer import OPDTrainer
        return OPDTrainer
    raise AttributeError(f"module 'opd' has no attribute {name!r}")

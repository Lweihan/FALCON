"""
common — FALCON 通用基础设施
================================
"""

__all__ = ['FALCONConfig', 'FMConfig', 'OPDConfig', 'LatentConfig', 'ScenarioConfig',
           'get_logger', 'set_seed', 'count_parameters', 'save_checkpoint', 'load_checkpoint']


def __getattr__(name):
    if name in ('FALCONConfig', 'FMConfig', 'OPDConfig', 'LatentConfig', 'ScenarioConfig'):
        from .config import FALCONConfig, FMConfig, OPDConfig, LatentConfig, ScenarioConfig
        return locals()[name]
    elif name == 'get_logger':
        from .logger import get_logger
        return get_logger
    elif name in ('set_seed', 'count_parameters', 'save_checkpoint', 'load_checkpoint'):
        from .utils import set_seed, count_parameters, save_checkpoint, load_checkpoint
        return locals()[name]
    raise AttributeError(f"module 'common' has no attribute {name!r}")

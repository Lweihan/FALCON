"""
distributed.py — 分布式训练工具
=================================
DDP (DistributedDataParallel) 辅助函数。

启动方式:
    torchrun --nproc_per_node=4 scripts/pretrain.py --stage fm ...

torchrun 会自动注入环境变量:
    RANK        全局进程序号 [0, WORLD_SIZE)
    WORLD_SIZE  总进程数 (= 总 GPU 数)
    LOCAL_RANK  本机内进程序号 (= 本机 GPU 序号)
    MASTER_ADDR / MASTER_PORT

所有函数在非分布式环境下安全降级 (world_size=1, rank=0)，
因此单卡直接 `python scripts/pretrain.py` 也能正常运行。
"""

import os
from typing import Optional

import torch
import torch.distributed as dist


# ────────────────────────────────────────────────────────────
# 环境检测
# ────────────────────────────────────────────────────────────

def is_distributed_env() -> bool:
    """是否处于 torchrun 启动的分布式环境"""
    return "WORLD_SIZE" in os.environ and int(os.environ.get("WORLD_SIZE", 1)) > 1


def is_initialized() -> bool:
    """DDP 进程组是否已初始化"""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """全局进程序号"""
    if is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def get_world_size() -> int:
    """总进程数"""
    if is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def get_local_rank() -> int:
    """本机内进程序号 (对应 GPU 序号)"""
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    """是否为主进程 (rank 0)，只有主进程写日志/checkpoint/TensorBoard"""
    return get_rank() == 0


# ────────────────────────────────────────────────────────────
# 初始化 / 清理
# ────────────────────────────────────────────────────────────

def setup_distributed(backend: str = "nccl") -> tuple:
    """
    初始化 DDP 进程组。

    Returns:
        (is_dist, rank, world_size, local_rank, device)
    """
    if not is_distributed_env():
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        return False, 0, 1, 0, device

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # CPU 环境退化为 gloo
    if not torch.cuda.is_available():
        backend = "gloo"
        device = "cpu"
    else:
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"

    if not is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    return True, rank, world_size, local_rank, device


def cleanup_distributed():
    """销毁进程组"""
    if is_initialized():
        dist.destroy_process_group()


def barrier():
    """同步所有进程"""
    if is_initialized():
        dist.barrier()


# ────────────────────────────────────────────────────────────
# 跨卡聚合
# ────────────────────────────────────────────────────────────

def reduce_value(value: float, average: bool = True, device: Optional[str] = None) -> float:
    """
    跨卡聚合标量 (用于 loss / metrics 统计)。

    Args:
        value: 本卡数值
        average: True 求均值, False 求和
        device: 张量所在设备

    Returns:
        聚合后数值 (所有卡得到相同结果)
    """
    if not is_initialized():
        return value

    device = device or (f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu")
    tensor = torch.tensor(value, dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if average:
        tensor /= get_world_size()
    return tensor.item()


def reduce_dict(d: dict, average: bool = True, device: Optional[str] = None) -> dict:
    """批量聚合 dict 中的所有标量"""
    if not is_initialized():
        return d
    return {k: reduce_value(v, average, device) if isinstance(v, (int, float)) else v
            for k, v in d.items()}


def gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    收集所有卡的张量并沿 dim 0 拼接 (用于评估时汇总预测结果)。

    要求各卡张量 shape 一致。
    """
    if not is_initialized():
        return tensor

    world_size = get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


# ────────────────────────────────────────────────────────────
# 模型工具
# ────────────────────────────────────────────────────────────

def unwrap_model(model):
    """剥离 DDP / compile 包装，取出原始模型 (保存 checkpoint 时使用)"""
    m = model
    # DDP 包装
    if hasattr(m, "module"):
        m = m.module
    # torch.compile 包装
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


def setup_performance_flags(use_tf32: bool = True, cudnn_benchmark: bool = True):
    """
    开启计算性能优化开关。

    - TF32: Ampere+ (A100/H100/RTX30+) 上矩阵乘使用 TF32 精度, 显著加速且精度损失极小
    - cudnn.benchmark: 输入尺寸固定时自动搜索最优 kernel
    """
    if not torch.cuda.is_available():
        return

    if use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # PyTorch 2.x 推荐写法
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass

    if cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

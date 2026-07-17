"""
config.py — 统一配置管理
=========================
使用 dataclass 管理所有子模块配置，支持从 YAML 文件加载。
"""

import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional


# ────────────────────────────────────────────────────────────
# Latent Clue 配置
# ────────────────────────────────────────────────────────────

@dataclass
class LatentConfig:
    """潜在线索提取配置 (XML 结构化 + Embedding + 融合)"""
    embed_model_path: str = "Qwen/Qwen3-Embedding-0.6B"
    embed_dim: int = 256          # 每部分 embedding 截取维度
    max_length: int = 512         # tokenizer 最大长度
    fusion_hidden_dim: int = 256  # 注意力融合隐藏维度
    fusion_output_dim: int = 256  # 融合输出维度
    fusion_num_heads: int = 4     # 注意力头数
    fusion_dropout: float = 0.1   # dropout


# ────────────────────────────────────────────────────────────
# Flow Matching 配置
# ────────────────────────────────────────────────────────────

@dataclass
class FMConfig:
    """Flow Matching (Teacher) 配置"""
    data_dim: int = 256           # 数据维度 (= LatentConfig.fusion_output_dim)
    time_dim: int = 128          # 时间编码维度
    cond_dim: int = 256          # 条件维度 (= data_dim)
    hidden_dim: int = 512        # Velocity Net 隐藏维度
    num_layers: int = 6          # Velocity Net 层数
    dropout: float = 0.1
    # 训练
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    save_every: int = 10
    # 采样
    sample_steps: int = 100
    sample_method: str = "euler"  # euler / rk4


# ────────────────────────────────────────────────────────────
# Online Policy Distillation 配置
# ────────────────────────────────────────────────────────────

@dataclass
class OPDConfig:
    """Online Policy Distillation 配置 (Student = LLM)"""
    # Student LLM 配置
    student_model_path: str = "Qwen/Qwen3-8B"
    lora_r: int = 16             # LoRA rank
    lora_alpha: int = 32         # LoRA alpha
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    max_new_tokens: int = 2048   # 生成长度
    temperature: float = 0.7
    top_p: float = 0.9
    # Distillation 配置
    distill_alpha: float = 1.0    # embedding 蒸馏损失权重
    distill_beta: float = 0.5     # per-section 蒸馏损失权重
    distill_gamma: float = 0.1   # KL 散度损失权重
    distill_tau: float = 2.0     # 温度缩放参数
    # Online 配置
    online_update_freq: int = 100 # 每 N 条新数据做一次 online update
    replay_buffer_size: int = 10000
    ema_decay: float = 0.999     # Teacher EMA 衰减
    # 训练
    epochs: int = 5
    batch_size: int = 4
    lr: float = 5e-5
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    warmup_steps: int = 100
    save_every: int = 1


# ────────────────────────────────────────────────────────────
# Scenario 配置
# ────────────────────────────────────────────────────────────

@dataclass
class ScenarioConfig:
    """广告场景应用配置"""
    data_path: str = ""
    max_users: int = -1
    train_ratio: float = 0.8
    num_workers: int = 0
    eval_metrics: list = field(default_factory=lambda: ["cosine_sim", "mse", "ndcg"])


# ────────────────────────────────────────────────────────────
# 全局配置
# ────────────────────────────────────────────────────────────

@dataclass
class FALCONConfig:
    """FALCON 全局配置"""
    seed: int = 42
    output_dir: str = "./output"
    device: str = "auto"          # auto / cuda:0 / cpu
    latent: LatentConfig = field(default_factory=LatentConfig)
    fm: FMConfig = field(default_factory=FMConfig)
    opd: OPDConfig = field(default_factory=OPDConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)

    def resolve_device(self) -> str:
        if self.device == "auto":
            try:
                import torch
                return "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device

    def save_yaml(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, path: str) -> "FALCONConfig":
        with open(path, 'r') as f:
            d = yaml.safe_load(f)
        if d is None:
            d = {}
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict) -> "FALCONConfig":
        """从嵌套字典构建，自动解析子配置"""
        latent = LatentConfig(**d.get("latent", {}))
        fm = FMConfig(**d.get("fm", {}))
        opd = OPDConfig(**d.get("opd", {}))
        scenario = ScenarioConfig(**d.get("scenario", {}))
        return cls(
            seed=d.get("seed", 42),
            output_dir=d.get("output_dir", "./output"),
            device=d.get("device", "auto"),
            latent=latent,
            fm=fm,
            opd=opd,
            scenario=scenario,
        )

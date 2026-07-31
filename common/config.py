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
    # 时间编码维度: 必须与 checkpoint 保存时一致, 否则 time_embed.1 权重形状不匹配
    # 若加载的 checkpoint 维度不同, JointPostTrainer.load_fm(auto_align_dims=True)
    # 会自动按 checkpoint 重建模型
    time_dim: int = 128
    cond_dim: int = 256          # 条件维度 (= data_dim)
    # Backbone 选择: dit (推荐) / transformer / mlp
    backbone_type: str = "dit"
    # ── DiT backbone ──
    # preset: tiny(~10M) / small(~80M) / base(~250M) / large(~1.2B) / xl(~4B)
    # 设为 "" 或 "custom" 时使用下面手动指定的参数
    dit_preset: str = "base"
    dit_dim: int = 1024
    dit_depth: int = 16
    dit_heads: int = 16
    dit_num_patches: int = 16
    dit_mlp_ratio: float = 4.0
    dit_dropout: float = 0.0
    dit_qk_norm: bool = True
    dit_use_rope: bool = True
    use_gradient_checkpointing: bool = False
    use_amp: bool = False          # 混合精度训练
    # ── MLP backbone ──
    hidden_dim: int = 512
    num_layers: int = 6
    dropout: float = 0.1
    # ── Transformer backbone (旧版, 保留兼容) ──
    transformer_dim: int = 1024
    transformer_depth: int = 8
    transformer_heads: int = 8
    transformer_mlp_ratio: float = 4.0
    transformer_dropout: float = 0.1
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
    # 时间加权
    time_weight_mode: str = "mid"   # none / mid / late / early
    time_weight_scale: float = 1.0  # 加权幅度缩放
    # EMA loss
    ema_loss_alpha: float = 0.1    # loss EMA 平滑系数
    # ── 配对策略 ──
    # last_target: 随机早期步 → 最后一步 (默认, 数据利用率 1/(T-1))
    # all_pairs:   枚举所有 (i,j) i<j     (利用率 100%)
    # adjacent:    相邻配对 (i, i+1)
    pair_mode: str = "last_target"
    use_delta_t_condition: bool = True   # 将真实时间跨度注入网络 (修训推不一致)
    delta_t_from_timestamp: bool = True  # 优先用时间戳算 delta_t, 失败回退步数比例
    # ── 数据加载 ──
    num_workers: int = 4
    lazy_load: bool = True          # 懒加载 .pt, 配合 num_workers 并行 IO
    prefetch_factor: int = 2
    drop_last: bool = False
    # ── 验证优化 (GPU 利用率关键) ──
    val_every: int = 1              # 每 N epoch 验证一次
    val_ode_steps: int = 10         # 验证 ODE 步数 (越小越快)
    val_max_batches: int = -1       # -1 全量; 正数则只跑前 N 个 batch
    val_compute_cos_sim: bool = True  # false 则完全跳过 ODE, 验证极快
    # ── 计算性能 ──
    use_tf32: bool = True           # Ampere+ 矩阵乘加速
    cudnn_benchmark: bool = True    # 固定尺寸下自动选最优 kernel
    use_compile: bool = False       # torch.compile (首次编译慢, 之后快 20-30%)
    grad_accum_steps: int = 1       # 梯度累积, 等效放大 batch
    # ── 多卡 ──
    distributed: bool = False       # 由 torchrun 环境变量自动检测覆盖

    def resolve_dit_params(self) -> dict:
        """
        解析 DiT 超参数。若指定了有效 preset 则用 preset，否则用手动参数。

        Returns:
            dict with dit_dim, dit_depth, dit_heads, dit_num_patches
        """
        if self.dit_preset and self.dit_preset not in ("", "custom"):
            from fm.dit_backbone import resolve_dit_preset
            return resolve_dit_preset(self.dit_preset)
        return {
            "dit_dim": self.dit_dim,
            "dit_depth": self.dit_depth,
            "dit_heads": self.dit_heads,
            "dit_num_patches": self.dit_num_patches,
        }


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
# Candidate-Filtered Online Policy Distillation (CF-OPD) 配置
# ────────────────────────────────────────────────────────────

@dataclass
class CFOPDConfig:
    """
    CF-OPD (Candidate-Filtered Online Policy Distillation) 配置。

    与原始 OPD 的核心区别:
      原始 OPD: Teacher → 单条 Student 生成 → 直接对齐
      CF-OPD:  Student 生成 N 条候选 → 分别 standardize → encode → ranker 打分
               → 筛选高分候选 (reward > threshold) → 对选中候选做 DPO / KL 学习

    流程:
      1. LLM 生成 N 条候选小传文本
      2. 每条 → standardize → XML sections → 分部分 encode → fused embedding
      3. fused embedding → FM Teacher predict_target → teacher_embedding
      4. Reward = cos_sim(student_emb, teacher_emb) + λ * ranker_cvr_score
      5. 筛选 reward > threshold 的候选 (正例), 低分的为负例
      6. DPO loss: log σ(β * (log π_θ(y_w|x) - log π_ref(y_w|x)
                            - log π_θ(y_l|x) + log π_ref(y_l|x)))
      或 KL loss: 对选中正例做 KL(π_θ_selected ‖ π_θ_ref) 引导 LLM 向高分方向
    """
    # ─── 候选采样 ───
    num_candidates: int = 4         # 每个用户生成 N 条候选
    candidate_temperature: float = 0.8  # 候选采样温度 (比训练温度稍高, 增加多样性)

    # ─── 筛选策略 ───
    filter_threshold: float = 0.5   # ranker CVR 分数阈值: > 0.5 为正例
    use_combined_reward: bool = True  # True = cos_sim + λ*cvr; False = 仅 cvr
    reward_lambda: float = 0.5     # 组合奖励中 ranker CVR 的权重
    emb_reward_weight: float = 0.5  # 组合奖励中 embedding cosine similarity 的权重
    min_positive_ratio: float = 0.1  # 至少 10% 候选被选为正例 (防止全被筛掉)
    max_positive_ratio: float = 0.8  # 最多 80% (防止全部为正例, DPO 无负例)

    # ─── 学习方式 ───
    learning_mode: str = "dpo"      # dpo / kl / combined
    # DPO 参数
    dpo_beta: float = 0.1           # DPO 隐式奖励缩放 (KL 约束强度)
    # KL 参数
    kl_tau: float = 2.0             # KL 温度缩放

    # ─── 继承 OPD 原始参数 ───
    distill_alpha: float = 1.0     # L_emb 权重 (可选: embedding 对齐损失)
    distill_beta: float = 0.5      # L_section 权重
    distill_gamma: float = 0.1     # L_kl 权重 (防止遗忘)

    # ─── 训练 ───
    epochs: int = 3
    batch_size: int = 2
    lr: float = 5e-5
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    warmup_steps: int = 50
    save_every: int = 1

    # ─── 下游 ranker ───
    ranker_weights_path: Optional[str] = None  # 已训练的 ranker 权重路径
    ranker_input_dim: int = 256
    ranker_hidden_dims: list = field(default_factory=lambda: [128, 64])


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
# 下游 Ranker 配置
# ────────────────────────────────────────────────────────────

@dataclass
class RankerConfig:
    """下游精排模型配置"""
    # 模型类型: dnn / deepfm / dcn / autoint / token_mixer
    ranker_types: list = field(default_factory=lambda: ["dnn", "deepfm", "dcn", "autoint", "token_mixer"])
    # 特征配置
    embed_dim: int = 16
    hidden_dims: list = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.1
    activation: str = "relu"
    # DCN 专属
    num_cross_layers: int = 3
    # AutoInt 专属
    num_attention_layers: int = 3
    num_attention_heads: int = 4
    attention_dim: int = 64
    # TokenMixer 专属
    num_blocks: int = 3
    num_groups: int = 4
    ff_mult: int = 4
    channel_ff_mult: int = 4
    # 联合训练
    joint_alpha: float = 0.5         # CVR loss 权重; (1-alpha)=FM flow loss 权重
    fm_weights: list = field(default_factory=lambda: [0.1, 0.5, 1.0])  # FM 损失权重消融
    freeze_fm: bool = False          # 是否冻结 FM backbone
    freeze_fm_backbone: bool = False # 只冻结 FM backbone, 仍训 condition proj
    fast_train: bool = True          # FM 快速训练模式 (跳过 ODE 求解)
    train_delta_t: float = 0.5
    train_ode_steps: int = 10
    eval_ode_steps: int = 50
    # 训练超参
    epochs: int = 3
    batch_size: int = 1024
    lr: float = 1e-3
    # weight_decay: 只对 2D 权重矩阵生效, bias/LayerNorm/Embedding 不衰减
    # (标准做法, 否则会退化嵌入向量)
    weight_decay: float = 0.01
    grad_clip: float = 5.0
    # Early stopping
    early_stop_patience: int = 3          # 连续 N 个 epoch 未改善即停; 0 = 禁用
    early_stop_min_delta: float = 1e-4    # 视为改善的最小指标提升
    # LR scheduler
    lr_scheduler: str = "cosine"          # none / cosine / linear
    warmup_ratio: float = 0.03            # 首 3% steps 做 linear warmup
    grad_accum: int = 4
    use_amp: bool = True
    save_every: int = 1
    # ── DDP 多卡 ──
    # distributed 由 torchrun 环境变量自动检测, 无需手动设置
    distributed: bool = False
    # find_unused_parameters: auto / true / false
    #   auto: freeze_fm 或 freeze_fm_backbone 开启时自动为 true
    #   (冻结部分参数时 DDP 会因"期待的梯度未到达"报错, 必须开启;
    #    但开启有性能损失, 故按需决定)
    ddp_find_unused_parameters: str = "auto"
    # 数据 (通用)
    data_path: str = ""
    max_samples: int = -1
    train_ratio: float = 0.8
    num_workers: int = 4
    # MAC 数据集专属
    # load_mode:
    #   local_dir     从本地 MAC 目录加载 (推荐; 需 train/ test/ vocabs/ 子目录)
    #   hf_streaming  直接从 HuggingFace 流式读 (无需下载)
    #   hf_download   先下载到本地缓存 (~80GB)
    #   local_parquet 单个 parquet 文件
    #   local_csv     单个 csv 文件
    mac_load_mode: str = "local_dir"
    mac_data_path: str = ""          # local_dir: MAC 根目录; local_*: 单文件路径
    mac_train_subdir: str = "train"   # 训练数据子目录 (local_dir 模式)
    mac_val_subdir: str = "test"     # 验证数据子目录 (MAC 数据集用 test/)
    mac_vocab_subdir: str = "vocabs"  # 词表子目录, 每字段一个 json (local_dir 模式)
    mac_parquet_batch_size: int = 10000   # parquet 流式读取的行组大小
    mac_label: str = "last"          # CVR 标签: linear / mta / first / last
    # ── 标签口径 ──
    # MAC 的 linear/mta/first/last 是多触点归因 (MTA) 权重, 为连续实数
    #   binary:     label = float(raw > mac_label_threshold), 标准 CVR 二分类
    #               → 评估用 AUC / GAUC / LogLoss / PosRate
    #   continuous: label = clip(raw, 0, 1), 保留归因权重强度
    #               → 评估用 MSE / MAE / Pearson / Spearman (+ 二值化参考 AUC)
    mac_label_mode: str = "binary"        # binary / continuous
    mac_label_threshold: float = 0.0      # binary 模式: > threshold 视为正例
    mac_continuous_loss: str = "bce"      # continuous 模式损失: bce (软标签) / mse
    mac_best_metric: str = "auto"         # best ckpt 判据: auto / auc / mse / logloss
    mac_vocab_path: str = ""         # 独立词表 json 路径 (非 local_dir 模式)
    mac_max_samples: int = -1        # -1 全量; 正数限制 (调试)
    mac_val_max_samples: int = 100_000  # val 最多读取样本数
    mac_val_ratio: float = 0.01      # 非 local_dir 模式: 前 N% 作 val
    mac_num_workers: int = 0         # streaming / IterableDataset 时必须 0
    seq_pool_mode: str = "mean"      # 序列特征池化: mean / max / sum
    mm_feat_proj_dim: int = 64       # mm_feat 投影后维度 (-1 = 不投影直接拼接)
    # 联合后训练输出
    joint_output_dir: str = ""       # 空则用 output_dir/joint_checkpoints
    # 评估
    eval_metrics: list = field(default_factory=lambda: ["auc", "gauc", "logloss"])
    ndcg_k: int = 10


# ────────────────────────────────────────────────────────────
# 预训练 / 后训练阶段配置
# ────────────────────────────────────────────────────────────

@dataclass
class PretrainConfig:
    """预训练阶段配置 (Latent 提取 + FM Teacher)"""
    stages: list = field(default_factory=lambda: ["latent", "fm"])
    jsonl_path: str = ""
    embedding_dir: str = ""     # 空则用 output_dir/embeddings
    fm_output_dir: str = ""     # 空则用 output_dir/fm_checkpoints
    resume_from: str = ""       # 断点续训 checkpoint 路径


@dataclass
class PosttrainConfig:
    """后训练阶段配置 (Joint Post-train + CF-OPD)"""
    stages: list = field(default_factory=lambda: ["joint_train", "cf_opd"])
    fm_checkpoint: str = ""          # 预训练 FM checkpoint
    joint_checkpoint: str = ""       # joint post-train 后的 checkpoint (cf_opd 阶段用)
    embedding_dir: str = ""          # 用户 embedding 目录 (cf_opd 阶段用)
    joint_output_dir: str = ""       # joint 输出目录
    cf_opd_output_dir: str = ""      # cf_opd 输出目录


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
    cf_opd: CFOPDConfig = field(default_factory=CFOPDConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    ranker: RankerConfig = field(default_factory=RankerConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    posttrain: PosttrainConfig = field(default_factory=PosttrainConfig)

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
        cf_opd = CFOPDConfig(**d.get("cf_opd", {}))
        scenario = ScenarioConfig(**d.get("scenario", {}))
        ranker = RankerConfig(**d.get("ranker", {}))
        pretrain = PretrainConfig(**d.get("pretrain", {}))
        posttrain = PosttrainConfig(**d.get("posttrain", {}))
        return cls(
            seed=d.get("seed", 42),
            output_dir=d.get("output_dir", "./output"),
            device=d.get("device", "auto"),
            latent=latent,
            fm=fm,
            opd=opd,
            cf_opd=cf_opd,
            scenario=scenario,
            ranker=ranker,
            pretrain=pretrain,
            posttrain=posttrain,
        )

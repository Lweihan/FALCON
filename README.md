# FALCON

**Flow matching in the Advertising scenarios based on Latent Clue Online policy distillatioN**

---

## 概述

FALCON 是一个基于 Flow Matching + Online Policy Distillation 的用户特征预测框架，用于广告场景下的用户画像演变建模。

### 核心架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         FALCON                                   │
│                                                                   │
│  ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Latent  │───→│    FM    │───→│   OPD    │───→│ Scenario │  │
│  │  Clue    │    │ (Teacher)│    │ (Student)│    │          │  │
│  │ 提取     │    │ Flow     │    │ LLM      │    │ 广告应用  │  │
│  │          │    │ Matching │    │ +LoRA    │    │          │  │
│  └──────────┘    └───────────┘    └──────────┘    └──────────┘  │
│       ↓               ↓                ↓              ↓          │
│  XML+Embed     预测未来Emb      蒸馏对齐LLM     在线推理服务     │
└─────────────────────────────────────────────────────────────────┘
```

### 四大模块

| 模块 | 文件夹 | 说明 |
|------|--------|------|
| **Latent Clue** | `latent/` | XML 结构化 → qwen3-embedding 分部分编码 → 注意力融合 → 融合 Embedding |
| **Flow Matching** | `fm/` | OT-CFM 前向加噪 + 逆向 ODE 求解，预测用户未来 Embedding (Teacher) |
| **Online Policy Distillation** | `opd/` | FM Teacher → LLM Student 知识蒸馏，含 Online 滚动更新 |
| **Scenario** | `scenario/` | 广告场景应用：数据加载、评估指标、在线服务 |

### 数据流

```
JSONL (用户画像时间序列)
  │
  ▼  latent/
XML 结构化 (4 sections: static, interest, consumption, demand)
  │
  ▼  latent/
分部分 Embedding (qwen3-embedding × 4)
  │
  ▼  latent/
注意力融合 → Fused Embedding per Timestep
  │
  ▼  fm/
Flow Matching (Teacher):
  最晚 embedding = 干净目标
  前向: x_t = (1-t)*e_T + t*ε
  逆向: v_θ(x_t, t, condition) → 预测未来 Embedding
  │
  ▼  opd/
Online Policy Distillation (Student):
  Teacher 预测 target embedding
  LLM 生成画像文本 → 编码为 embedding
  蒸馏对齐: L = L_gen + α*L_emb + β*L_section + γ*L_kl
  │
  ▼  scenario/
广告场景: 预测用户未来画像 → 优化广告投放策略
```

---

## 项目结构

```
falcon/
├── common/                      # 通用基础设施
│   ├── config.py               # 统一配置管理 (dataclass + YAML)
│   ├── logger.py               # 日志
│   └── utils.py                # 工具函数
│
├── latent/                      # Latent Clue 潜在线索提取
│   ├── xml_structured.py       # JSONL → XML 4-part 结构化
│   ├── embedder.py             # qwen3-embedding 分部分编码
│   ├── attention_fusion.py    # Multi-Head Attention 融合
│   └── pipeline.py             # 端到端提取管线
│
├── fm/                          # Flow Matching (Teacher)
│   ├── flow_core.py            # OT-CFM 前向过程 (加噪 + 速度场)
│   ├── dit_backbone.py         # DiT 速度场网络 (推荐, 可配置到 4B)
│   ├── velocity_net.py         # MLP / Transformer 速度场网络 (轻量/兼容)
│   ├── model.py                # FlowMatchingModel (训练 + 推理 + 可微优化)
│   └── trainer.py              # FM 训练器 (AMP + EMA loss + TensorBoard)
│
├── opd/                         # Online Policy Distillation (Student)
│   ├── teacher.py              # FM Teacher 封装 (predict + EMA)
│   ├── student.py              # LLM Student 封装 (LoRA + 生成 + KL)
│   ├── distillation.py         # 蒸馏核心 (OPD Loss + Engine)
│   ├── reward.py               # 奖励塑形 + Replay Buffer
│   └── trainer.py              # OPD 训练器 (Offline + Online)
│
├── ranker/                      # 下游精排模型 (验证 FALCON embedding 效果)
│   ├── base_ranker.py          # 精排模型基类 (extract_embedding 接口)
│   ├── models.py               # DNN / DeepFM / DCN / AutoInt / TokenMixer
│   ├── joint_falcon_ranker.py  # FALCON FM + Ranker 联合训练
│   ├── data.py                 # 稀疏特征数据集 (支持合成数据)
│   ├── metrics.py              # AUC / GAUC / LogLoss / NDCG@K
│   └── trainer.py              # 网格实验训练器 (模型 × fm_weight)
│
├── scenario/                    # 广告场景应用
│   ├── data_loader.py          # 数据集 + DataLoader
│   ├── evaluator.py            # 评估指标 (cos_sim, MSE, NDCG)
│   └── serving.py              # 在线推理服务
│
├── scripts/                     # 入口脚本
│   ├── pretrain.py             # 【预训练】Latent 提取 + FM Teacher
│   ├── posttrain.py            # 【后训练】OPD 蒸馏 + 下游 Ranker
│   ├── preprocess.py           # 数据预处理 (单步)
│   ├── train_fm.py             # 训练 FM Teacher (单步)
│   ├── train_opd.py            # 训练 OPD (单步)
│   ├── inference.py            # 推理
│   └── evaluate.py             # 评估
│
├── configs/                     # 配置文件
│   └── falcon_config.yaml      # 全局配置
│
├── feature_extract/             # 参考代码 (原始处理脚本)
├── requirements.txt
└── README.md
```

---

## FM Backbone 架构

FALCON 的速度场网络提供三种 backbone，通过 `fm.backbone_type` 切换。

### DiT (推荐默认)

现代化 Diffusion Transformer 架构，关键设计：

| 技术 | 作用 |
|------|------|
| **Patchify Tokenization** | 将 `data_dim=256` 切分为 16 个 token，让 Self-Attention 真正有序列可建模（旧版只有 3 个 token，注意力近乎退化） |
| **AdaLN-Zero** | 时间 + 条件 → 每个 Block 的 (shift, scale, gate)×2；gate 零初始化保证深层网络稳定收敛 |
| **QK-Norm** | Query/Key 做 RMSNorm 后再算注意力，抑制 logits 爆炸 |
| **RoPE** | 旋转位置编码，外推性优于绝对位置编码 |
| **SwiGLU FFN** | 表达力优于标准 GELU FFN |
| **RMSNorm** | 无 bias、不算 mean，更轻更稳 |
| **Zero-init 输出层** | 训练初期输出零速度场，等价恒等流 |
| **Gradient Checkpointing** | 大规模训练用计算换显存 |

规模预设（`fm.dit_preset`）：

| preset | dim | depth | heads | patches | 参数量 | 适用场景 |
|--------|-----|-------|-------|---------|--------|---------|
| `tiny` | 384 | 4 | 6 | 8 | ~11M | 快速调试 |
| `small` | 768 | 8 | 12 | 16 | ~89M | 小数据集 |
| **`base`** | 1024 | 16 | 16 | 16 | **~308M** | **推荐默认** |
| `large` | 1536 | 32 | 16 | 16 | ~1.37B | 大数据集 |
| `xl` | 2048 | 48 | 16 | 32 | ~3.65B | 超大规模 |

### 其他 backbone

| backbone_type | 说明 | 参数量 |
|---------------|------|--------|
| `transformer` | 旧版 3-token Transformer（保留兼容） | ~33M |
| `mlp` | ResidualBlock MLP（轻量） | ~3.6M |

---

## 下游 Ranker 模块

用于验证 FALCON 产出的 embedding 在真实广告排序任务上的效果。

### 两条对比路径

```
直接预测:     features → Ranker → CTR
FALCON 优化:  features → Ranker.extract_embedding → FM 流形优化 → CTR
```

### 联合训练损失

```
L = α·L_direct + (1-α)·[fm_weight·L_velocity + (1-fm_weight)·L_BCE_falcon]
```

关键点：Ranker 的 embedding 层与 FM 的 velocity_net 同时接收梯度，
使 Ranker 学会生成更适合 FM 流形优化的 embedding。

### 支持的精排模型

| 模型 | 核心机制 |
|------|---------|
| **DNN** | Embedding + MLP |
| **DeepFM** | 一阶线性 + FM 二阶交叉 + Deep |
| **DCN** | Cross Network 显式高阶交叉 + Deep |
| **AutoInt** | Multi-Head Self-Attention 自动特征交互 |
| **TokenMixer** | Token Mixing (RMSNorm + PertokenSwiGLU) |

### 评估指标

AUC / GAUC (按用户分组) / LogLoss / NDCG@K，并自动计算 FALCON 相对直接预测的 **lift**。

---

## OPD (Online Policy Distillation) 详解

### 核心思想

将 Flow Matching Teacher 学到的用户演变规律蒸馏到 LLM Student，使 LLM 能直接生成与未来特征对齐的用户画像。

### 蒸馏损失

```
L_total = L_gen + α·L_emb + β·L_section + γ·L_kl

L_gen:     语言模型 CrossEntropy 生成损失
L_emb:     1 - cos_sim(e_student, e_teacher) + MSE
L_section: Σ_k [1 - cos_sim(s_k, t_k) + MSE]   (4 部分对齐)
L_kl:      KL(π_θ || π_ref) · τ²                (防止遗忘)
```

### Teacher (FM Model)

- 根据当前 condition embedding 预测未来 embedding
- 使用 EMA 保持稳定蒸馏信号
- 推理时通过 ODE 求解: `x_{t-dt} = x_t - v_θ(x_t, t) · dt`

### Student (LLM + LoRA)

- 大模型 + LoRA 高效微调
- 生成用户画像文本 → 通过 Latent Pipeline 编码为 embedding
- 与 Teacher 预测 target 对齐
- KL 散度防止遗忘原始能力

### Online 策略

1. **滚动更新**: 新数据到达 → Teacher 更新预测 → 蒸馏给 Student
2. **Replay Buffer**: 存储历史蒸馏样本，防止灾难性遗忘
3. **EMA Teacher**: Teacher 权重 EMA 平滑，提供稳定蒸馏目标
4. **Reward Shaping**: Teacher-Student 对齐度 → 标量奖励 → 策略梯度

---

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 预训练 (推荐入口)

预训练一次完成 Latent Clue 提取 + FM Teacher 训练：

```bash
python scripts/pretrain.py \
    --config configs/falcon_config.yaml \
    --jsonl_path /path/to/output_cleaned.jsonl
```

分阶段执行：

```bash
# 仅提取 Latent Clue
python scripts/pretrain.py --stage latent --jsonl_path /path/to/data.jsonl

# 仅训练 FM (已有 embedding)
python scripts/pretrain.py --stage fm --embedding_dir ./output/embeddings
```

切换模型规模 / 开启显存优化：

```bash
# 用 large 规模 (~1.37B) + 混合精度 + 梯度检查点
python scripts/pretrain.py \
    --jsonl_path /path/to/data.jsonl \
    --dit_preset large --use_amp --grad_ckpt

# 用轻量 MLP 快速验证
python scripts/pretrain.py --jsonl_path /path/to/data.jsonl --backbone mlp
```

### 3. 后训练 (推荐入口)

后训练一次完成 OPD 蒸馏 + 下游 Ranker 训练：

```bash
python scripts/posttrain.py \
    --config configs/falcon_config.yaml \
    --fm_checkpoint ./output/fm_checkpoints/best_fm.pt \
    --embedding_dir ./output/embeddings
```

分阶段执行：

```bash
# 仅 OPD 蒸馏 (FM Teacher → LLM Student)
python scripts/posttrain.py --stage opd \
    --fm_checkpoint ./output/fm_checkpoints/best_fm.pt \
    --embedding_dir ./output/embeddings

# 仅下游 Ranker (用合成数据快速验证流程)
python scripts/posttrain.py --stage ranker \
    --fm_checkpoint ./output/fm_checkpoints/best_fm.pt \
    --synthetic --max_samples 20000

# Ranker 消融实验: 指定模型和 FM 权重
python scripts/posttrain.py --stage ranker \
    --rankers dnn deepfm token_mixer \
    --fm_weights 0.1 0.5 1.0

# 使用真实排序数据
python scripts/posttrain.py --stage ranker \
    --data_path /path/to/ranking_data.parquet \
    --sparse_features user_id item_id cate_id brand_id
```

### 4. TensorBoard 可视化

```bash
# 预训练监控
tensorboard --logdir ./output/fm_checkpoints/tensorboard

# 后训练监控
tensorboard --logdir ./output/ranker_checkpoints/tensorboard
```

### 5. 评估与推理

```bash
# 评估 FM Teacher
python scripts/evaluate.py \
    --config configs/falcon_config.yaml \
    --fm_checkpoint ./output/fm_checkpoints/best_fm.pt \
    --data_dir ./output/embeddings

# FM Teacher 推理
python scripts/inference.py \
    --fm_checkpoint ./output/fm_checkpoints/best_fm.pt \
    --embedding_file ./output/embeddings/user_000000.pt

# FALCON 完整推理 (Teacher + Student)
python scripts/inference.py \
    --config configs/falcon_config.yaml \
    --fm_checkpoint ./output/fm_checkpoints/best_fm.pt \
    --lora_path ./output/opd_checkpoints/final_lora \
    --jsonl_path /path/to/output_cleaned.jsonl
```

### 6. 单步脚本 (向后兼容)

```bash
python scripts/preprocess.py --jsonl_path /path/to/data.jsonl
python scripts/train_fm.py --data_dir ./output/embeddings
python scripts/train_opd.py --fm_checkpoint ... --data_dir ...
```

---

## 配置说明

全局配置在 `configs/falcon_config.yaml`，关键参数:

### Latent Clue

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `latent.embed_model_path` | Qwen/Qwen3-Embedding-0.6B | Embedding 模型路径 |
| `latent.embed_dim` | 256 | 每部分 embedding 维度 |
| `latent.fusion_output_dim` | 256 | 融合输出维度 (须 = `fm.data_dim`) |

### Flow Matching

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `fm.backbone_type` | dit | dit / transformer / mlp |
| `fm.dit_preset` | base | tiny / small / base / large / xl |
| `fm.dit_num_patches` | 16 | Patch 数量 (须能整除 data_dim) |
| `fm.dit_qk_norm` | true | QK-Norm 稳定性技术 |
| `fm.dit_use_rope` | true | 旋转位置编码 |
| `fm.use_gradient_checkpointing` | false | 显存优化 (大规模必开) |
| `fm.use_amp` | false | 混合精度训练 |
| `fm.time_weight_mode` | mid | 时间加权: none / mid / late / early |
| `fm.ema_loss_alpha` | 0.1 | Loss EMA 平滑系数 |

### OPD 蒸馏

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `opd.student_model_path` | Qwen/Qwen3-8B | Student LLM 路径 |
| `opd.lora_r` | 16 | LoRA rank |
| `opd.distill_alpha` | 1.0 | L_emb 权重 |
| `opd.distill_beta` | 0.5 | L_section 权重 |
| `opd.distill_gamma` | 0.1 | L_kl 权重 (防遗忘) |
| `opd.distill_tau` | 2.0 | KL 温度参数 |
| `opd.ema_decay` | 0.999 | Teacher EMA 衰减 |

### 下游 Ranker

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ranker.ranker_types` | 全部 5 种 | 要训练的精排模型 |
| `ranker.joint_alpha` | 0.5 | 直接预测损失权重 |
| `ranker.fm_weights` | [0.1, 0.5, 1.0] | FM velocity 损失权重消融 |
| `ranker.freeze_fm` | false | 是否冻结 FM backbone |
| `ranker.fast_train` | true | 训练时跳过 ODE 求解 (更快) |
| `ranker.eval_ode_steps` | 50 | 评估时 ODE 步数 |
| `ranker.data_path` | "" | 排序数据路径 (留空用合成数据) |

---

## 输入数据格式

JSONL 文件每行一个用户，含 `records` 数组，每个 record 有时间步和 4 个画像部分:

```json
{
  "records": [
    {
      "time": "202606262200",
      "basic_profile": {"年龄区间": "31-40", "性别": "男", ...},
      "local_scene": {"居住环境": "...", ...},
      "other_scene": {"作息与活跃时段": "...", ...},
      "interest_profile": [{"领域": "...", "评分": "9/10"}, ...],
      "consumption_profile": {"能力层级": "...", "消费预测": [...], ...},
      "demand_targets": [{"行业": "...", "深转潜力": "9/10", ...}, ...]
    }
  ]
}
```

---

## License

Internal use only.

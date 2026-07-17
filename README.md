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
│   ├── velocity_net.py         # 速度场预测网络 v_θ
│   ├── model.py                # FlowMatchingModel (训练 + 推理)
│   └── trainer.py              # FM 训练器
│
├── opd/                         # Online Policy Distillation (Student)
│   ├── teacher.py              # FM Teacher 封装 (predict + EMA)
│   ├── student.py              # LLM Student 封装 (LoRA + 生成 + KL)
│   ├── distillation.py         # 蒸馏核心 (OPD Loss + Engine)
│   ├── reward.py               # 奖励塑形 + Replay Buffer
│   └── trainer.py              # OPD 训练器 (Offline + Online)
│
├── scenario/                    # 广告场景应用
│   ├── data_loader.py          # 数据集 + DataLoader
│   ├── evaluator.py            # 评估指标 (cos_sim, MSE, NDCG)
│   └── serving.py              # 在线推理服务
│
├── scripts/                     # 入口脚本
│   ├── preprocess.py           # 数据预处理
│   ├── train_fm.py             # 训练 FM Teacher
│   ├── train_opd.py            # 训练 OPD (Student 蒸馏)
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

### 2. 数据预处理

```bash
python scripts/preprocess.py \
    --config configs/falcon_config.yaml \
    --jsonl_path /path/to/output_cleaned.jsonl
```

### 3. 训练 FM Teacher

```bash
python scripts/train_fm.py \
    --config configs/falcon_config.yaml \
    --data_dir ./output/embeddings
```

### 4. OPD 蒸馏 (FM → LLM)

```bash
python scripts/train_opd.py \
    --config configs/falcon_config.yaml \
    --fm_checkpoint ./output/fm_checkpoints/best_fm.pt \
    --data_dir ./output/embeddings
```

### 5. 评估

```bash
python scripts/evaluate.py \
    --config configs/falcon_config.yaml \
    --fm_checkpoint ./output/fm_checkpoints/best_fm.pt \
    --data_dir ./output/embeddings
```

### 6. 推理

```bash
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

---

## 配置说明

全局配置在 `configs/falcon_config.yaml`，关键参数:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `latent.embed_dim` | 256 | 每部分 embedding 维度 |
| `latent.fusion_output_dim` | 256 | 融合输出维度 (= fm.data_dim) |
| `fm.hidden_dim` | 512 | Velocity Network 隐藏维度 |
| `fm.num_layers` | 6 | Velocity Network 层数 |
| `opd.student_model_path` | Qwen/Qwen3-8B | Student LLM 路径 |
| `opd.lora_r` | 16 | LoRA rank |
| `opd.distill_alpha` | 1.0 | L_emb 权重 |
| `opd.distill_beta` | 0.5 | L_section 权重 |
| `opd.distill_gamma` | 0.1 | L_kl 权重 |
| `opd.distill_tau` | 2.0 | KL 温度参数 |
| `opd.ema_decay` | 0.999 | Teacher EMA 衰减 |

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

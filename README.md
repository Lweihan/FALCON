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

## 训练样本构造策略

### 配对模式 (`fm.pair_mode`)

FM 预训练的输入是「(condition, target) 样本对」，从用户的时间序列 embedding 中构造。
三种策略可配置切换：

| 模式 | 说明 | T=4 时样本数 | 特点 |
|------|------|------------|------|
| **`last_target`** | 随机早期步 → 最后一步 | 1 | 默认；目标固定为最终态 |
| **`all_pairs`** | 枚举所有 (i,j) i&lt;j | 6 | 数据利用率 100%，覆盖所有时间跨度 |
| **`adjacent`** | 相邻配对 (i, i+1) | 3 | 只学短期演变 |

```bash
# 切换到全配对模式 (样本量提升 6 倍)
python scripts/pretrain.py --stage fm --pair_mode all_pairs
```

### delta_t 条件注入

**解决的问题**：训练时 Flow Matching 的 `t` 只是噪声插值系数，与用户真实时间跨度无关。
若不注入 `delta_t`，`e0 → e3`（跨 3 步）和 `e2 → e3`（跨 1 步）在模型看来是完全相同的任务，
但推理时却用 `delta_t` 控制预测多远的未来——存在训练/推理语义不一致。

**做法**：把真实时间跨度作为额外条件注入网络。

```
c = time_embed(t) + cond_embed(condition) + delta_t_embed(delta_t)
```

`delta_t` 计算优先解析真实时间戳，失败则回退到步数比例：

| 配对 | 真实时间差 | delta_t (时间戳) | delta_t (步数) |
|------|-----------|-----------------|---------------|
| 0→1 | 12h | 0.231 | 0.333 |
| 2→3 | 20h | 0.385 | 0.333 |
| 0→3 | 52h | 1.000 | 1.000 |

时间戳版本能反映真实间隔差异，步数版本无法区分。

关闭方式（退化为旧行为，便于 AB 对比）：
```bash
python scripts/pretrain.py --stage fm --no_delta_t_condition
```

---

## 多卡训练与性能优化

### 多卡训练 (DDP)

使用 `torchrun` 启动，数据并行：

```bash
# 4 卡训练，每卡 batch=64，全局 batch=256
torchrun --nproc_per_node=4 scripts/pretrain.py \
    --stage fm --embedding_dir ./output/embeddings \
    --batch_size 64

# 也可以直接指定全局 batch，自动换算 per-GPU
torchrun --nproc_per_node=4 scripts/pretrain.py \
    --stage fm --embedding_dir ./output/embeddings \
    --global_batch_size 256
```

设计要点：
- `batch_size` 语义明确为 **per-GPU**，日志会打印 `per_gpu / world_size / global`
- 用 `DistributedDataParallel`（非 `DataParallel`，后者有 GIL 瓶颈）
- `DistributedSampler` + 每 epoch `set_epoch()` 保证 shuffle 有效
- 只有 rank 0 写 TensorBoard / 保存 checkpoint / 打印日志
- loss/metrics 通过 `all_reduce` 跨卡平均后才记录
- checkpoint 自动剥离 DDP 包装（`unwrap_model`）
- 单卡直接 `python scripts/pretrain.py` 即可，无需 torchrun

### GPU 利用率优化策略

我们观察到默认配置下 GPU 利用率约 50%，主要有三个瓶颈。以下是对应的优化手段。

#### 瓶颈 1：验证阶段的 ODE 串行求解（影响最大）

验证时要通过 ODE 求解预测未来 embedding，这个过程是**严格串行迭代**的：

```python
for step in range(num_steps):       # 50 次串行前向
    v = velocity_net(x_t, t, cond)  # 每次都要等上一步结果
    x_t = x_t - v * dt
```

GPU 在这期间无法并行，且每次前向的 batch 很小，kernel launch 开销占比高。

| 配置项 | 默认 | 作用 |
|--------|------|------|
| `val_ode_steps` | **10** | 验证 ODE 步数（原 50）。验证只为看趋势，不需要高精度 |
| `val_every` | 1 | 每 N epoch 才验证一次 |
| `val_max_batches` | -1 | 限制验证只跑前 N 个 batch |
| `val_compute_cos_sim` | true | 设 false 则**完全跳过 ODE**，验证瞬间完成 |

```bash
# 激进优化: 每 5 epoch 验证一次，且不跑 ODE
python scripts/pretrain.py --stage fm --val_every 5 --no_val_cos_sim
```

#### 瓶颈 2：数据加载单线程

原实现 `num_workers=0`，且 `__init__` 一次性把所有 `.pt` 加载进内存，
CPU 单线程喂数据导致 GPU 饿着。

| 配置项 | 默认 | 作用 |
|--------|------|------|
| `num_workers` | **4** | 并行 IO 进程数，建议 4-8 |
| `lazy_load` | **true** | 懒加载 `.pt`，让 IO 在 worker 进程并行 |
| `prefetch_factor` | 2 | 每个 worker 预取批次数 |
| `persistent_workers` | 自动 | `num_workers>0` 时开启，避免每 epoch 重建进程 |
| `pin_memory` | 自动 | CUDA 可用时开启，加速 H2D 拷贝 |

数据传输还配合了 `non_blocking=True` 实现异步拷贝。

#### 瓶颈 3：batch 过小 + 未开启计算加速

DiT base 有 308M 参数，`batch_size=32` 时单次前向太快，GPU 大部分时间在等 kernel launch。

| 配置项 | 默认 | 作用 |
|--------|------|------|
| `use_tf32` | **true** | Ampere+ (A100/H100/RTX30+) 矩阵乘用 TF32，显著加速且精度损失极小 |
| `cudnn_benchmark` | **true** | 输入尺寸固定时自动搜索最优 kernel |
| `use_amp` | false | 混合精度，建议开启，速度提升明显 |
| `use_compile` | false | `torch.compile`，稳态后快 20-30%，但首个 epoch 编译较慢 |
| `grad_accum_steps` | 1 | 梯度累积，显存不足时等效放大 batch |

DDP + 梯度累积组合时，累积中间步会自动 `no_sync()` 跳过梯度同步，减少通信开销。

### 推荐配置组合

**4 卡追求吞吐**
```bash
torchrun --nproc_per_node=4 scripts/pretrain.py --stage fm \
    --embedding_dir ./output/embeddings \
    --batch_size 128 --num_workers 8 --use_amp \
    --val_every 5 --val_ode_steps 10
```

**显存受限（大模型）**
```bash
torchrun --nproc_per_node=4 scripts/pretrain.py --stage fm \
    --embedding_dir ./output/embeddings \
    --dit_preset large --batch_size 16 --grad_accum 8 \
    --use_amp --grad_ckpt
```

**快速调试**
```bash
python scripts/pretrain.py --stage fm \
    --embedding_dir ./output/embeddings \
    --dit_preset tiny --epochs 2 --no_val_cos_sim
```

### 性能监控

TensorBoard 中新增了性能指标，可直接观察优化效果：

| 指标 | 说明 |
|------|------|
| `perf/epoch_seconds` | 每 epoch 耗时 |
| `perf/samples_per_sec` | 吞吐量（样本/秒） |
| `meta/global_batch_size` | 实际全局 batch |
| `meta/world_size` | 卡数 |
| `meta/num_train_pairs` | 训练样本对数（可看出 pair_mode 的扩增效果） |
| `step/delta_t_mean` | 平均时间跨度 |

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

### 训练样本构造

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `fm.pair_mode` | last_target | last_target / all_pairs / adjacent |
| `fm.use_delta_t_condition` | true | delta_t 条件注入 (修训推不一致) |
| `fm.delta_t_from_timestamp` | true | 优先用真实时间戳算 delta_t |

### 性能与多卡

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `fm.num_workers` | 4 | DataLoader 并行进程数 |
| `fm.lazy_load` | true | 懒加载 .pt (配合 num_workers) |
| `fm.val_every` | 1 | 每 N epoch 验证一次 |
| `fm.val_ode_steps` | 10 | 验证 ODE 步数 (越小越快) |
| `fm.val_compute_cos_sim` | true | false 则跳过 ODE，验证极快 |
| `fm.use_tf32` | true | Ampere+ 矩阵乘加速 |
| `fm.cudnn_benchmark` | true | 自动选最优 kernel |
| `fm.use_compile` | false | torch.compile |
| `fm.grad_accum_steps` | 1 | 梯度累积 |
| `fm.distributed` | false | 由 torchrun 自动检测 |

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

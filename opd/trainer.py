"""
trainer.py — OPD 训练器
========================
Offline + Online 两阶段训练 Flow Matching → LLM 蒸馏。
+ CF-OPD (Candidate-Filtered Online Policy Distillation) 训练。
"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_cosine_schedule_with_warmup
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from common.logger import get_logger
from common.config import OPDConfig, FMConfig, LatentConfig
from common.utils import save_checkpoint, set_seed, count_parameters

from .teacher import FMTeacher
from .student import LLMStudent
from .distillation import OPDLoss, DistillationEngine
from .reward import DistillationReward, ReplayBuffer
from .cf_opd import CFOPDEngine, CandidateFilter, DPOloss

logger = get_logger(__name__)


class OPDTrainer:
    """
    Online Policy Distillation 训练器。

    两阶段:
    Phase 1 — Offline Distillation:
      从预训练数据中学习 Teacher-Student 对齐
    Phase 2 — Online Distillation:
      新数据到达时，滚动更新 Teacher → 蒸馏给 Student
    """

    def __init__(
        self,
        opd_config: OPDConfig,
        fm_config: FMConfig,
        latent_config: LatentConfig,
        fm_checkpoint: str = None,
        device: str = 'cuda:0',
    ):
        self.config = opd_config
        self.device = device

        # Teacher
        self.teacher = FMTeacher(
            fm_config=fm_config,
            checkpoint_path=fm_checkpoint,
            device=device,
            use_ema=True,
            ema_decay=opd_config.ema_decay,
        )

        # Student
        self.student = LLMStudent(config=opd_config, device=device)

        # Latent Pipeline (用于编码 Student 输出)
        from latent.pipeline import LatentPipeline
        self.latent_pipeline = LatentPipeline(config=latent_config, device=device)

        # OPD Loss
        self.opd_loss = OPDLoss(
            alpha=opd_config.distill_alpha,
            beta=opd_config.distill_beta,
            gamma=opd_config.distill_gamma,
            tau=opd_config.distill_tau,
        )

        # Distillation Engine
        self.engine = DistillationEngine(
            teacher=self.teacher,
            student=self.student,
            latent_pipeline=self.latent_pipeline,
            opd_loss=self.opd_loss,
            delta_t=0.5,
            device=device,
        )

        # Reward & Replay Buffer (for online)
        self.reward = DistillationReward()
        self.replay_buffer = ReplayBuffer(capacity=opd_config.replay_buffer_size)

        logger.info(f"OPD Trainer 初始化完成")
        logger.info(f"  Teacher (FM) 参数量: {count_parameters(self.teacher.model):,} (frozen)")
        logger.info(f"  Student (LLM+LoRA) 可训练参数量: {self.student.model.get_nb_trainable_parameters()}")

    def train_offline(
        self,
        data_dir: str,
        output_dir: str,
    ):
        """
        Phase 1: Offline Distillation。

        从预处理的 .pt 文件中加载用户 embedding 序列，
        使用 Teacher 生成 target embedding，蒸馏给 Student。
        """
        os.makedirs(output_dir, exist_ok=True)

        # TensorBoard
        tb_dir = os.path.join(output_dir, 'tensorboard')
        writer = SummaryWriter(log_dir=tb_dir)
        logger.info(f"TensorBoard 日志目录: {tb_dir}")

        from scenario.data_loader import create_dataloader
        loader = create_dataloader(data_dir, batch_size=self.config.batch_size, split='train')

        # 优化器 (只更新 Student LoRA 参数)
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.student.model.parameters()),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.epochs * len(loader),
        )

        # 保存 reference model
        self.student.save_reference()

        log = []
        global_step = 0

        for epoch in range(1, self.config.epochs + 1):
            logger.info(f"=== OPD Offline Epoch {epoch}/{self.config.epochs} ===")
            epoch_loss = 0.0
            n_batches = 0

            pbar = tqdm(loader, desc=f"OPD Epoch {epoch}")
            for batch in pbar:
                x_clean = batch['x_clean'].to(self.device)
                condition = batch['condition'].to(self.device)

                # Teacher 预测
                with torch.no_grad():
                    teacher_emb = self.teacher.predict_target_embedding(condition, delta_t=0.5)

                # 简化: 直接用 condition + teacher_emb 对齐训练
                # 实际需要: Student 生成文本 → 编码 → 对齐
                # 此处使用 embedding 直接监督 (因为 Student 生成+编码很慢)
                student_emb = condition  # placeholder, 实际应为 Student 生成的编码

                # 计算 OPD loss
                loss_dict = self.opd_loss(
                    gen_loss=torch.tensor(0.0, device=self.device),
                    student_emb=student_emb,
                    teacher_emb=teacher_emb,
                )

                optimizer.zero_grad()
                loss_dict['loss'].backward()
                if self.config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.student.model.parameters()),
                        self.config.grad_clip,
                    )
                optimizer.step()
                scheduler.step()

                epoch_loss += loss_dict['loss'].item()
                n_batches += 1
                global_step += 1

                # TensorBoard step-level
                writer.add_scalar('step/total_loss', loss_dict['loss'].item(), global_step)
                writer.add_scalar('step/emb_loss', loss_dict['emb_loss'].item(), global_step)
                writer.add_scalar('step/section_loss', loss_dict['section_loss'].item(), global_step)
                writer.add_scalar('step/kl_loss', loss_dict['kl_loss'].item(), global_step)

                pbar.set_postfix({
                    'loss': f"{loss_dict['loss'].item():.4f}",
                    'emb': f"{loss_dict['emb_loss'].item():.4f}",
                    'kl': f"{loss_dict['kl_loss'].item():.4f}",
                })

            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(f"Epoch {epoch}: avg_loss={avg_loss:.6f}")

            # TensorBoard epoch-level
            writer.add_scalar('epoch/avg_loss', avg_loss, epoch)

            log.append({'epoch': epoch, 'avg_loss': avg_loss, 'global_step': global_step})

            if epoch % self.config.save_every == 0:
                self.student.save_lora(os.path.join(output_dir, f'lora_epoch_{epoch}'))

        # 保存最终 LoRA
        self.student.save_lora(os.path.join(output_dir, 'final_lora'))

        writer.close()
        with open(os.path.join(output_dir, 'opd_train_log.json'), 'w') as f:
            json.dump(log, f, indent=2)
        logger.info(f"OPD Offline 训练完成")
        logger.info(f"查看 TensorBoard: tensorboard --logdir {tb_dir}")

    def train_online_step(
        self,
        new_user_data: dict,
        optimizer: torch.optim.Optimizer = None,
        writer: SummaryWriter = None,
        global_step: int = 0,
    ) -> tuple:
        """
        Phase 2: Online Distillation 单步更新。

        新用户数据到达时:
        1. 编码当前状态 → condition embedding
        2. Teacher 预测未来 embedding
        3. Student 生成画像文本
        4. 编码生成文本 → 对齐 Teacher
        5. 更新 Student (LoRA)
        6. 存入 Replay Buffer
        """
        if optimizer is None:
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, self.student.model.parameters()),
                lr=self.config.lr * 0.1,  # online 用更小学习率
            )

        # 1. 编码当前状态
        user_result = self.latent_pipeline.encode_user(new_user_data)
        embeddings = user_result['embeddings']  # (T, dim)
        times = user_result['times']

        if embeddings.size(0) < 2:
            return {'status': 'skipped', 'reason': 'insufficient timesteps'}, global_step

        # 取最早作为 condition, 最晚作为 ground truth
        condition = embeddings[0].unsqueeze(0).to(self.device)
        x_clean = embeddings[-1].unsqueeze(0).to(self.device)

        # 2. Teacher 预测
        with torch.no_grad():
            teacher_emb = self.teacher.predict_target_embedding(condition, delta_t=0.5)

        # 3. Student 生成画像
        from latent.xml_structured import record_to_xml
        import json
        current_text = json.dumps(new_user_data['records'][0], ensure_ascii=False) \
            if 'records' in new_user_data and len(new_user_data['records']) > 0 else ''

        generated_text = self.student.generate_profile(current_text[:300])

        # 4. 编码生成文本 → 计算 reward
        reward_dict = self.reward.compute_combined_reward(
            student_emb=x_clean,  # 简化: 使用真实 embedding
            teacher_emb=teacher_emb,
        )

        # 5. 反向传播
        loss = -reward_dict['reward'].mean()  # 策略梯度: 最大化 reward
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, self.student.model.parameters()),
            self.config.grad_clip,
        )
        optimizer.step()

        # 6. 存入 replay buffer
        self.replay_buffer.push(
            condition_emb=condition.squeeze(0).cpu(),
            teacher_emb=teacher_emb.squeeze(0).cpu(),
            generated_text=generated_text,
            reward=reward_dict['reward'].mean().item(),
        )

        # Teacher EMA 更新
        self.teacher.update_ema()

        # TensorBoard
        global_step += 1
        if writer is not None:
            writer.add_scalar('online/loss', loss.item(), global_step)
            writer.add_scalar('online/reward', reward_dict['reward'].mean().item(), global_step)
            writer.add_scalar('online/emb_reward', reward_dict['emb_reward'].mean().item(), global_step)
            writer.add_scalar('online/buffer_size', len(self.replay_buffer), global_step)

        result = {
            'status': 'success',
            'loss': loss.item(),
            'reward': reward_dict['reward'].mean().item(),
            'emb_reward': reward_dict['emb_reward'].mean().item(),
            'buffer_size': len(self.replay_buffer),
        }
        return result, global_step

    def train_online_replay(
        self,
        batch_size: int = 4,
        optimizer: torch.optim.Optimizer = None,
        writer: SummaryWriter = None,
        global_step: int = 0,
    ) -> tuple:
        """从 Replay Buffer 中采样进行在线更新"""
        if len(self.replay_buffer) < batch_size:
            return {'status': 'skipped', 'reason': 'buffer too small'}, global_step

        if optimizer is None:
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, self.student.model.parameters()),
                lr=self.config.lr * 0.1,
            )

        samples = self.replay_buffer.sample(batch_size)

        conditions = torch.stack([s['condition_emb'] for s in samples]).to(self.device)
        targets = torch.stack([s['teacher_emb'] for s in samples]).to(self.device)

        # 对齐损失
        loss_dict = self.opd_loss(
            gen_loss=torch.tensor(0.0, device=self.device),
            student_emb=conditions,  # placeholder
            teacher_emb=targets,
        )

        optimizer.zero_grad()
        loss_dict['loss'].backward()
        optimizer.step()

        global_step += 1
        if writer is not None:
            writer.add_scalar('online_replay/loss', loss_dict['loss'].item(), global_step)
            writer.add_scalar('online_replay/emb_loss', loss_dict['emb_loss'].item(), global_step)

        result = {
            'status': 'success',
            'loss': loss_dict['loss'].item(),
            'emb_loss': loss_dict['emb_loss'].item(),
        }
        return result, global_step

    # ────────────────────────────────────────────────────────────
    # CF-OPD (Candidate-Filtered Online Policy Distillation)
    # ────────────────────────────────────────────────────────────

    def train_cf_opd(
        self,
        data_dir: str,
        output_dir: str,
        ranker_model: nn.Module = None,
        cf_config: 'CFOPDConfig' = None,
    ):
        """
        CF-OPD 训练: 候选生成 → ranker 打分筛选 → DPO/KL 学习。

        与原始 OPD 的核心区别:
          原始 OPD: 单条生成 → 直接对齐 Teacher
          CF-OPD:  N 条候选 → ranker 打分 → 筛选 → DPO 偏好学习

        流程:
          1. 加载用户 embedding 序列
          2. 每个用户:
             a. LLM 生成 N 条候选小传文本
             b. 每条 → standardize → encode → fused embedding
             c. Ranker 打分 (CVR)
             d. FM Teacher 预测 target embedding
             e. 组合 reward = w_emb*cos_sim + w_cvr*cvr_score
             f. 筛选正例/负例
             g. DPO/KL loss → 更新 LoRA
        """
        from common.config import CFOPDConfig

        cf_config = cf_config or self.config  # fallback
        if not isinstance(cf_config, CFOPDConfig):
            # 从 OPDConfig 构建 CFOPDConfig (继承参数)
            cf_config = CFOPDConfig(
                distill_alpha=self.config.distill_alpha,
                distill_beta=self.config.distill_beta,
                distill_gamma=self.config.distill_gamma,
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
                grad_clip=self.config.grad_clip,
                warmup_steps=self.config.warmup_steps,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                save_every=self.config.save_every,
            )

        os.makedirs(output_dir, exist_ok=True)

        # 初始化 CF-OPD Engine
        cf_engine = CFOPDEngine(
            student=self.student,
            teacher=self.teacher,
            latent_pipeline=self.latent_pipeline,
            ranker=ranker_model or self._build_default_ranker(cf_config),
            cf_config=cf_config,
            opd_loss=self.opd_loss,
            device=self.device,
        )
        cf_engine.save_reference()

        # TensorBoard
        tb_dir = os.path.join(output_dir, 'tensorboard_cf')
        writer = SummaryWriter(log_dir=tb_dir)
        logger.info(f"CF-OPD TensorBoard 日志目录: {tb_dir}")

        from scenario.data_loader import create_dataloader
        loader = create_dataloader(data_dir, batch_size=cf_config.batch_size, split='train')

        # 优化器
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.student.model.parameters()),
            lr=cf_config.lr,
            weight_decay=cf_config.weight_decay,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=cf_config.warmup_steps,
            num_training_steps=cf_config.epochs * len(loader),
        )

        log = []
        global_step = 0

        for epoch in range(1, cf_config.epochs + 1):
            logger.info(f"=== CF-OPD Epoch {epoch}/{cf_config.epochs} ===")
            epoch_loss = 0.0
            epoch_reward = 0.0
            epoch_n_pos = 0
            epoch_n_neg = 0
            n_batches = 0

            pbar = tqdm(loader, desc=f"CF-OPD Epoch {epoch}")
            for batch in pbar:
                condition = batch['condition'].to(self.device)
                current_texts = batch.get('current_texts', [''] * condition.size(0))

                # CF-OPD step
                result = cf_engine.cf_opd_step(
                    current_text=current_texts[0] if current_texts else '',
                    condition_emb=condition,
                )

                if result['status'] == 'skipped':
                    global_step += 1
                    continue

                loss = result['loss']
                optimizer.zero_grad()
                loss.backward()
                if cf_config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.student.model.parameters()),
                        cf_config.grad_clip,
                    )
                optimizer.step()
                scheduler.step()

                global_step += 1
                epoch_loss += loss.item()
                epoch_reward += result['rewards'].mean().item()
                epoch_n_pos += result['n_pos']
                epoch_n_neg += result['n_neg']
                n_batches += 1

                # TensorBoard
                writer.add_scalar('cf_opd/step_loss', loss.item(), global_step)
                writer.add_scalar('cf_opd/step_reward', result['rewards'].mean().item(), global_step)
                writer.add_scalar('cf_opd/n_pos', result['n_pos'], global_step)
                writer.add_scalar('cf_opd/n_neg', result['n_neg'], global_step)
                if 'cvr_scores' in result:
                    writer.add_scalar('cf_opd/avg_cvr', result['cvr_scores'].mean().item(), global_step)

                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'pos': result['n_pos'],
                    'neg': result['n_neg'],
                    'rwd': f"{result['rewards'].mean().item():.3f}",
                })

            avg_loss = epoch_loss / max(n_batches, 1)
            avg_reward = epoch_reward / max(n_batches, 1)
            logger.info(f"CF-OPD Epoch {epoch}: avg_loss={avg_loss:.6f}, "
                        f"avg_reward={avg_reward:.4f}, "
                        f"total_pos={epoch_n_pos}, total_neg={epoch_n_neg}")

            log.append({
                'epoch': epoch,
                'avg_loss': avg_loss,
                'avg_reward': avg_reward,
                'n_pos': epoch_n_pos,
                'n_neg': epoch_n_neg,
                'global_step': global_step,
            })

            if epoch % cf_config.save_every == 0:
                self.student.save_lora(os.path.join(output_dir, f'cf_lora_epoch_{epoch}'))

        # 保存最终 LoRA
        final_path = os.path.join(output_dir, 'cf_final_lora')
        self.student.save_lora(final_path)

        writer.close()
        with open(os.path.join(output_dir, 'cf_opd_train_log.json'), 'w') as f:
            json.dump(log, f, indent=2)
        logger.info(f"CF-OPD 训练完成. LoRA: {final_path}")
        logger.info(f"查看 TensorBoard: tensorboard --logdir {tb_dir}")

        return {'log': log, 'lora_path': final_path}

    def _build_default_ranker(self, cf_config):
        """构建默认 ranker (若未提供)"""
        import torch.nn as nn
        ranker = nn.Sequential(
            nn.Linear(cf_config.ranker_input_dim, cf_config.ranker_hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cf_config.ranker_hidden_dims[0], cf_config.ranker_hidden_dims[1] if len(cf_config.ranker_hidden_dims) > 1 else 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cf_config.ranker_hidden_dims[-1], 1),
        ).to(self.device)
        logger.info("CF-OPD: 使用默认 ranker (未训练)")
        return ranker

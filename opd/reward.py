"""
reward.py — 蒸馏奖励信号
===========================
用于 Online 策略蒸馏的奖励塑形 (Reward Shaping)。

在 Online 场景中，Teacher 提供的蒸馏信号可以作为 reward，
指导 Student 生成更准确的画像。支持:
  - Embedding 相似度奖励
  - 分部分对齐奖励
  - 时间一致性奖励
  - 组合奖励
"""

import torch
import torch.nn.functional as F
from collections import deque
from typing import Optional

from common.logger import get_logger

logger = get_logger(__name__)


class DistillationReward:
    """
    蒸馏奖励计算器。

    将 Teacher-Student 对齐程度转化为标量奖励信号，
    支持 Online 训练场景下的策略梯度优化。

    组合奖励:
    R = w1 * R_emb + w2 * R_section + w3 * R_temporal

    R_emb:     cosine_sim(e_student, e_teacher) ∈ [0, 1]
    R_section: Σ cosine_sim(s_k, t_k) / 4      ∈ [0, 1]
    R_temporal: 时间一致性奖励 (相邻时间步不应突变)
    """

    def __init__(
        self,
        w_emb: float = 1.0,
        w_section: float = 0.5,
        w_temporal: float = 0.1,
        ema_alpha: float = 0.01,    # 奖励 EMA 平滑
        reward_clip: float = 5.0,   # 奖励裁剪
    ):
        self.w_emb = w_emb
        self.w_section = w_section
        self.w_temporal = w_temporal
        self.ema_alpha = ema_alpha
        self.reward_clip = reward_clip

        # Running mean/std for reward normalization
        self._reward_mean = 0.0
        self._reward_var = 1.0
        self._count = 0

    def compute_embedding_reward(
        self,
        student_emb: torch.Tensor,
        teacher_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算整体 embedding 相似度奖励。

        Returns:
            (batch,) 每个样本的奖励 ∈ [0, 1]
        """
        cos_sim = F.cosine_similarity(student_emb, teacher_emb, dim=-1)
        return cos_sim.clamp(0.0, 1.0)

    def compute_section_reward(
        self,
        student_sections: dict,
        teacher_sections: dict,
    ) -> torch.Tensor:
        """
        计算分部分对齐奖励。

        Returns:
            (batch,) 平均 section 相似度 ∈ [0, 1]
        """
        section_names = ['static_information', 'interest_profile',
                         'consumption_profile', 'demand_targets']
        rewards = []
        for name in section_names:
            s_key = f'{name}_emb'
            t_key = f'{name}_target'
            if s_key in student_sections and t_key in teacher_sections:
                s = student_sections[s_key]
                t = teacher_sections[t_key]
                min_dim = min(s.size(-1), t.size(-1))
                sim = F.cosine_similarity(s[:, :min_dim], t[:, :min_dim], dim=-1)
                rewards.append(sim.clamp(0.0, 1.0))

        if rewards:
            return torch.stack(rewards, dim=0).mean(dim=0)
        return torch.zeros(student_sections.get('static_information_emb', torch.zeros(1)).size(0),
                           device=student_sections.get('static_information_emb', torch.zeros(1)).device)

    def compute_temporal_reward(
        self,
        current_emb: torch.Tensor,
        previous_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        时间一致性奖励: 相邻时间步的 embedding 不应有突变。

        Returns:
            (batch,) 一致性奖励 ∈ [0, 1]
        """
        if previous_emb is None:
            return torch.ones(current_emb.size(0), device=current_emb.device)

        # 相邻步的余弦相似度应较高
        sim = F.cosine_similarity(current_emb, previous_emb, dim=-1)
        # 奖励: 相似度越高越好，但不能完全一样 (需要变化)
        # R_temporal = sim * (1 - |sim - 0.9|)  鼓励相似度在 0.9 左右
        optimal_sim = 0.9
        reward = sim * (1.0 - torch.abs(sim - optimal_sim))
        return reward.clamp(0.0, 1.0)

    def compute_combined_reward(
        self,
        student_emb: torch.Tensor,
        teacher_emb: torch.Tensor,
        student_sections: dict = None,
        teacher_sections: dict = None,
        previous_emb: torch.Tensor = None,
    ) -> dict:
        """
        计算组合奖励。

        Returns:
            dict with 'reward' (batch,), 'emb_reward', 'section_reward', 'temporal_reward'
        """
        r_emb = self.compute_embedding_reward(student_emb, teacher_emb)

        r_section = torch.zeros_like(r_emb)
        if student_sections and teacher_sections:
            r_section = self.compute_section_reward(student_sections, teacher_sections)

        r_temporal = self.compute_temporal_reward(student_emb, previous_emb)

        # 组合
        combined = self.w_emb * r_emb + self.w_section * r_section + self.w_temporal * r_temporal

        # 裁剪
        combined = combined.clamp(-self.reward_clip, self.reward_clip)

        # EMA 平滑 & 归一化
        self._update_stats(combined)
        if self._count > 10:
            combined = (combined - self._reward_mean) / (self._reward_var.clamp(min=0.01) ** 0.5 + 1e-8)

        return {
            'reward': combined,
            'emb_reward': r_emb,
            'section_reward': r_section,
            'temporal_reward': r_temporal,
        }

    def _update_stats(self, rewards: torch.Tensor):
        """EMA 更新奖励的均值和方差"""
        mean = rewards.mean().item()
        var = rewards.var().item() if rewards.numel() > 1 else 1.0
        self._count += 1
        self._reward_mean = (1 - self.ema_alpha) * self._reward_mean + self.ema_alpha * mean
        self._reward_var = (1 - self.ema_alpha) * self._reward_var + self.ema_alpha * var


class ReplayBuffer:
    """
    经验回放缓冲区，用于 Online 蒸馏防止灾难性遗忘。

    存储: (condition_emb, teacher_emb, generated_text, reward)
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        condition_emb: torch.Tensor,
        teacher_emb: torch.Tensor,
        generated_text: str,
        reward: float,
    ):
        """添加一条经验"""
        self.buffer.append({
            'condition_emb': condition_emb.cpu(),
            'teacher_emb': teacher_emb.cpu(),
            'generated_text': generated_text,
            'reward': reward,
        })

    def sample(self, batch_size: int) -> list[dict]:
        """随机采样一批经验"""
        import random
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)

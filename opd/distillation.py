"""
distillation.py — 蒸馏核心逻辑
=================================
实现 Teacher-Student 间的知识蒸馏损失计算与蒸馏引擎。

蒸馏损失组成:
  L_total = L_gen + α * L_emb + β * L_section + γ * L_kl

  - L_gen:    语言模型生成 loss (CrossEntropy)
  - L_emb:    整体 embedding 对齐 (Student 输出 vs Teacher 预测)
  - L_section: 分部分 embedding 对齐 (4 个部分分别对齐)
  - L_kl:     KL 散度 (防止 Student 遗忘原始能力)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.logger import get_logger

logger = get_logger(__name__)


class OPDLoss(nn.Module):
    """
    Online Policy Distillation 联合损失。

    L = L_gen + alpha * L_emb + beta * L_section + gamma * L_kl

    L_emb:    1 - cos_sim(e_student, e_teacher)  + MSE
    L_section: Σ_{k=1}^{4} [1 - cos_sim(s_k_student, s_k_teacher) + MSE(s_k)]
    L_kl:     KL(π_θ || π_ref) * tau^2
    """

    SECTION_NAMES = ['static_information', 'interest_profile', 'consumption_profile', 'demand_targets']

    def __init__(
        self,
        alpha: float = 1.0,     # L_emb 权重
        beta: float = 0.5,       # L_section 权重
        gamma: float = 0.1,      # L_kl 权重
        tau: float = 2.0,        # KL 温度
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

    def embedding_loss(
        self,
        student_emb: torch.Tensor,
        teacher_emb: torch.Tensor,
    ) -> dict:
        """
        整体 embedding 对齐损失。

        Args:
            student_emb: Student 生成的画像经编码后的 embedding, (batch, dim)
            teacher_emb: Teacher 预测的未来 embedding, (batch, dim)

        Returns:
            dict with 'cos_loss', 'mse_loss', 'total'
        """
        # Cosine similarity loss: 1 - cos_sim
        cos_sim = F.cosine_similarity(student_emb, teacher_emb, dim=-1)
        cos_loss = (1.0 - cos_sim).mean()

        # MSE loss
        mse_loss = F.mse_loss(student_emb, teacher_emb)

        return {
            'cos_loss': cos_loss,
            'mse_loss': mse_loss,
            'total': cos_loss + 0.5 * mse_loss,
        }

    def section_loss(
        self,
        student_sections: dict,
        teacher_sections: dict,
    ) -> dict:
        """
        分部分 embedding 对齐损失。

        Args:
            student_sections: {name_emb: (batch, dim)} for 4 sections
            teacher_sections: {name_target: (batch, dim)} for 4 sections

        Returns:
            dict with per-section losses and total
        """
        total = 0.0
        details = {}

        for name in self.SECTION_NAMES:
            s_key = f'{name}_emb'
            t_key = f'{name}_target'

            if s_key in student_sections and t_key in teacher_sections:
                s_emb = student_sections[s_key]
                t_emb = teacher_sections[t_key]

                # 对齐到相同维度
                min_dim = min(s_emb.size(-1), t_emb.size(-1))
                s_emb = s_emb[:, :min_dim]
                t_emb = t_emb[:, :min_dim]

                cos_sim = F.cosine_similarity(s_emb, t_emb, dim=-1)
                sec_cos = (1.0 - cos_sim).mean()
                sec_mse = F.mse_loss(s_emb, t_emb)

                details[f'{name}_cos'] = sec_cos.item()
                details[f'{name}_mse'] = sec_mse.item()
                total += sec_cos + 0.5 * sec_mse
            else:
                details[f'{name}_cos'] = 0.0
                details[f'{name}_mse'] = 0.0

        return {
            'details': details,
            'total': total / max(len(self.SECTION_NAMES), 1),
        }

    def forward(
        self,
        gen_loss: torch.Tensor,
        student_emb: torch.Tensor,
        teacher_emb: torch.Tensor,
        student_sections: dict = None,
        teacher_sections: dict = None,
        kl_loss: torch.Tensor = None,
    ) -> dict:
        """
        计算完整的 OPD 损失。

        Args:
            gen_loss: 语言模型生成 loss (标量)
            student_emb: Student 编码后融合 embedding, (batch, dim)
            teacher_emb: Teacher 预测目标 embedding, (batch, dim)
            student_sections: 可选, 分部分 student embedding
            teacher_sections: 可选, 分部分 teacher target embedding
            kl_loss: 可选, KL 散度 loss

        Returns:
            dict with 'loss' (标量), 和各子损失
        """
        # L_emb
        emb_result = self.embedding_loss(student_emb, teacher_emb)

        # L_section
        section_result = {'total': torch.tensor(0.0, device=student_emb.device)}
        if student_sections is not None and teacher_sections is not None:
            section_result = self.section_loss(student_sections, teacher_sections)

        # L_kl
        kl_val = kl_loss if kl_loss is not None else torch.tensor(0.0, device=student_emb.device)

        # Total
        total = gen_loss + \
                self.alpha * emb_result['total'] + \
                self.beta * section_result['total'] + \
                self.gamma * kl_val

        return {
            'loss': total,
            'gen_loss': gen_loss,
            'emb_loss': emb_result['total'],
            'emb_cos': emb_result['cos_loss'],
            'emb_mse': emb_result['mse_loss'],
            'section_loss': section_result['total'],
            'kl_loss': kl_val,
        }


class DistillationEngine:
    """
    蒸馏引擎: 协调 Teacher, Student, Latent Pipeline 完成蒸馏步骤。

    单步蒸馏流程:
    1. Teacher 预测 target embedding
    2. Student 生成画像文本
    3. Latent Pipeline 编码生成文本
    4. 计算 OPD 联合损失
    5. 反向传播更新 Student (LoRA 参数)
    """

    def __init__(
        self,
        teacher,
        student,
        latent_pipeline,
        opd_loss: OPDLoss,
        delta_t: float = 0.5,
        device: str = 'cuda:0',
    ):
        self.teacher = teacher
        self.student = student
        self.latent_pipeline = latent_pipeline
        self.opd_loss = opd_loss
        self.delta_t = delta_t
        self.device = device

    def distill_step(
        self,
        current_embeddings: torch.Tensor,
        current_texts: list[str],
        target_texts: list[str] = None,
        input_ids: torch.Tensor = None,
        labels: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ) -> dict:
        """
        执行一步蒸馏。

        Args:
            current_embeddings: 当前用户 embedding, (batch, data_dim)
            current_texts: 当前用户画像文本 (用于构造 prompt)
            target_texts: 目标画像文本 (如果有, 用于 gen_loss)
            input_ids: 已 tokenized 的 input (用于 gen_loss & KL)
            labels: 对应 labels
            attention_mask: attention mask

        Returns:
            dict with all losses
        """
        batch_size = current_embeddings.size(0)

        # 1. Teacher 预测 target embedding
        with torch.no_grad():
            teacher_emb = self.teacher.predict_target_embedding(
                condition=current_embeddings,
                delta_t=self.delta_t,
            )
            teacher_sections = self.teacher.predict_target_section_embeddings(
                condition_emb=current_embeddings,
                delta_t=self.delta_t,
            )

        # 2. Student 生成画像 (并编码)
        # 这里有两种模式:
        #   A) Student 直接生成文本 → 编码为 embedding (推理时)
        #   B) 已有 target 文本，编码后与 Teacher 对齐 (训练时)

        if target_texts is not None:
            # 模式 B: 使用已有 target 文本编码
            from latent.xml_structured import parse_profile_xml, record_to_xml
            import json

            student_sections_list = []
            for text in target_texts:
                try:
                    data = json.loads(text)
                    xml_str = record_to_xml(data)
                    parsed = parse_profile_xml(xml_str)
                    student_sections_list.append(parsed)
                except (json.JSONDecodeError, Exception):
                    student_sections_list.append({
                        'static_information': text[:200],
                        'interest_profile': '',
                        'consumption_profile': '',
                        'demand_targets': '',
                    })

            student_emb = self.latent_pipeline.encode_text_sections(student_sections_list).to(self.device)
        else:
            # 模式 A: Student 生成文本
            generated_texts = []
            for text in current_texts:
                gen = self.student.generate_profile(text, delta_t_desc="未来")
                generated_texts.append(gen)

            # 编码生成文本 (简化: 使用整段文本而非分部分)
            # 实际中需要将生成文本解析为结构化再分段编码
            gen_embs = self.latent_pipeline.embedder.encode_texts(
                generated_texts, show_progress=False
            ).to(self.device)
            # 通过 fusion
            student_emb = self.latent_pipeline.fusion.forward_from_stacked(
                gen_embs.reshape(batch_size, 4, -1)[:, :4, :] if gen_embs.size(1) >= 4
                else gen_embs.unsqueeze(1).expand(-1, 4, -1)
            )

        # 3. Gen loss
        gen_loss = torch.tensor(0.0, device=self.device)
        if input_ids is not None and labels is not None:
            gen_loss = self.student.compute_generation_loss(input_ids, labels, attention_mask)

        # 4. KL loss
        kl_loss = torch.tensor(0.0, device=self.device)
        if input_ids is not None:
            kl_loss = self.student.compute_kl_divergence(input_ids, attention_mask, self.opd_loss.tau)

        # 5. 计算完整 OPD loss
        loss_dict = self.opd_loss(
            gen_loss=gen_loss,
            student_emb=student_emb,
            teacher_emb=teacher_emb,
            kl_loss=kl_loss,
        )

        return loss_dict

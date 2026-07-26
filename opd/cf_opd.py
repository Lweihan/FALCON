"""
cf_opd.py — Candidate-Filtered Online Policy Distillation (CF-OPD)
====================================================================
CF-OPD 核心逻辑: 替代原始 OPD 的单条生成+直接对齐模式。

CF-OPD 流程:
  1. LLM 生成 N 条候选小传文本
  2. 每条 → standardize → XML sections → 分部分 encode → fused embedding
  3. fused embedding → FM Teacher → predict_target_embedding
  4. Reward = w_emb * cos_sim(student_emb, teacher_emb)
           + w_cvr * ranker_cvr_score
  5. 按 reward 筛选: 高分候选 = 正例 (y_w), 低分候选 = 负例 (y_l)
  6. 学习:
     - DPO mode:  L_dpo = -log σ(β * (log π_θ(y_w|x) - log π_ref(y_w|x)
                                    - log π_θ(y_l|x) + log π_ref(y_l|x)))
     - KL mode:   对选中正例做 KL(π_θ_selected ‖ π_ref_selected) 引导向高分
     - combined:  L = L_dpo + α*L_emb + γ*L_kl

与原始 OPD 的区别:
  原始 OPD: Teacher → 单条 Student 生成 → 直接 embedding 对齐
  CF-OPD:  Student 生成 N 条 → ranker 打分 → 筛选 → DPO/KL 学习
           将下游任务信号 (ranker CVR) 反馈给 LLM, 形成闭环
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict

from common.logger import get_logger
from common.config import CFOPDConfig

logger = get_logger(__name__)


class CandidateFilter:
    """
    候选筛选器: 根据 reward 对 N 条候选排序, 选出正例和负例。

    Reward 计算:
      combined mode:  R = w_emb * cos_sim(student, teacher) + w_cvr * cvr_score
      cvr-only mode:  R = cvr_score
    """

    def __init__(self, config: CFOPDConfig):
        self.config = config

    def compute_rewards(
        self,
        student_embs: torch.Tensor,    # (N, dim)
        teacher_embs: torch.Tensor,    # (N, dim) or (1, dim) broadcast
        cvr_scores: torch.Tensor,      # (N,)
    ) -> torch.Tensor:
        """
        计算 N 条候选的组合 reward。

        Returns:
            rewards: (N,) ∈ ℝ
        """
        cfg = self.config

        # Embedding similarity reward
        emb_reward = F.cosine_similarity(student_embs, teacher_embs, dim=-1)  # (N,)

        if cfg.use_combined_reward:
            rewards = cfg.emb_reward_weight * emb_reward + cfg.reward_lambda * cvr_scores
        else:
            rewards = cvr_scores

        return rewards

    def filter_candidates(
        self,
        rewards: torch.Tensor,    # (N,)
        texts: List[str],          # N 条候选文本
        embeddings: torch.Tensor,  # (N, dim)
    ) -> Dict:
        """
        筛选候选: 按 reward 分出正例 (win) 和负例 (lose)。

        策略:
          - 如果所有候选 reward > threshold: 取最高分做正例, 最低分做负例
          - 如果有候选 > threshold 有 < threshold: > threshold 为正, < threshold 为负
          - 保证 min_positive_ratio ~ max_positive_ratio

        Returns:
            {
              'positive_indices': list[int],
              'negative_indices': list[int],
              'pairs': list of (win_idx, lose_idx),  -- DPO 用
              'positive_texts': list[str],
              'negative_texts': list[str],
              'positive_embs': Tensor,
              'negative_embs': Tensor,
              'rewards': Tensor,
            }
        """
        N = len(rewards)
        cfg = self.config
        threshold = cfg.filter_threshold

        # 按 reward 排序 (降序)
        sorted_indices = torch.argsort(rewards, descending=True)

        # 分组: > threshold 为正, <= threshold 为负
        pos_indices = [i for i in range(N) if rewards[i].item() > threshold]
        neg_indices = [i for i in range(N) if rewards[i].item() <= threshold]

        # 修正: 保证正例/负例比例在合理范围
        n_pos = len(pos_indices)
        min_pos = max(int(N * cfg.min_positive_ratio), 1)
        max_pos = int(N * cfg.max_positive_ratio)

        if n_pos < min_pos:
            # 正例不够: 从 neg 中取 reward 最高的补入
            neg_sorted = sorted(neg_indices, key=lambda i: rewards[i].item(), reverse=True)
            need = min_pos - n_pos
            pos_indices.extend(neg_sorted[:need])
            neg_indices = neg_sorted[need:]
        elif n_pos > max_pos:
            # 正例太多: 把 reward 最低的正例降级为负例
            pos_sorted = sorted(pos_indices, key=lambda i: rewards[i].item())
            need = n_pos - max_pos
            neg_indices.extend(pos_sorted[:need])
            pos_indices = pos_sorted[need:]

        # 构建 DPO 对: 每个 positive 与 reward 最低的 negative 配对
        pairs = []
        if pos_indices and neg_indices:
            worst_neg = min(neg_indices, key=lambda i: rewards[i].item())
            for pi in pos_indices:
                pairs.append((pi, worst_neg))

        return {
            'positive_indices': pos_indices,
            'negative_indices': neg_indices,
            'pairs': pairs,
            'positive_texts': [texts[i] for i in pos_indices],
            'negative_texts': [texts[i] for i in neg_indices],
            'positive_embs': embeddings[pos_indices] if pos_indices else torch.empty(0, embeddings.size(1)),
            'negative_embs': embeddings[neg_indices] if neg_indices else torch.empty(0, embeddings.size(1)),
            'rewards': rewards,
        }


class DPOloss(nn.Module):
    """
    DPO (Direct Preference Optimization) 损失。

    L_DPO = -E[log σ(β * (log π_θ(y_w|x) - log π_ref(y_w|x)
                            - log π_θ(y_l|x) + log π_ref(y_l|x)))]

    其中:
      y_w = 正例 (高 reward 候选)
      y_l = 负例 (低 reward 候选)
      π_θ = 当前 Student LLM (带 LoRA)
      π_ref = 参考 Student LLM (初始 LoRA, 冻结)
      β = 隐式奖励缩放参数
    """

    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,   # log π_θ(y_w|x) for chosen
        policy_rejected_logps: torch.Tensor,  # log π_θ(y_l|x) for rejected
        reference_chosen_logps: torch.Tensor,  # log π_ref(y_w|x)
        reference_rejected_logps: torch.Tensor, # log π_ref(y_l|x)
    ) -> torch.Tensor:
        """
        计算 DPO loss。

        Returns:
            标量 loss
        """
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps)

        logits = chosen_rewards - rejected_rewards
        loss = -F.logsigmoid(logits).mean()
        return loss


class CFOPDEngine:
    """
    CF-OPD 引擎: 协调候选生成、筛选、学习全流程。

    单步 CF-OPD 流程:
      1. Student 生成 N 条候选文本
      2. 每条候选 → standardize → encode → fused embedding
      3. FM Teacher 预测 target embedding
      4. Ranker 对每条候选打分 (CVR)
      5. CandidateFilter 筛选正例/负例
      6. DPO/KL/Combined loss → 反向传播更新 Student (LoRA)
    """

    def __init__(
        self,
        student,          # LLMStudent (带 LoRA)
        teacher,          # FMTeacher
        latent_pipeline,  # LatentPipeline (standardize + encode)
        ranker,           # 下游 ranker (CVR prediction)
        cf_config: CFOPDConfig,
        opd_loss=None,    # 原始 OPDLoss (可选, 用于 combined mode)
        device: str = 'cuda:0',
    ):
        self.student = student
        self.teacher = teacher
        self.latent_pipeline = latent_pipeline
        self.ranker = ranker
        self.cf_config = cf_config
        self.opd_loss = opd_loss
        self.device = device

        self.candidate_filter = CandidateFilter(cf_config)
        self.dpo_loss = DPOloss(beta=cf_config.dpo_beta)

        # Reference model (DPO 需要, 初始 LoRA 的 deep copy)
        self._ref_model = None

        logger.info(f"CF-OPD Engine 初始化完成")
        logger.info(f"  num_candidates={cf_config.num_candidates}, "
                     f"filter_threshold={cf_config.filter_threshold}, "
                     f"learning_mode={cf_config.learning_mode}")

    def save_reference(self):
        """保存当前 Student 作为 reference model (DPO 需要)"""
        import copy
        self._ref_model = copy.deepcopy(self.student.model)
        self._ref_model.eval()
        for p in self._ref_model.parameters():
            p.requires_grad = False
        logger.info("CF-OPD: Reference model 已保存")

    def generate_candidates(
        self,
        prompt_text: str,
        n: int = None,
    ) -> List[str]:
        """
        Step 1: LLM 生成 N 条候选文本。

        Args:
            prompt_text: prompt 文本 (当前用户状态)
            n: 候选数量 (默认用 config)

        Returns:
            list of N 条候选文本
        """
        n = n or self.cf_config.num_candidates
        candidates = []
        for _ in range(n):
            text = self.student.generate_profile(
                prompt_text[:500],
                temperature=self.cf_config.candidate_temperature,
            )
            candidates.append(text)
        return candidates

    @torch.no_grad()
    def score_candidates(
        self,
        candidate_texts: List[str],
        condition_emb: torch.Tensor,
    ) -> Dict:
        """
        Step 2-4: 对 N 条候选做 standardize → encode → FM target → ranker 打分。

        Args:
            candidate_texts: N 条候选文本
            condition_emb: 当前 condition embedding (1, dim), FM Teacher 用

        Returns:
            {
              'student_embs': Tensor(N, dim),
              'teacher_embs': Tensor(N, dim) or Tensor(1, dim),
              'cvr_scores': Tensor(N,),
              'sections_list': list of N section dicts,
            }
        """
        N = len(candidate_texts)

        # Step 2: 每条候选 → standardize → sections
        sections_list = []
        for text in candidate_texts:
            try:
                from feature_extract.standard import extract_slots, slots_to_xml
                from latent.xml_structured import parse_profile_xml
                slots = extract_slots(text)
                xml_str = slots_to_xml(slots)
                parsed = parse_profile_xml(xml_str)
                sections_list.append(parsed)
            except Exception:
                sections_list.append({
                    'static_information': text[:200],
                    'interest_profile': '',
                    'consumption_profile': '',
                    'demand_targets': '',
                })

        # Step 3: 编码 → fused embedding
        student_embs = self.latent_pipeline.encode_text_sections(sections_list).to(self.device)
        # (N, dim)

        # Step 4a: FM Teacher 预测 target embedding
        teacher_emb = self.teacher.predict_target_embedding(
            condition_emb, delta_t=0.5,
        )  # (1, dim) — 广播到 N 条候选

        # Step 4b: Ranker 打分
        cvr_scores = self.ranker.predict_proba(student_embs).squeeze(-1)  # (N,)

        return {
            'student_embs': student_embs,
            'teacher_embs': teacher_emb.expand(N, -1),  # 广播
            'cvr_scores': cvr_scores,
            'sections_list': sections_list,
        }

    def compute_dpo_loss(
        self,
        prompt_input_ids: torch.Tensor,     # (1, seq_len)
        prompt_attention_mask: torch.Tensor, # (1, seq_len)
        chosen_input_ids: torch.Tensor,      # (n_pos, seq_len)
        chosen_attention_mask: torch.Tensor,
        rejected_input_ids: torch.Tensor,     # (n_neg, seq_len)
        rejected_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Step 6a: 计算 DPO loss。

        需要 π_θ 和 π_ref 对 chosen / rejected 的 log probs。
        """
        if self._ref_model is None:
            logger.warning("CF-OPD: reference model 未保存, DPO loss 退化为 SFT")
            # 退化为 SFT: 只对 chosen 做 cross-entropy
            outputs = self.student.model(
                input_ids=chosen_input_ids,
                attention_mask=chosen_attention_mask,
                labels=chosen_input_ids,
            )
            return outputs.loss

        # Current policy log probs
        policy_chosen_logps = self._get_sequence_log_probs(
            self.student.model, chosen_input_ids, chosen_attention_mask
        )
        policy_rejected_logps = self._get_sequence_log_probs(
            self.student.model, rejected_input_ids, rejected_attention_mask
        )

        # Reference log probs (no grad)
        with torch.no_grad():
            ref_chosen_logps = self._get_sequence_log_probs(
                self._ref_model, chosen_input_ids, chosen_attention_mask
            )
            ref_rejected_logps = self._get_sequence_log_probs(
                self._ref_model, rejected_input_ids, rejected_attention_mask
            )

        return self.dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
        )

    def compute_kl_loss(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Step 6b: 对选中正例做 KL(π_θ ‖ π_ref), 引导 LLM 向高分方向。
        """
        return self.student.compute_kl_divergence(
            chosen_input_ids, chosen_attention_mask, self.cf_config.kl_tau
        )

    def _get_sequence_log_probs(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        获取序列的平均 log probability。

        Returns:
            (batch,) 每个 sample 的平均 log prob
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (batch, seq_len, vocab_size)

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Log softmax
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather log probs for correct tokens
        per_token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

        # Mask padding
        mask = (shift_labels != -100).float()
        if attention_mask is not None:
            mask = mask * attention_mask[:, 1:].float()

        avg_log_probs = (per_token_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        return avg_log_probs

    def cf_opd_step(
        self,
        current_text: str,
        condition_emb: torch.Tensor,
        prompt_input_ids: torch.Tensor = None,
        prompt_attention_mask: torch.Tensor = None,
    ) -> Dict:
        """
        执行一步完整的 CF-OPD。

        Args:
            current_text: 当前用户文本 (用于构造 prompt)
            condition_emb: 当前 condition embedding (1, dim)
            prompt_input_ids: tokenized prompt (可选, 若已有)
            prompt_attention_mask: prompt attention mask

        Returns:
            dict with loss, rewards, filter_result, etc.
        """
        cfg = self.cf_config

        # Step 1: 生成 N 条候选
        candidate_texts = self.generate_candidates(current_text, n=cfg.num_candidates)

        # Step 2-4: 编码 + 打分
        score_result = self.score_candidates(candidate_texts, condition_emb)

        # Step 5: 筛选
        filter_result = self.candidate_filter.filter_candidates(
            rewards=score_result['rewards'],
            texts=candidate_texts,
            embeddings=score_result['student_embs'],
        )

        # 如果没有正例或没有负例, 跳过本步
        if not filter_result['positive_indices'] or not filter_result['negative_indices']:
            logger.warning("CF-OPD: 候选筛选后无正例或无负例, 跳过本步")
            return {
                'status': 'skipped',
                'reason': 'no_valid_pairs',
                'n_pos': len(filter_result['positive_indices']),
                'n_neg': len(filter_result['negative_indices']),
                'rewards': score_result['rewards'],
            }

        # Step 6: 计算 loss
        loss = torch.tensor(0.0, device=self.device)

        if cfg.learning_mode == 'dpo':
            # Tokenize chosen / rejected
            chosen_texts = filter_result['positive_texts']
            rejected_texts = filter_result['negative_texts']

            # 用 student.tokenizer 编码
            chosen_enc = self.student.tokenizer(
                chosen_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=self.student.config.max_new_tokens,
            ).to(self.device)
            rejected_enc = self.student.tokenizer(
                rejected_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=self.student.config.max_new_tokens,
            ).to(self.device)

            loss = self.compute_dpo_loss(
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask,
                chosen_input_ids=chosen_enc['input_ids'],
                chosen_attention_mask=chosen_enc['attention_mask'],
                rejected_input_ids=rejected_enc['input_ids'],
                rejected_attention_mask=rejected_enc['attention_mask'],
            )

        elif cfg.learning_mode == 'kl':
            chosen_enc = self.student.tokenizer(
                filter_result['positive_texts'], return_tensors="pt", padding=True,
                truncation=True, max_length=self.student.config.max_new_tokens,
            ).to(self.device)
            loss = self.compute_kl_loss(
                chosen_enc['input_ids'], chosen_enc['attention_mask'],
            )

        elif cfg.learning_mode == 'combined':
            # DPO + L_emb + L_kl
            # 1) DPO 部分
            chosen_enc = self.student.tokenizer(
                filter_result['positive_texts'], return_tensors="pt", padding=True,
                truncation=True, max_length=self.student.config.max_new_tokens,
            ).to(self.device)
            rejected_enc = self.student.tokenizer(
                filter_result['negative_texts'], return_tensors="pt", padding=True,
                truncation=True, max_length=self.student.config.max_new_tokens,
            ).to(self.device)

            dpo = self.compute_dpo_loss(
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask,
                chosen_input_ids=chosen_enc['input_ids'],
                chosen_attention_mask=chosen_enc['attention_mask'],
                rejected_input_ids=rejected_enc['input_ids'],
                rejected_attention_mask=rejected_enc['attention_mask'],
            )

            # 2) L_emb 部分: 正例 embedding 与 Teacher 对齐
            pos_embs = filter_result['positive_embs'].to(self.device)
            teacher_embs = score_result['teacher_embs'][:len(pos_embs)]
            if self.opd_loss is not None:
                emb_result = self.opd_loss.embedding_loss(pos_embs, teacher_embs)
                l_emb = emb_result['total']
            else:
                l_emb = F.mse_loss(pos_embs, teacher_embs)

            # 3) L_kl 部分
            l_kl = self.compute_kl_loss(
                chosen_enc['input_ids'], chosen_enc['attention_mask'],
            )

            loss = dpo + cfg.distill_alpha * l_emb + cfg.distill_gamma * l_kl

        return {
            'status': 'success',
            'loss': loss,
            'n_candidates': len(candidate_texts),
            'n_pos': len(filter_result['positive_indices']),
            'n_neg': len(filter_result['negative_indices']),
            'rewards': score_result['rewards'],
            'cvr_scores': score_result['cvr_scores'],
            'filter_result': filter_result,
        }

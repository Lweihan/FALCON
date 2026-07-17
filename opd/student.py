"""
student.py — LLM Student 封装
================================
将大模型 (如 Qwen3) 作为 Student，支持 LoRA 高效微调。

核心功能:
  - generate_profile(): 根据当前用户信息生成未来画像文本
  - compute_generation_loss(): 计算语言模型生成损失
  - get_log_probs(): 获取生成概率 (用于 KL 散度计算)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from typing import Optional

from common.logger import get_logger
from common.config import OPDConfig

logger = get_logger(__name__)


# ────────────────────────────────────────────────────────────
# Prompt 模板
# ────────────────────────────────────────────────────────────

PROFILE_GENERATION_PROMPT = """你是一个用户画像专家。根据以下用户当前状态信息，预测该用户在未来的画像特征。

## 用户当前状态
{current_profile}

## 要求
请基于当前状态，预测该用户在未来 {delta_t_desc} 的画像，严格按照以下 JSON 格式输出:
{{
  "basic_profile": {{...}},
  "local_scene": {{...}},
  "other_scene": {{...}},
  "interest_profile": [...],
  "consumption_profile": {{...}},
  "demand_targets": [...]
}}

## 预测的未来画像
"""

# 用于 OPD 训练的简短 prompt (减少 token 数)
TRAINING_PROMPT = """基于用户当前状态，预测未来画像:

当前状态:
{current_summary}

未来画像:"""


class LLMStudent:
    """
    LLM Student 封装。

    支持:
    - LoRA 高效微调
    - 生成用户画像
    - 计算生成 loss & KL 散度
    """

    def __init__(
        self,
        config: OPDConfig,
        device: str = 'cuda:0',
    ):
        self.config = config
        self.device = device

        logger.info(f"加载 Student LLM: {config.student_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.student_model_path, padding_side='left', trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(
            config.student_model_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )

        # 应用 LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # 保存 reference model (用于 KL 散度)
        self._ref_model = None

    def generate_profile(
        self,
        current_profile_text: str,
        delta_t_desc: str = "一段时间后",
        max_new_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
    ) -> str:
        """
        生成用户未来画像文本。

        Args:
            current_profile_text: 当前用户画像文本
            delta_t_desc: 时间描述
            max_new_tokens: 最大生成长度
            temperature: 采样温度
            top_p: nucleus 采样阈值

        Returns:
            生成的画像文本
        """
        self.model.eval()

        prompt = TRAINING_PROMPT.format(current_summary=current_profile_text[:500])

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.config.max_new_tokens,
                temperature=temperature or self.config.temperature,
                top_p=top_p or self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # 解码，去掉 prompt 部分
        generated = outputs[0][inputs['input_ids'].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text

    def compute_generation_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        计算语言模型生成 loss (CrossEntropy)。

        Args:
            input_ids: (batch, seq_len)
            labels: (batch, seq_len) 与 input_ids 对齐，-100 表示忽略
            attention_mask: (batch, seq_len)

        Returns:
            标量 loss
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.loss

    def get_log_probs(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        获取每个 token 的 log probability (用于 KL 散度计算)。

        Returns:
            (batch,) 每个 sample 的平均 log prob
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (batch, seq_len, vocab_size)

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Log softmax
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather log probs for correct tokens
        per_token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

        # Mask out -100 labels
        mask = (shift_labels != -100).float()
        avg_log_probs = (per_token_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)

        return avg_log_probs

    def compute_kl_divergence(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        tau: float = None,
    ) -> torch.Tensor:
        """
        计算 Student 与 Reference Model 之间的 KL 散度 (防止遗忘)。

        KL(π_θ || π_ref) = Σ π_θ * (log π_θ - log π_ref)

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            tau: 温度缩放参数

        Returns:
            标量 KL 散度
        """
        tau = tau or self.config.distill_tau

        # Student logits
        s_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        s_logits = s_outputs.logits[:, :-1, :] / tau

        # Reference logits (no grad)
        if self._ref_model is None:
            # 首次调用时保存 reference
            return torch.tensor(0.0, device=input_ids.device)

        with torch.no_grad():
            r_outputs = self._ref_model(input_ids=input_ids, attention_mask=attention_mask)
            r_logits = r_outputs.logits[:, :-1, :] / tau

        # KL 散度
        s_log_probs = F.log_softmax(s_logits, dim=-1)
        r_log_probs = F.log_softmax(r_logits, dim=-1)
        r_probs = F.softmax(r_logits, dim=-1)

        kl = (r_probs * (r_log_probs - s_log_probs)).sum(dim=-1)
        # 按 sequence 平均
        kl = kl.mean()
        return kl * (tau ** 2)  # 温度缩放修正

    def save_reference(self):
        """保存当前模型作为 reference (用于 KL 散度)"""
        import copy
        self._ref_model = copy.deepcopy(self.model)
        self._ref_model.eval()
        for p in self._ref_model.parameters():
            p.requires_grad = False
        logger.info("Reference model 已保存 (用于 KL 散度计算)")

    def save_lora(self, path: str):
        """保存 LoRA 权重"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"LoRA 权重保存至: {path}")

    def load_lora(self, path: str):
        """加载 LoRA 权重"""
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, path)
        logger.info(f"LoRA 权重加载: {path}")

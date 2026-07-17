"""
serving.py — 在线服务
========================
FALCON 的端到端在线推理管线。
"""

import torch
import json
from typing import Optional

from common.logger import get_logger
from common.config import FALCONConfig

logger = get_logger(__name__)


class FALCONServing:
    """
    FALCON 在线服务管线。

    端到端流程:
    1. 接收用户当前画像 (JSON)
    2. Latent Pipeline 编码 → condition embedding
    3. FM Teacher 预测 → target embedding
    4. LLM Student 生成 → 未来画像文本
    5. (可选) 验证: 编码生成文本 → 与 target 对比
    6. 返回未来画像 + embedding

    用于广告场景:
    - 根据预测的用户未来特征，优化广告投放策略
    - 预测用户兴趣演变，提前布局内容
    - 识别用户需求目标变化，调整推荐
    """

    def __init__(
        self,
        config: FALCONConfig,
        fm_checkpoint: str = None,
        lora_path: str = None,
        fusion_checkpoint: str = None,
    ):
        self.config = config
        device = config.resolve_device()

        # Latent Pipeline
        from latent.pipeline import LatentPipeline
        self.pipeline = LatentPipeline(config.latent, device=device)
        if fusion_checkpoint:
            self.pipeline.load_fusion_state(fusion_checkpoint)

        # Teacher
        from opd.teacher import FMTeacher
        self.teacher = FMTeacher(
            fm_config=config.fm,
            checkpoint_path=fm_checkpoint,
            device=device,
        )

        # Student
        from opd.student import LLMStudent
        self.student = LLMStudent(config.opd, device=device)
        if lora_path:
            self.student.load_lora(lora_path)

        self.device = device

    def predict(
        self,
        user_current_profile: dict,
        delta_t: float = 0.5,
        use_student: bool = True,
    ) -> dict:
        """
        预测用户未来画像。

        Args:
            user_current_profile: 当前用户画像 dict (含 records 列表)
            delta_t: 时间偏移 [0, 1]
            use_student: 是否使用 Student 生成文本 (True) 或仅用 Teacher embedding (False)

        Returns:
            {
                'condition_emb': Tensor,
                'teacher_target_emb': Tensor,
                'generated_text': str (if use_student),
                'student_emb': Tensor (if use_student),
                'cosine_sim': float,
            }
        """
        # 1. 编码当前状态
        user_result = self.pipeline.encode_user(user_current_profile)
        embeddings = user_result['embeddings']

        if embeddings.size(0) == 0:
            return {'error': 'No valid timesteps in user data'}

        # 使用最近的 embedding 作为 condition
        condition = embeddings[-1].unsqueeze(0).to(self.device)

        # 2. Teacher 预测
        teacher_emb = self.teacher.predict_target_embedding(condition, delta_t)

        # 3. Student 生成 (可选)
        result = {
            'condition_emb': condition.squeeze(0).cpu(),
            'teacher_target_emb': teacher_emb.squeeze(0).cpu(),
        }

        if use_student:
            # 构造 prompt
            from latent.xml_structured import record_to_xml
            records = user_current_profile.get('records', [])
            if records:
                current_xml = record_to_xml(records[-1])
                current_text = current_xml[:500]
            else:
                current_text = ""

            delta_desc = self._delta_t_to_text(delta_t)
            generated = self.student.generate_profile(current_text, delta_desc)

            # 编码生成文本
            import re
            try:
                # 尝试解析为 JSON
                gen_json = json.loads(generated)
                gen_xml = record_to_xml(gen_json)
            except (json.JSONDecodeError, Exception):
                gen_xml = generated  # 原始文本

            from latent.xml_structured import parse_profile_xml
            gen_sections = [parse_profile_xml(gen_xml)]
            student_emb = self.pipeline.encode_text_sections(gen_sections).to(self.device)

            result['generated_text'] = generated
            result['student_emb'] = student_emb.squeeze(0).cpu()

            # 计算相似度
            import torch.nn.functional as F
            cos_sim = F.cosine_similarity(
                student_emb.squeeze(0).cpu().unsqueeze(0),
                teacher_emb.squeeze(0).cpu().unsqueeze(0),
            ).item()
            result['cosine_sim'] = cos_sim

        return result

    def batch_predict(
        self,
        users: list[dict],
        delta_t: float = 0.5,
    ) -> list[dict]:
        """批量预测"""
        return [self.predict(u, delta_t, use_student=False) for u in users]

    @staticmethod
    def _delta_t_to_text(delta_t: float) -> str:
        if delta_t <= 0.2:
            return "数小时后"
        elif delta_t <= 0.4:
            return "1天后"
        elif delta_t <= 0.6:
            return "3天后"
        elif delta_t <= 0.8:
            return "1周后"
        else:
            return "1个月后"

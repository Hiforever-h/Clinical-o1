"""Base、SFT、GRPO 与 DAPO 共用的英文医疗选择题评测框架。"""

from medical_grpo.evaluation.config import EvaluationConfig, load_evaluation_config

__all__ = ["EvaluationConfig", "load_evaluation_config"]

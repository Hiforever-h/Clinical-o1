"""M1 英文主线使用的规范数据结构与逐条校验规则。

本模块只保留当前已经验收的三类数据：SFT、RL prompt 和选择题评测。
训练器、Reward、答案解析器等后续阶段能力不在这里提前实现。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


VALID_SPLITS = {"train", "dev", "test"}
MCQ_REQUIRED_FIELDS = {"id", "source", "question", "options", "answer", "split"}
SFT_REQUIRED_FIELDS = {"id", "source", "question", "reasoning", "response", "messages", "split"}
RL_REQUIRED_FIELDS = {"id", "source", "prompt", "ground_truth_answer", "split"}


@dataclass
class MedicalMCQSample:
    """MedQA、MedMCQA、PubMedQA 共用的选择题格式。"""

    id: str
    source: str
    question: str
    options: dict[str, str]
    answer: str
    split: str
    answer_text: str | None = None
    explanation: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSONL 写入器可直接序列化的字典。"""

        return asdict(self)


@dataclass
class MedicalSFTSample:
    """Huatuo English SFT 的问题、推理过程和最终回答。"""

    id: str
    source: str
    question: str
    reasoning: str
    response: str
    messages: list[dict[str, str]]
    split: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSONL 写入器可直接序列化的字典。"""

        return asdict(self)


@dataclass
class MedicalRLPromptSample:
    """Huatuo verifiable problem 的 prompt 与可验证标准答案。"""

    id: str
    source: str
    prompt: str
    ground_truth_answer: str
    split: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSONL 写入器可直接序列化的字典。"""

        return asdict(self)


def _text(value: Any) -> str:
    """将外部字段统一转为去除首尾空白的字符串。"""

    return str(value).strip()


def _optional_text(value: Any) -> str | None:
    """保留缺失值为 None，其余值统一清理为字符串。"""

    return None if value is None else _text(value)


def _meta(value: Any) -> dict[str, Any]:
    """保留来源 revision、原始 split 和行号等可追溯信息。"""

    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    return {"raw_meta": value}


def _options(value: Any) -> dict[str, str]:
    """把选择题选项键统一为大写并清理选项文本。"""

    if not isinstance(value, Mapping):
        return {}
    return {
        _text(key).upper(): _text(option)
        for key, option in value.items()
        if _text(key)
    }


def build_sft_messages(question: str, reasoning: str, response: str) -> list[dict[str, str]]:
    """按 HuatuoGPT-o1 格式构造可直接套用 chat template 的单轮对话。"""

    assistant = f"## Thinking\n{reasoning}\n\n## Final Response\n{response}"
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": assistant},
    ]


def _messages(value: Any) -> list[dict[str, str]]:
    """从外部数据中提取 role/content，丢弃未知附加字段。"""

    if not isinstance(value, list):
        return []
    messages: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, Mapping):
            messages.append({"role": _text(item.get("role", "")), "content": _text(item.get("content", ""))})
    return messages


def _common_errors(
    record: Mapping[str, Any], required: set[str], prompt_field: str
) -> list[str]:
    """校验三类数据共有的 ID、来源、文本、split 与 provenance。"""

    errors: list[str] = []
    missing = sorted(required - set(record))
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")
    if not _text(record.get("id", "")):
        errors.append("id must be a non-empty string")
    if not _text(record.get("source", "")):
        errors.append("source must be a non-empty string")
    if not _text(record.get(prompt_field, "")):
        errors.append(f"{prompt_field} must be a non-empty string")
    if _text(record.get("split", "")).lower() not in VALID_SPLITS:
        errors.append(f"split must be one of {sorted(VALID_SPLITS)}")
    if record.get("meta") is not None and not isinstance(record.get("meta"), Mapping):
        errors.append("meta must be an object when provided")
    return errors


def normalize_mcq_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """把不同来源的选择题记录规范化为统一字段和值类型。"""

    options = _options(record.get("options", {}))
    answer = _text(record.get("answer", "")).upper()
    answer_text = _optional_text(record.get("answer_text"))
    if not answer_text and answer in options:
        answer_text = options[answer]
    return {
        "id": _text(record.get("id", "")),
        "source": _text(record.get("source", "")),
        "question": _text(record.get("question", "")),
        "options": options,
        "answer": answer,
        "split": _text(record.get("split", "")).lower(),
        "answer_text": answer_text,
        "explanation": _optional_text(record.get("explanation")),
        "meta": _meta(record.get("meta")),
    }


def validate_mcq_record(record: Mapping[str, Any]) -> list[str]:
    """返回选择题的全部 schema 错误，不在首个错误处提前退出。"""

    errors = _common_errors(record, MCQ_REQUIRED_FIELDS, "question")
    options = _options(record.get("options", {}))
    if len(options) < 2:
        errors.append("options must contain at least two choices")
    empty = [key for key, value in options.items() if not value]
    if empty:
        errors.append(f"option values must be non-empty: {', '.join(empty)}")
    answer = _text(record.get("answer", "")).upper()
    if not answer:
        errors.append("answer must be a non-empty option key")
    elif options and answer not in options:
        errors.append(f"answer '{answer}' is not in options: {', '.join(sorted(options))}")
    return errors


def to_mcq_sample(record: Mapping[str, Any]) -> MedicalMCQSample:
    """规范化并校验选择题，成功后构造强类型样本。"""

    normalized = normalize_mcq_record(record)
    errors = validate_mcq_record(normalized)
    if errors:
        raise ValueError(f"invalid MCQ sample {normalized.get('id') or '<unknown>'}: {'; '.join(errors)}")
    return MedicalMCQSample(**normalized)


def normalize_sft_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """规范化 SFT 字段，并在缺少 messages 时按 Huatuo 格式补建。"""

    question = _text(record.get("question", ""))
    reasoning = _text(record.get("reasoning", ""))
    response = _text(record.get("response", ""))
    messages = _messages(record.get("messages"))
    if not messages and question and reasoning and response:
        messages = build_sft_messages(question, reasoning, response)
    return {
        "id": _text(record.get("id", "")),
        "source": _text(record.get("source", "")),
        "question": question,
        "reasoning": reasoning,
        "response": response,
        "messages": messages,
        "split": _text(record.get("split", "")).lower(),
        "meta": _meta(record.get("meta")),
    }


def validate_sft_record(record: Mapping[str, Any]) -> list[str]:
    """检查 SFT 推理、回答和 user/assistant 对话结构。"""

    errors = _common_errors(record, SFT_REQUIRED_FIELDS, "question")
    if not _text(record.get("reasoning", "")):
        errors.append("reasoning must be a non-empty string")
    if not _text(record.get("response", "")):
        errors.append("response must be a non-empty string")
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        errors.append("messages must contain at least user and assistant messages")
    else:
        roles = [item.get("role") for item in messages if isinstance(item, Mapping)]
        if "user" not in roles or "assistant" not in roles:
            errors.append("messages must contain user and assistant roles")
        for index, item in enumerate(messages):
            if not isinstance(item, Mapping) or not _text(item.get("content", "")):
                errors.append(f"messages[{index}] must contain non-empty content")
    return errors


def to_sft_sample(record: Mapping[str, Any]) -> MedicalSFTSample:
    """规范化并校验 SFT 记录，成功后构造强类型样本。"""

    normalized = normalize_sft_record(record)
    errors = validate_sft_record(normalized)
    if errors:
        raise ValueError(f"invalid SFT sample {normalized.get('id') or '<unknown>'}: {'; '.join(errors)}")
    return MedicalSFTSample(**normalized)


def normalize_rl_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """规范化 RL prompt、可验证答案、split 与 provenance。"""

    return {
        "id": _text(record.get("id", "")),
        "source": _text(record.get("source", "")),
        "prompt": _text(record.get("prompt", "")),
        "ground_truth_answer": _text(record.get("ground_truth_answer", "")),
        "split": _text(record.get("split", "")).lower(),
        "meta": _meta(record.get("meta")),
    }


def validate_rl_record(record: Mapping[str, Any]) -> list[str]:
    """检查 RL prompt 与 ground-truth answer 是否完整。"""

    errors = _common_errors(record, RL_REQUIRED_FIELDS, "prompt")
    if not _text(record.get("ground_truth_answer", "")):
        errors.append("ground_truth_answer must be a non-empty string")
    return errors


def to_rl_sample(record: Mapping[str, Any]) -> MedicalRLPromptSample:
    """规范化并校验 RL 记录，成功后构造强类型样本。"""

    normalized = normalize_rl_record(record)
    errors = validate_rl_record(normalized)
    if errors:
        raise ValueError(f"invalid RL sample {normalized.get('id') or '<unknown>'}: {'; '.join(errors)}")
    return MedicalRLPromptSample(**normalized)

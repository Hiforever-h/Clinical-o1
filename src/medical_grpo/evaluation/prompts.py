"""冻结的 direct/cot 选择题 Prompt 模板与合同版本。"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Mapping


PROMPT_VERSION = "mcq_prompt_v1"

DIRECT_INSTRUCTION = """You are answering a medical multiple-choice question.
Select the single best option.
Return only one line in this exact format:
Answer: <OPTION_LETTER>"""

COT_INSTRUCTION = """You are solving a medical multiple-choice question.
Analyze the case step by step and select the single best option.
Use exactly this response structure:

## Thinking
Your step-by-step reasoning.

## Final Response
Answer: <OPTION_LETTER>"""


def format_options(options: Mapping[str, str]) -> str:
    """按数据中的稳定插入顺序渲染选项，不对答案键重新排序。"""

    return "\n".join(f"{key}. {value}" for key, value in options.items())


def build_messages(record: Mapping[str, Any], protocol: str) -> list[dict[str, str]]:
    """把 canonical MCQ 转换为 Qwen chat template 所需的单轮 user 消息。"""

    if protocol == "direct":
        instruction = DIRECT_INSTRUCTION
    elif protocol == "cot":
        instruction = COT_INSTRUCTION
    else:
        raise ValueError(f"未知 protocol={protocol!r}")
    content = (
        f"{instruction}\n\n"
        f"Question:\n{record['question']}\n\n"
        f"Options:\n{format_options(record['options'])}"
    )
    return [{"role": "user", "content": content}]


def prompt_contract() -> dict[str, str]:
    """返回进入公平评测 contract hash 的模板文本和版本。"""

    return {
        "version": PROMPT_VERSION,
        "direct_instruction": DIRECT_INSTRUCTION,
        "cot_instruction": COT_INSTRUCTION,
    }


def prompt_contract_sha256() -> str:
    """计算稳定模板哈希，防止不同模型使用不同 Prompt。"""

    encoded = json.dumps(prompt_contract(), ensure_ascii=False, sort_keys=True).encode("utf-8")
    return sha256(encoded).hexdigest()

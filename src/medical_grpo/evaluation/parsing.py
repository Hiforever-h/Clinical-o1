"""医疗选择题严格答案解析、格式检查和对抗性边界处理。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import unicodedata
from typing import Mapping


PARSER_VERSION = "strict_mcq_parser_v1"


@dataclass(frozen=True)
class ParseResult:
    """单条模型输出的答案、解析方法和格式状态。"""

    parsed_answer: str | None
    parse_method: str | None
    parse_success: bool
    ambiguous: bool
    format_valid: bool

    def to_dict(self) -> dict[str, object]:
        """转换为逐题 JSONL 可直接展开的字典。"""

        return asdict(self)


def _normalize_text(text: str) -> str:
    """用于选项文本匹配的 NFKC、小写和空白归一化。"""

    normalized = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", normalized).strip(" \t\r\n:.-*`")


def _last_final_section(text: str) -> tuple[str, bool]:
    """返回最后一个 Final Response 后的文本及标题顺序是否正确。"""

    thinking_index = text.find("## Thinking")
    final_index = text.rfind("## Final Response")
    valid_order = thinking_index >= 0 and final_index > thinking_index
    return (text[final_index + len("## Final Response") :] if final_index >= 0 else text, valid_order)


def _last_marker_answer(section: str, allowed: set[str]) -> tuple[str | None, bool]:
    """提取最后一个 Answer/Final Answer 标记，并检测同段多答案歧义。"""

    pattern = re.compile(
        r"(?:final\s+answer|answer)\s*(?:is\s*)?[:=\-]?\s*[\(\[]?([A-Za-z])[\)\]]?(?=\s|$|[.,;])",
        flags=re.IGNORECASE,
    )
    answers = [match.upper() for match in pattern.findall(section) if match.upper() in allowed]
    return (answers[-1] if answers else None, len(set(answers)) > 1)


def parse_mcq_answer(text: str, options: Mapping[str, str], protocol: str) -> ParseResult:
    """按冻结优先级解析答案；无法唯一判断时返回未解析而不是猜测。"""

    raw = str(text).strip()
    allowed = {str(key).upper() for key in options}
    final_section, cot_headers_valid = _last_final_section(raw)

    # 去除 Markdown 强调符后再匹配，使 Answer: **B** 与普通答案等价。
    marker_section = re.sub(r"[*_`]", "", final_section)
    answer, ambiguous = _last_marker_answer(marker_section, allowed)
    if answer is not None:
        format_valid = cot_headers_valid if protocol == "cot" else True
        return ParseResult(answer, "final_answer_marker", True, ambiguous, format_valid)

    # 支持常见的 LaTeX boxed 输出，但只接受合法选项键。
    boxed = [value.upper() for value in re.findall(r"\\boxed\s*\{\s*([A-Za-z])\s*\}", final_section)]
    boxed = [value for value in boxed if value in allowed]
    if boxed:
        return ParseResult(
            boxed[-1],
            "boxed_letter",
            True,
            len(set(boxed)) > 1,
            cot_headers_valid if protocol == "cot" else True,
        )

    # 输出只有一个字母时可以安全解析；不会在长 CoT 中搜索任意孤立字母。
    single = re.fullmatch(r"\s*[\(\[]?([A-Za-z])[\)\].]?\s*", marker_section)
    if single and single.group(1).upper() in allowed:
        return ParseResult(
            single.group(1).upper(),
            "single_letter",
            True,
            False,
            cot_headers_valid if protocol == "cot" else True,
        )

    normalized_section = _normalize_text(final_section)
    # PubMedQA 常输出语义答案；只有整段精确等于 yes/no/maybe 才映射。
    semantic = {"yes": "A", "no": "B", "maybe": "C"}
    if normalized_section in semantic and semantic[normalized_section] in allowed:
        return ParseResult(
            semantic[normalized_section],
            "pubmed_semantic",
            True,
            False,
            cot_headers_valid if protocol == "cot" else True,
        )

    # 选项文本只在 Final Response 整段唯一匹配时启用，避免从推理正文误判。
    text_matches = [
        key.upper()
        for key, option_text in options.items()
        if normalized_section == _normalize_text(str(option_text))
    ]
    if len(text_matches) == 1:
        return ParseResult(
            text_matches[0],
            "exact_option_text",
            True,
            False,
            cot_headers_valid if protocol == "cot" else True,
        )
    return ParseResult(None, None, False, len(text_matches) > 1, False)

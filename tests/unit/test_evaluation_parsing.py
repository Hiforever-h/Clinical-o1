"""严格 MCQ 答案解析器的正常和对抗性单元测试。"""

from __future__ import annotations

from medical_grpo.evaluation.parsing import parse_mcq_answer


OPTIONS = {"A": "aspirin", "B": "heparin", "C": "warfarin", "D": "insulin"}


def test_parser_prefers_final_response_over_reasoning_letters() -> None:
    """推理中出现的错误选项不能覆盖 Final Response 的明确答案。"""

    text = "## Thinking\nA is tempting, but it is wrong.\n## Final Response\nAnswer: B"
    result = parse_mcq_answer(text, OPTIONS, "cot")

    assert result.parsed_answer == "B"
    assert result.parse_success is True
    assert result.format_valid is True


def test_parser_supports_boxed_single_letter_and_exact_option_text() -> None:
    """安全的常见输出变体应解析，不能要求模型只使用一种标点。"""

    assert parse_mcq_answer(r"\boxed{C}", OPTIONS, "direct").parsed_answer == "C"
    assert parse_mcq_answer("(d)", OPTIONS, "direct").parsed_answer == "D"
    assert parse_mcq_answer("Answer: **B**", OPTIONS, "direct").parsed_answer == "B"
    assert parse_mcq_answer("heparin", OPTIONS, "direct").parsed_answer == "B"


def test_parser_does_not_guess_from_unanchored_cot() -> None:
    """没有答案标记的长文本即使包含选项字母也应判为未解析。"""

    result = parse_mcq_answer("A may work, while B may also work.", OPTIONS, "cot")

    assert result.parsed_answer is None
    assert result.parse_success is False


def test_pubmed_semantic_answer_maps_to_option_key() -> None:
    """PubMedQA 的 yes/no/maybe 整段语义答案应映射到 A/B/C。"""

    options = {"A": "yes", "B": "no", "C": "maybe"}
    assert parse_mcq_answer("no", options, "direct").parsed_answer == "B"

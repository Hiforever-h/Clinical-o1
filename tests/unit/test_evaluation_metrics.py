"""评测准确率、解析率和 Wilson 区间的单元测试。"""

from __future__ import annotations

from medical_grpo.evaluation.metrics import summarize_predictions, wilson_interval


def test_metrics_keep_parse_failures_in_accuracy_denominator() -> None:
    """未解析样本在总体 Accuracy 中必须算错，不能被静默排除。"""

    rows = [
        {
            "correct": True,
            "parse_success": True,
            "format_valid": True,
            "hit_max_new_tokens": False,
            "completion_tokens": 4,
            "prompt_tokens": 20,
            "raw_completion": "Answer: A",
            "estimated_sample_seconds": 1.0,
            "gold_answer": "A",
            "parsed_answer": "A",
            "parse_method": "final_answer_marker",
        },
        {
            "correct": False,
            "parse_success": False,
            "format_valid": False,
            "hit_max_new_tokens": True,
            "completion_tokens": 32,
            "prompt_tokens": 30,
            "raw_completion": "unknown",
            "estimated_sample_seconds": 1.0,
            "gold_answer": "B",
            "parsed_answer": None,
            "parse_method": None,
        },
    ]

    summary = summarize_predictions(rows)

    assert summary["accuracy"] == 0.5
    assert summary["parse_success"] == 0.5
    assert summary["accuracy_on_parsed"] == 1.0
    assert summary["hit_max_new_tokens_rate"] == 0.5


def test_wilson_interval_is_bounded() -> None:
    """Wilson 区间必须落在 0 到 1，且覆盖样本比例。"""

    lower, upper = wilson_interval(60, 100)

    assert 0.0 <= lower <= 0.6 <= upper <= 1.0

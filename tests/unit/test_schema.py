from __future__ import annotations

import pytest

from medical_grpo.data.schema import to_mcq_sample, to_rl_sample, to_sft_sample


def test_sft_schema_builds_huatuo_chat_messages() -> None:
    sample = to_sft_sample(
        {
            "id": "sft-1",
            "source": "huatuo",
            "question": "What is the diagnosis?",
            "reasoning": "The findings support pneumonia.",
            "response": "Pneumonia.",
            "split": "train",
        }
    )

    assert [message["role"] for message in sample.messages] == ["user", "assistant"]
    assert sample.messages[-1]["content"].startswith("## Thinking")
    assert "## Final Response" in sample.messages[-1]["content"]


def test_mcq_schema_rejects_answer_outside_options() -> None:
    with pytest.raises(ValueError, match="is not in options"):
        to_mcq_sample(
            {
                "id": "mcq-1",
                "source": "medqa",
                "question": "Question",
                "options": {"A": "one", "B": "two"},
                "answer": "C",
                "split": "test",
            }
        )


def test_rl_schema_requires_ground_truth_answer() -> None:
    with pytest.raises(ValueError, match="ground_truth_answer"):
        to_rl_sample(
            {
                "id": "rl-1",
                "source": "huatuo",
                "prompt": "Question",
                "ground_truth_answer": "",
                "split": "train",
            }
        )

from __future__ import annotations

from medical_grpo.data.quality import _summarize_lengths


def test_summarize_lengths_reports_tail_percentiles() -> None:
    summary = _summarize_lengths(range(1, 101))

    assert summary == {
        "min": 1,
        "mean": 50.5,
        "p50": 51,
        "p95": 96,
        "p99": 100,
        "max": 100,
    }

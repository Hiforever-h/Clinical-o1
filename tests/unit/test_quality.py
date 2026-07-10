"""数据长度分布统计的单元测试。"""

from __future__ import annotations

from medical_grpo.data.quality import _summarize_lengths


def test_summarize_lengths_reports_tail_percentiles() -> None:
    """长度摘要必须稳定报告均值及 P50/P95/P99 长尾指标。"""

    summary = _summarize_lengths(range(1, 101))

    assert summary == {
        "min": 1,
        "mean": 50.5,
        "p50": 51,
        "p95": 96,
        "p99": 100,
        "max": 100,
    }

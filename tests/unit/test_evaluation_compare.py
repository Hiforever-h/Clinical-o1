"""相同评测合同下逐题配对比较的单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medical_grpo.evaluation.compare import compare_evaluation_runs


def _write_run(path: Path, contract: str, correctness: list[bool]) -> None:
    """构造最小 run 目录，供比较逻辑验证 ID 对齐和统计转移。"""

    path.mkdir(parents=True)
    (path / "evaluation_contract.json").write_text(
        json.dumps({"evaluation_contract_sha256": contract}),
        encoding="utf-8",
    )
    predictions = path / "predictions"
    predictions.mkdir()
    with (predictions / "medqa_direct.jsonl").open("w", encoding="utf-8") as handle:
        for index, correct in enumerate(correctness):
            handle.write(json.dumps({"id": str(index), "correct": correct}) + "\n")


def test_compare_reports_paired_correctness_changes(tmp_path: Path) -> None:
    """候选模型新增和丢失的正确题必须进入 McNemar 配对计数。"""

    baseline = tmp_path / "base"
    candidate = tmp_path / "candidate"
    _write_run(baseline, "same", [True, True, False, False])
    _write_run(candidate, "same", [True, False, True, False])

    report = compare_evaluation_runs(baseline, candidate)
    comparison = report["comparisons"]["medqa_direct"]

    assert comparison["both_correct"] == 1
    assert comparison["baseline_only_correct"] == 1
    assert comparison["candidate_only_correct"] == 1
    assert comparison["both_wrong"] == 1
    assert comparison["accuracy_delta"] == 0.0
    assert comparison["mcnemar_holm_p"] == 1.0


def test_compare_rejects_different_contracts(tmp_path: Path) -> None:
    """Prompt、数据或生成参数不同的两个 run 不得生成误导性对比。"""

    baseline = tmp_path / "base"
    candidate = tmp_path / "candidate"
    _write_run(baseline, "contract-a", [True])
    _write_run(candidate, "contract-b", [True])

    with pytest.raises(ValueError, match="contract"):
        compare_evaluation_runs(baseline, candidate)

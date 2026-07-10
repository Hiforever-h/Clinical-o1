"""两个相同评测 contract run 的逐题配对比较与显著性统计。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.stats import binomtest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取单个最终预测文件。"""

    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _exact_mcnemar(base_only: int, candidate_only: int) -> float:
    """用二项分布计算两侧 exact McNemar p-value。"""

    discordant = base_only + candidate_only
    if discordant == 0:
        return 1.0
    return float(binomtest(base_only, discordant, p=0.5, alternative="two-sided").pvalue)


def _apply_holm_correction(comparisons: dict[str, dict[str, Any]]) -> None:
    """对所有 dataset/protocol 的 McNemar p-value 做 Holm 校正。"""

    ordered = sorted(
        comparisons,
        key=lambda name: float(comparisons[name]["mcnemar_exact_p"]),
    )
    running_max = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * float(comparisons[name]["mcnemar_exact_p"]))
        running_max = max(running_max, adjusted)
        comparisons[name]["mcnemar_holm_p"] = running_max


def _paired_bootstrap_delta(
    baseline: np.ndarray,
    candidate: np.ndarray,
    seed: int = 42,
    samples: int = 10_000,
) -> list[float]:
    """固定 seed 对配对正确性差值做 bootstrap 95% 区间。"""

    generator = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(samples):
        indices = generator.integers(0, len(baseline), size=len(baseline))
        deltas.append(float((candidate[indices] - baseline[indices]).mean()))
    lower, upper = np.quantile(np.asarray(deltas), [0.025, 0.975])
    return [float(lower), float(upper)]


def _load_run_predictions(run_dir: Path) -> tuple[str, dict[str, list[dict[str, Any]]]]:
    """读取 run contract 与所有 dataset/protocol 最终预测文件。"""

    contract = json.loads((run_dir / "evaluation_contract.json").read_text(encoding="utf-8"))
    contract_sha = str(contract["evaluation_contract_sha256"])
    groups = {
        path.stem: _read_jsonl(path)
        for path in sorted((run_dir / "predictions").glob("*.jsonl"))
    }
    if not groups:
        raise FileNotFoundError(f"run 没有最终预测文件：{run_dir}")
    return contract_sha, groups


def compare_evaluation_runs(baseline_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    """要求 contract 完全一致后，按 ID 计算提升、转移矩阵和显著性。"""

    baseline_dir = baseline_dir.resolve()
    candidate_dir = candidate_dir.resolve()
    baseline_contract, baseline_groups = _load_run_predictions(baseline_dir)
    candidate_contract, candidate_groups = _load_run_predictions(candidate_dir)
    if baseline_contract != candidate_contract:
        raise ValueError("两个 run 的 evaluation_contract_sha256 不一致，禁止比较")
    if set(baseline_groups) != set(candidate_groups):
        raise ValueError("两个 run 的 dataset/protocol 预测文件集合不一致")

    comparisons: dict[str, Any] = {}
    for group_name in sorted(baseline_groups):
        baseline_rows = baseline_groups[group_name]
        candidate_rows = candidate_groups[group_name]
        baseline_by_id = {str(row["id"]): row for row in baseline_rows}
        candidate_by_id = {str(row["id"]): row for row in candidate_rows}
        if list(baseline_by_id) != list(candidate_by_id):
            raise ValueError(f"{group_name} 的 ID 或顺序不一致")
        baseline_correct = np.asarray(
            [int(bool(baseline_by_id[id_value].get("correct"))) for id_value in baseline_by_id],
            dtype=np.float64,
        )
        candidate_correct = np.asarray(
            [int(bool(candidate_by_id[id_value].get("correct"))) for id_value in baseline_by_id],
            dtype=np.float64,
        )
        base_only = int(((baseline_correct == 1) & (candidate_correct == 0)).sum())
        candidate_only = int(((baseline_correct == 0) & (candidate_correct == 1)).sum())
        comparisons[group_name] = {
            "rows": len(baseline_correct),
            "baseline_accuracy": float(baseline_correct.mean()),
            "candidate_accuracy": float(candidate_correct.mean()),
            "accuracy_delta": float((candidate_correct - baseline_correct).mean()),
            "accuracy_delta_bootstrap_95": _paired_bootstrap_delta(
                baseline_correct, candidate_correct
            ),
            "both_correct": int(((baseline_correct == 1) & (candidate_correct == 1)).sum()),
            "baseline_only_correct": base_only,
            "candidate_only_correct": candidate_only,
            "both_wrong": int(((baseline_correct == 0) & (candidate_correct == 0)).sum()),
            "mcnemar_exact_p": _exact_mcnemar(base_only, candidate_only),
        }
    _apply_holm_correction(comparisons)
    return {
        "baseline_run": str(baseline_dir),
        "candidate_run": str(candidate_dir),
        "evaluation_contract_sha256": baseline_contract,
        "comparisons": comparisons,
    }


def write_comparison(path: Path, report: Mapping[str, Any]) -> None:
    """原子写入配对比较报告。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)

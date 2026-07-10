"""评测准确率、解析率、格式、长度、重复率和置信区间统计。"""

from __future__ import annotations

from collections import Counter
import math
import re
from typing import Any, Iterable, Mapping


def repetition_ratio(text: str, n: int = 4) -> float:
    """计算重复 n-gram 占比，用于识别 CoT 循环输出。"""

    tokens = re.findall(r"[A-Za-z0-9]+|[^\W\s]", text.lower(), flags=re.UNICODE)
    if len(tokens) < n * 2:
        return 0.0
    ngrams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(ngrams)


def _percentile(values: list[float], percentile: float) -> float:
    """使用确定性最近秩近似计算分位数。"""

    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return float(ordered[index])


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    """计算二项比例的 95% Wilson 置信区间。"""

    if total <= 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z**2 / (4 * total**2)
    ) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _macro_f1(records: list[Mapping[str, Any]]) -> float:
    """按所有 gold/predicted 类别计算未加权 macro-F1。"""

    labels = sorted(
        {str(record["gold_answer"]) for record in records}
        | {str(record["parsed_answer"]) for record in records if record.get("parsed_answer")}
    )
    scores: list[float] = []
    for label in labels:
        true_positive = sum(
            record.get("gold_answer") == label and record.get("parsed_answer") == label
            for record in records
        )
        false_positive = sum(
            record.get("gold_answer") != label and record.get("parsed_answer") == label
            for record in records
        )
        false_negative = sum(
            record.get("gold_answer") == label and record.get("parsed_answer") != label
            for record in records
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def _subgroup_accuracy(records: list[Mapping[str, Any]], meta_field: str) -> dict[str, Any]:
    """按一个 meta 字段汇总行数和 Accuracy，空字段不进入结果。"""

    groups: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        meta = record.get("meta", {})
        value = meta.get(meta_field) if isinstance(meta, Mapping) else None
        if value is not None and str(value).strip():
            groups.setdefault(str(value), []).append(record)
    return {
        name: {
            "rows": len(group),
            "accuracy": sum(bool(row.get("correct")) for row in group) / len(group),
        }
        for name, group in sorted(groups.items())
    }


def summarize_predictions(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """对一个 dataset/protocol 的逐题预测生成完整指标摘要。"""

    rows = list(records)
    total = len(rows)
    if not rows:
        raise ValueError("不能汇总空预测集合")
    correct = sum(bool(row.get("correct")) for row in rows)
    parsed = sum(bool(row.get("parse_success")) for row in rows)
    ambiguous = sum(bool(row.get("ambiguous")) for row in rows)
    format_valid = sum(bool(row.get("format_valid")) for row in rows)
    hit_max = sum(bool(row.get("hit_max_new_tokens")) for row in rows)
    completion_lengths = [float(row.get("completion_tokens", 0)) for row in rows]
    prompt_lengths = [float(row.get("prompt_tokens", 0)) for row in rows]
    repetitions = [
        float(row.get("repetition_4gram_ratio", repetition_ratio(str(row.get("raw_completion", "")))))
        for row in rows
    ]
    total_seconds = sum(float(row.get("estimated_sample_seconds", 0.0)) for row in rows)
    confusion = Counter(
        f"{row.get('gold_answer')}->{row.get('parsed_answer') or 'UNPARSED'}" for row in rows
    )
    return {
        "rows": total,
        "correct": correct,
        "accuracy": correct / total,
        "accuracy_wilson_95": wilson_interval(correct, total),
        "parsed_rows": parsed,
        "parse_success": parsed / total,
        "ambiguous_parse_rate": ambiguous / total,
        "accuracy_on_parsed": correct / parsed if parsed else 0.0,
        "format_compliance": format_valid / total,
        "macro_f1": _macro_f1(rows),
        "hit_max_new_tokens_rate": hit_max / total,
        "completion_tokens": {
            "mean": sum(completion_lengths) / total,
            "p50": _percentile(completion_lengths, 0.50),
            "p95": _percentile(completion_lengths, 0.95),
            "max": max(completion_lengths),
        },
        "prompt_tokens": {
            "mean": sum(prompt_lengths) / total,
            "p95": _percentile(prompt_lengths, 0.95),
            "max": max(prompt_lengths),
        },
        "repetition_4gram": {
            "mean": sum(repetitions) / total,
            "p95": _percentile(repetitions, 0.95),
            "abnormal_rate_over_0_20": sum(value > 0.20 for value in repetitions) / total,
        },
        "estimated_generation_seconds": total_seconds,
        "estimated_examples_per_second": total / total_seconds if total_seconds > 0 else None,
        "parse_methods": dict(Counter(str(row.get("parse_method")) for row in rows)),
        "confusion_matrix": dict(sorted(confusion.items())),
        "subgroups": {
            "medqa_meta_info": _subgroup_accuracy(rows, "meta_info"),
            "medmcqa_subject_name": _subgroup_accuracy(rows, "subject_name"),
        },
    }

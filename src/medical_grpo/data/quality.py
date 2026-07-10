"""M1 规范数据的质量门禁和字符/token 长度统计。"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import math
import random
from typing import Any, Callable, Iterable, Mapping

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from medical_grpo.data.contamination import normalize_for_audit
from medical_grpo.data.sources import BASE_MODEL, BASE_MODEL_REVISION
from medical_grpo.data.schema import validate_mcq_record, validate_rl_record, validate_sft_record


Validator = Callable[[Mapping[str, Any]], list[str]]


def _percentile(sorted_values: list[int], percentile: float) -> int:
    """从已排序整数中取向上取整的经验百分位。"""

    if not sorted_values:
        return 0
    index = math.ceil((len(sorted_values) - 1) * percentile)
    return int(sorted_values[index])


def _summarize_lengths(values: Iterable[int]) -> dict[str, int | float]:
    """生成固定字段的 min/mean/P50/P95/P99/max 摘要。"""

    ordered = sorted(int(value) for value in values)
    if not ordered:
        return {"min": 0, "mean": 0.0, "p50": 0, "p95": 0, "p99": 0, "max": 0}
    return {
        "min": ordered[0],
        "mean": round(sum(ordered) / len(ordered), 2),
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1],
    }


def _token_lengths(
    texts: list[str], tokenizer: PreTrainedTokenizerBase, *, batch_size: int = 128
) -> list[int]:
    """使用 fast tokenizer 分批统计，不做 padding 或 truncation。"""

    lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_length=True,
        )
        lengths.extend(int(value) for value in encoded["length"])
    return lengths


def _field_lengths(
    records: list[dict[str, Any]], field: str, tokenizer: PreTrainedTokenizerBase
) -> dict[str, Any]:
    """同时统计一个字段的字符长度、token 长度和超长数量。"""

    texts = [str(record.get(field, "")) for record in records]
    token_lengths = _token_lengths(texts, tokenizer)
    return {
        "characters": _summarize_lengths(len(text) for text in texts),
        "tokens": _summarize_lengths(token_lengths),
        "over_4096_tokens": sum(length > 4096 for length in token_lengths),
    }


def _render_sft_conversations(
    records: list[dict[str, Any]], tokenizer: PreTrainedTokenizerBase
) -> list[str]:
    """使用固定 revision 的 Qwen chat template 渲染完整 SFT 对话。"""

    return [
        str(tokenizer.apply_chat_template(record["messages"], tokenize=False, add_generation_prompt=False))
        for record in records
    ]


def _duplicate_count(records: list[dict[str, Any]], text_field: str) -> int:
    """统计标准化 prompt 中除第一条外的重复记录数量。"""

    counts = Counter(normalize_for_audit(str(record[text_field])) for record in records)
    return sum(count - 1 for count in counts.values() if count > 1)


def _schema_summary(records: list[dict[str, Any]], validator: Validator) -> dict[str, Any]:
    """校验全部记录，但只保留前 20 个错误示例，防止报告无限膨胀。"""

    invalid: list[dict[str, Any]] = []
    invalid_count = 0
    for record in records:
        errors = validator(record)
        if errors:
            invalid_count += 1
            if len(invalid) < 20:
                invalid.append({"id": record.get("id"), "errors": errors})
    return {
        "rows": len(records),
        "valid_rows": len(records) - invalid_count,
        "invalid_rows": invalid_count,
        "first_errors": invalid,
    }


def build_quality_report(
    datasets: Mapping[str, list[dict[str, Any]]], *, seed: int = 42
) -> dict[str, Any]:
    """逐条校验所有数据，并计算 Qwen tokenizer 长度分布。"""

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        revision=BASE_MODEL_REVISION,
        use_fast=True,
        trust_remote_code=False,
    )
    # 每种正式文件显式绑定 schema、去重字段和需要统计长度的字段。
    specifications: dict[str, tuple[str, Validator, list[str]]] = {
        "sft_train": ("question", validate_sft_record, ["question", "reasoning", "response"]),
        "sft_dev": ("question", validate_sft_record, ["question", "reasoning", "response"]),
        "rl_train": ("prompt", validate_rl_record, ["prompt", "ground_truth_answer"]),
        "rl_dev": ("prompt", validate_rl_record, ["prompt", "ground_truth_answer"]),
        "medqa_test": ("question", validate_mcq_record, ["question"]),
        "medmcqa_validation": ("question", validate_mcq_record, ["question"]),
        "pubmedqa_labeled": ("question", validate_mcq_record, ["question"]),
    }

    dataset_reports: dict[str, Any] = {}
    rng = random.Random(seed)
    for name, records in datasets.items():
        text_field, validator, length_fields = specifications[name]
        # 固定种子冻结人工抽查 ID，后续重新生成时可以检查样本是否漂移。
        sampled_ids = [str(record["id"]) for record in rng.sample(records, min(100, len(records)))]
        report = {
            "schema": _schema_summary(records, validator),
            "duplicate_normalized_prompts": _duplicate_count(records, text_field),
            "sampled_ids": sampled_ids,
            "lengths": {
                field: _field_lengths(records, field, tokenizer) for field in length_fields
            },
        }
        if name.startswith("sft_"):
            conversations = _render_sft_conversations(records, tokenizer)
            conversation_tokens = _token_lengths(conversations, tokenizer)
            report["lengths"]["chat_template"] = {
                "characters": _summarize_lengths(len(text) for text in conversations),
                "tokens": _summarize_lengths(conversation_tokens),
                "over_4096_tokens": sum(length > 4096 for length in conversation_tokens),
            }
        dataset_reports[name] = report

    total_rows = sum(len(records) for records in datasets.values())
    total_invalid = sum(report["schema"]["invalid_rows"] for report in dataset_reports.values())
    training_names = ("sft_train", "sft_dev", "rl_train", "rl_dev")
    evaluation_names = ("medqa_test", "medmcqa_validation", "pubmedqa_labeled")
    training_duplicates = sum(
        dataset_reports[name]["duplicate_normalized_prompts"] for name in training_names
    )
    evaluation_duplicates = sum(
        dataset_reports[name]["duplicate_normalized_prompts"] for name in evaluation_names
    )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer": {"repository": BASE_MODEL, "revision": BASE_MODEL_REVISION},
        "summary": {
            "total_rows": total_rows,
            "schema_invalid_rows": total_invalid,
            "training_duplicate_normalized_prompts": training_duplicates,
            "official_evaluation_duplicate_normalized_prompts": evaluation_duplicates,
            "sampled_rows": sum(len(report["sampled_ids"]) for report in dataset_reports.values()),
        },
        "datasets": dataset_reports,
    }


def quality_report_markdown(report: Mapping[str, Any]) -> str:
    """将机器可读 JSON 摘要转换成便于查看的 Markdown 表格。"""

    lines = [
        "# English Mainline Data Quality Report",
        "",
        f"Generated at: `{report['generated_at']}`",
        "",
        f"Tokenizer: `{report['tokenizer']['repository']}` at `{report['tokenizer']['revision']}`",
        "",
        "| Dataset | Rows | Invalid | Duplicate prompts | Chat/prompt P95 tokens | Over 4096 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, dataset in report["datasets"].items():
        length_key = "chat_template" if "chat_template" in dataset["lengths"] else next(
            iter(dataset["lengths"])
        )
        length_stats = dataset["lengths"][length_key]
        lines.append(
            f"| {name} | {dataset['schema']['rows']} | {dataset['schema']['invalid_rows']} | "
            f"{dataset['duplicate_normalized_prompts']} | {length_stats['tokens']['p95']} | "
            f"{length_stats['over_4096_tokens']} |"
        )
    lines.extend(
        [
            "",
            "The pipeline validates every row. The `sampled_ids` arrays in the JSON report freeze a "
            "deterministic sample of up to 100 rows per split for manual inspection.",
            "Official evaluation rows are retained unchanged; duplicate counts there are reported, not removed.",
            "",
        ]
    )
    return "\n".join(lines)

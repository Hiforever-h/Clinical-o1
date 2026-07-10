"""不加载 7B 权重的评测数据、Prompt、解析器和 token 长度检查。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medical_grpo.evaluation.config import EvaluationConfig
from medical_grpo.evaluation.contract import build_evaluation_contract
from medical_grpo.evaluation.data import load_evaluation_datasets
from medical_grpo.evaluation.modeling import load_eval_tokenizer
from medical_grpo.evaluation.parsing import parse_mcq_answer
from medical_grpo.evaluation.prompts import build_messages


def _parser_self_check() -> dict[str, Any]:
    """用合成对抗输出验证解析优先级，不接触 benchmark 正确率。"""

    options = {"A": "alpha", "B": "beta", "C": "maybe", "D": "delta"}
    cases = [
        ("Answer: B", "direct", "B"),
        ("b", "direct", "B"),
        (r"\boxed{C}", "direct", "C"),
        ("beta", "direct", "B"),
        ("## Thinking\nA is wrong.\n## Final Response\nAnswer: B", "cot", "B"),
        ("## Thinking\nReason.\n## Final Response\nFinal Answer is D", "cot", "D"),
    ]
    failures: list[dict[str, str]] = []
    for text, protocol, expected in cases:
        parsed = parse_mcq_answer(text, options, protocol)
        if parsed.parsed_answer != expected:
            failures.append({"text": text, "expected": expected, "actual": str(parsed.parsed_answer)})
    if failures:
        raise AssertionError(f"parser self-check 失败：{failures}")
    return {"cases": len(cases), "failures": 0, "status": "ok"}


def run_evaluation_dry_run(
    config: EvaluationConfig,
    repo_root: Path,
    selected_datasets: tuple[str, ...],
    selected_protocols: tuple[str, ...],
    max_samples_per_dataset: int | None,
) -> dict[str, Any]:
    """全量构造所选 Prompt 并检查输入长度、数据哈希与解析器合同。"""

    datasets, data_report = load_evaluation_datasets(
        config,
        repo_root,
        selected_datasets,
        max_samples_per_dataset,
    )
    tokenizer = load_eval_tokenizer(config)
    length_report: dict[str, Any] = {}
    total_predictions = 0
    for dataset_name in selected_datasets:
        records = datasets[dataset_name]
        for protocol_name in selected_protocols:
            texts = [
                tokenizer.apply_chat_template(
                    build_messages(record, protocol_name),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for record in records
            ]
            lengths = tokenizer(
                texts,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=False,
            )["input_ids"]
            token_lengths = [len(ids) for ids in lengths]
            overlong = sum(length > config.inference.max_input_length for length in token_lengths)
            if overlong:
                raise ValueError(
                    f"{dataset_name}/{protocol_name} 有 {overlong} 条输入超过 "
                    f"max_input_length={config.inference.max_input_length}"
                )
            key = f"{dataset_name}/{protocol_name}"
            length_report[key] = {
                "rows": len(records),
                "min_tokens": min(token_lengths),
                "mean_tokens": round(sum(token_lengths) / len(token_lengths), 2),
                "max_tokens": max(token_lengths),
                "overlong_rows": overlong,
            }
            total_predictions += len(records)

    contract = build_evaluation_contract(
        config,
        selected_datasets,
        selected_protocols,
        max_samples_per_dataset,
        data_report,
    )
    return {
        "status": "ok",
        "profile": config.profile_name,
        "selected_datasets": list(selected_datasets),
        "selected_protocols": list(selected_protocols),
        "max_samples_per_dataset": max_samples_per_dataset,
        "total_predictions": total_predictions,
        "data": data_report,
        "input_token_audit": length_report,
        "parser_self_check": _parser_self_check(),
        "evaluation_contract": contract,
    }


def write_evaluation_dry_run(path: Path, report: dict[str, Any]) -> None:
    """原子写入 eval dry-run 报告。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)

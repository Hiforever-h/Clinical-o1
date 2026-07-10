"""M0/M1 的 CPU-only 合同检查。

这里不模拟尚未实现的训练、Reward 或模型推理，只验证已经落地的数据文件、
SFT 对话结构和评测答案映射，避免 dry-run 给出虚假的阶段完成信号。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medical_grpo.data.schema import to_mcq_sample, to_rl_sample, to_sft_sample
from medical_grpo.tracking.artifacts import sha256_file


VALID_COMPONENTS = {"all", "data", "sft", "rl", "eval"}


def _read_first_jsonl(path: Path, count: int = 1) -> list[dict[str, Any]]:
    """只读取少量样本做合同检查，避免 dry-run 重复加载大文件。"""

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
            if len(records) >= count:
                break
    if len(records) < count:
        raise ValueError(f"JSONL 样本不足：{path}")
    return records


def _data_check(repo_root: Path) -> dict[str, Any]:
    """重算所有正式数据的行数和 SHA256，并检查两个质量门禁。"""

    manifest = repo_root / "data" / "manifests" / "english_mainline.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"M1 manifest 不存在：{manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    checked_files: dict[str, int] = {}
    for name, metadata in payload["files"].items():
        path = repo_root / metadata["path"]
        if not path.is_file():
            raise FileNotFoundError(f"manifest 文件不存在：{path}")
        with path.open("r", encoding="utf-8") as handle:
            rows = sum(1 for _ in handle)
        if rows != metadata["rows"]:
            raise AssertionError(f"{name} 行数不一致：{rows} != {metadata['rows']}")
        digest = sha256_file(path)
        if digest != metadata["sha256"]:
            raise AssertionError(f"{name} SHA256 不一致：{digest} != {metadata['sha256']}")
        checked_files[name] = rows

    quality = json.loads((repo_root / payload["data_quality_report"]).read_text(encoding="utf-8"))
    if quality["summary"]["schema_invalid_rows"] != 0:
        raise AssertionError("质量报告仍包含 schema 错误")
    if quality["summary"]["training_duplicate_normalized_prompts"] != 0:
        raise AssertionError("质量报告仍包含重复训练 prompt")

    contamination = json.loads(
        (repo_root / payload["contamination_report"]).read_text(encoding="utf-8")
    )
    unresolved = sum(audit["unresolved_review_count"] for audit in contamination["audits"])
    if unresolved:
        raise AssertionError(f"污染报告仍有 {unresolved} 个未决候选")

    return {
        "manifest": str(manifest.relative_to(repo_root)),
        "aggregate_sha256": payload["aggregate_sha256"],
        "checked_files": checked_files,
        "schema_invalid_rows": 0,
        "training_duplicate_normalized_prompts": 0,
        "unresolved_contamination_candidates": 0,
        "status": "ok",
    }


def _sft_check(repo_root: Path) -> dict[str, Any]:
    """确认正式 SFT 文件能转换为规范的 user/assistant 对话。"""

    paths = [
        repo_root / "data/processed/sft/huatuo_o1_sft_en_train.jsonl",
        repo_root / "data/processed/sft/huatuo_o1_sft_en_dev.jsonl",
    ]
    raw_records = [_read_first_jsonl(path)[0] for path in paths]
    samples = [to_sft_sample(record) for record in raw_records]
    for raw, sample in zip(raw_records, samples, strict=True):
        if sample.to_dict() != raw:
            raise AssertionError(f"SFT schema round-trip 改变了正式记录：{sample.id}")
        roles = [message["role"] for message in sample.messages]
        if roles != ["user", "assistant"]:
            raise AssertionError(f"SFT 对话角色异常：{sample.id} -> {roles}")
        assistant = sample.messages[-1]["content"]
        if "## Thinking" not in assistant or "## Final Response" not in assistant:
            raise AssertionError(f"SFT 输出格式异常：{sample.id}")
    return {
        "files": [str(path.relative_to(repo_root)) for path in paths],
        "sample_ids": [sample.id for sample in samples],
        "status": "ok",
    }


def _eval_check(repo_root: Path) -> dict[str, Any]:
    """确认三个评测集合的选项顺序和答案键均可由 canonical schema 接受。"""

    paths = [
        repo_root / "data/processed/eval/medqa_usmle_test.jsonl",
        repo_root / "data/processed/eval/medmcqa_validation.jsonl",
        repo_root / "data/processed/eval/pubmedqa_labeled_test.jsonl",
    ]
    raw_records = [_read_first_jsonl(path)[0] for path in paths]
    samples = [to_mcq_sample(record) for record in raw_records]
    for raw, sample in zip(raw_records, samples, strict=True):
        if sample.to_dict() != raw:
            raise AssertionError(f"评测 schema round-trip 改变了正式记录：{sample.id}")
    return {
        "files": [str(path.relative_to(repo_root)) for path in paths],
        "sample_answers": {sample.id: sample.answer for sample in samples},
        "status": "ok",
    }


def _rl_check(repo_root: Path) -> dict[str, Any]:
    """确认 RL train/dev 中的 prompt 与标准答案可以无损通过 canonical schema。"""

    paths = [
        repo_root / "data/processed/rl/huatuo_o1_verifiable_en_train.jsonl",
        repo_root / "data/processed/rl/huatuo_o1_verifiable_en_dev.jsonl",
    ]
    raw_records = [_read_first_jsonl(path)[0] for path in paths]
    samples = [to_rl_sample(record) for record in raw_records]
    for raw, sample in zip(raw_records, samples, strict=True):
        if sample.to_dict() != raw:
            raise AssertionError(f"RL schema round-trip 改变了正式记录：{sample.id}")
    return {
        "files": [str(path.relative_to(repo_root)) for path in paths],
        "sample_ids": [sample.id for sample in samples],
        "status": "ok",
    }


def run_dry_checks(repo_root: Path, component: str = "all") -> dict[str, Any]:
    """执行指定组件；`all` 只覆盖当前确实完成的 M0/M1 能力。"""

    if component not in VALID_COMPONENTS:
        raise ValueError(f"component 必须是 {sorted(VALID_COMPONENTS)} 之一")
    checks = {"data": _data_check, "sft": _sft_check, "rl": _rl_check, "eval": _eval_check}
    selected = checks if component == "all" else {component: checks[component]}
    return {name: check(repo_root.resolve()) for name, check in selected.items()}

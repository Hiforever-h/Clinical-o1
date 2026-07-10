"""英文主线 M1 数据准备总管线。

流程固定为：加载指定 revision → 转换 canonical schema → 集合内去重 →
隔离 benchmark/SFT 重叠 → 固定种子切分 → 原子写入数据、manifest 和报告。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

from datasets import Dataset, load_dataset

from medical_grpo.data.contamination import (
    AuditRecord,
    AuditResult,
    audit_records,
    normalize_for_audit,
    promote_review_candidates_to_exclusions,
)
from medical_grpo.data.quality import build_quality_report, quality_report_markdown
from medical_grpo.data.sources import BASE_MODEL, BASE_MODEL_REVISION, SOURCES, DatasetSource
from medical_grpo.data.schema import to_mcq_sample, to_rl_sample, to_sft_sample
from medical_grpo.tracking.artifacts import sha256_file


@dataclass(frozen=True)
class DataPreparationConfig:
    """英文 M1 管线的全部可调参数；默认值就是正式数据基线。"""

    repo_root: Path
    seed: int = 42
    sft_dev_ratio: float = 0.02
    rl_dev_ratio: float = 0.02
    max_sft_samples: int | None = None
    max_rl_samples: int | None = None
    fuzzy_review_threshold: float = 0.82
    fuzzy_exclude_threshold: float = 0.93
    generate_quality_report: bool = True


def _load_pinned_source(source: DatasetSource) -> Dataset:
    """只从配置中声明的固定 revision 加载一个 Hugging Face split。"""

    kwargs: dict[str, Any] = {
        "path": source.repository,
        "split": source.split,
        "revision": source.revision,
    }
    if source.config is not None:
        kwargs["name"] = source.config
    dataset = load_dataset(**kwargs)
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected Dataset for {source.key}, got {type(dataset).__name__}")
    return dataset


def _limited_rows(dataset: Dataset, limit: int | None) -> Iterable[Mapping[str, Any]]:
    """为 smoke test 提供稳定的前 N 条截断；正式运行传入 ``None``。"""

    row_count = len(dataset) if limit is None else min(len(dataset), limit)
    for index in range(row_count):
        yield dataset[index]


def _source_provenance(source: DatasetSource, row_index: int) -> dict[str, Any]:
    """把原始仓库、revision、split 和行号写进每条记录的 meta。"""

    return {
        "dataset": source.repository,
        "revision": source.revision,
        "config": source.config,
        "hf_split": source.split,
        "row_index": row_index,
    }


def _convert_sft(dataset: Dataset, source: DatasetSource, limit: int | None) -> list[dict[str, Any]]:
    """将 Huatuo English SFT 三列转换为项目规范格式。"""

    records: list[dict[str, Any]] = []
    for index, row in enumerate(_limited_rows(dataset, limit), start=1):
        record = {
            "id": f"huatuo_sft_en_{index:06d}",
            "source": "huatuo_o1_sft_en",
            "question": row.get("Question"),
            "reasoning": row.get("Complex_CoT"),
            "response": row.get("Response"),
            "split": "train",
            "meta": _source_provenance(source, index),
        }
        records.append(to_sft_sample(record).to_dict())
    return records


def _convert_rl(dataset: Dataset, source: DatasetSource, limit: int | None) -> list[dict[str, Any]]:
    """将 Huatuo verifiable problem 转为 RL prompt/answer 结构。"""

    records: list[dict[str, Any]] = []
    for index, row in enumerate(_limited_rows(dataset, limit), start=1):
        record = {
            "id": f"huatuo_rl_en_{index:06d}",
            "source": "huatuo_o1_verifiable_en",
            "prompt": row.get("Open-ended Verifiable Question"),
            "ground_truth_answer": row.get("Ground-True Answer"),
            "split": "train",
            "meta": _source_provenance(source, index),
        }
        records.append(to_rl_sample(record).to_dict())
    return records


def _normalize_options(raw_options: Any) -> dict[str, str]:
    """兼容 Hugging Face 中常见的字典、label/text 和键值列表选项结构。"""

    if isinstance(raw_options, Mapping):
        labels = raw_options.get("label")
        texts = raw_options.get("text")
        if isinstance(labels, list) and isinstance(texts, list):
            return {str(label): str(text) for label, text in zip(labels, texts, strict=True)}
        return {str(key): str(value) for key, value in raw_options.items()}
    if isinstance(raw_options, list):
        options: dict[str, str] = {}
        for item in raw_options:
            if isinstance(item, Mapping):
                key = item.get("key", item.get("label"))
                value = item.get("value", item.get("text"))
                if key is not None and value is not None:
                    options[str(key)] = str(value)
        return options
    raise ValueError(f"Unsupported options value: {type(raw_options).__name__}")


def _convert_medqa(dataset: Dataset, source: DatasetSource) -> list[dict[str, Any]]:
    """保持 MedQA 选项键和官方答案映射不变地转换 test split。"""

    records: list[dict[str, Any]] = []
    for index, row in enumerate(dataset, start=1):
        record = {
            "id": row.get("id") or f"medqa_test_{index:06d}",
            "source": "medqa",
            "question": row.get("question"),
            "options": _normalize_options(row.get("options")),
            "answer": row.get("answer_idx"),
            "answer_text": row.get("answer"),
            "split": "test",
            "meta": {
                **_source_provenance(source, index),
                "meta_info": row.get("meta_info"),
            },
        }
        records.append(to_mcq_sample(record).to_dict())
    return records


def _convert_medmcqa(dataset: Dataset, source: DatasetSource) -> list[dict[str, Any]]:
    """将 MedMCQA 的 opa/opb/opc/opd 与 0-based cop 映射为 A/B/C/D。"""

    records: list[dict[str, Any]] = []
    for index, row in enumerate(dataset, start=1):
        options = {
            "A": str(row.get("opa", "")),
            "B": str(row.get("opb", "")),
            "C": str(row.get("opc", "")),
            "D": str(row.get("opd", "")),
        }
        answer_index = int(row.get("cop"))
        if answer_index not in range(4):
            raise ValueError(f"MedMCQA validation row {index} has invalid cop={answer_index}")
        answer = "ABCD"[answer_index]
        record = {
            "id": row.get("id") or f"medmcqa_validation_{index:06d}",
            "source": "medmcqa",
            "question": row.get("question"),
            "options": options,
            "answer": answer,
            "answer_text": options[answer],
            "explanation": row.get("exp"),
            "split": "dev",
            "meta": {
                **_source_provenance(source, index),
                "choice_type": row.get("choice_type"),
                "subject_name": row.get("subject_name"),
                "topic_name": row.get("topic_name"),
            },
        }
        records.append(to_mcq_sample(record).to_dict())
    return records


def _convert_pubmedqa(dataset: Dataset, source: DatasetSource) -> list[dict[str, Any]]:
    """将 PubMedQA labeled 的 yes/no/maybe 映射为三选一评测格式。"""

    answer_map = {"yes": "A", "no": "B", "maybe": "C"}
    options = {"A": "yes", "B": "no", "C": "maybe"}
    records: list[dict[str, Any]] = []
    for index, row in enumerate(dataset, start=1):
        context = row.get("context") or {}
        contexts = context.get("contexts", []) if isinstance(context, Mapping) else []
        context_text = "\n".join(str(value) for value in contexts)
        question = f"Abstract context:\n{context_text}\n\nQuestion:\n{row.get('question', '')}"
        final_decision = str(row.get("final_decision", "")).strip().lower()
        if final_decision not in answer_map:
            raise ValueError(f"PubMedQA row {index} has invalid final_decision={final_decision!r}")
        answer = answer_map[final_decision]
        record = {
            "id": f"pubmedqa_{row.get('pubid', index)}",
            "source": "pubmedqa",
            "question": question,
            "options": options,
            "answer": answer,
            "answer_text": options[answer],
            "explanation": row.get("long_answer"),
            "split": "test",
            "meta": {
                **_source_provenance(source, index),
                "pubid": row.get("pubid"),
                "question_only": row.get("question"),
            },
        }
        records.append(to_mcq_sample(record).to_dict())
    return records


def _audit_record(record: Mapping[str, Any], text_field: str) -> AuditRecord:
    """从完整记录中提取污染审计需要的 ID、来源和问题文本。"""

    return AuditRecord(id=str(record["id"]), source=str(record["source"]), text=str(record[text_field]))


def _deduplicate_records(
    records: list[dict[str, Any]], text_field: str, audit_name: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """保留首个标准化 prompt，并记录每个重复项指向的保留 ID。"""

    seen: dict[str, str] = {}
    retained: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for record in records:
        normalized = normalize_for_audit(str(record[text_field]))
        kept_id = seen.get(normalized)
        if kept_id is None:
            seen[normalized] = str(record["id"])
            retained.append(record)
            continue
        excluded.append(
            {
                "audit": audit_name,
                "id": str(record["id"]),
                "decision": "exclude_duplicate_normalized_prompt",
                "reference_id": kept_id,
            }
        )
    return retained, excluded


def _split_records(
    records: list[dict[str, Any]], dev_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """使用局部随机数生成器做确定性 train/dev 切分，不修改输入记录。"""

    if not 0.0 <= dev_ratio < 1.0:
        raise ValueError("dev_ratio must be in [0, 1)")
    shuffled = [dict(record) for record in records]
    random.Random(seed).shuffle(shuffled)
    dev_size = 0 if dev_ratio == 0.0 else max(1, round(len(shuffled) * dev_ratio))
    dev = shuffled[:dev_size]
    train = shuffled[dev_size:]
    for record in train:
        record["split"] = "train"
    for record in dev:
        record["split"] = "dev"
    return train, dev


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    """先写临时文件再替换目标，避免中断时留下半个 JSONL。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """原子写入带缩进的 UTF-8 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _file_manifest(
    repo_root: Path,
    path: Path,
    rows: int,
    schema: str,
    sources: list[DatasetSource],
    exclusions: list[str] | None = None,
) -> dict[str, Any]:
    """生成仓库相对路径、行数、大小、schema、来源和 SHA256 清单。"""

    return {
        "path": path.relative_to(repo_root).as_posix(),
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "schema": schema,
        "sources": [asdict(source) for source in sources],
        "exclusions": exclusions or [],
    }


def _write_audit_candidates(path: Path, audits: list[AuditResult]) -> int:
    """汇总多次审计的候选对，保留分数和文本摘录。"""

    records: list[dict[str, Any]] = []
    for audit in audits:
        for candidate in audit.candidates_as_dicts():
            records.append({"audit": audit.name, **candidate})
    return _write_jsonl(path, records)


def _contamination_markdown(report: Mapping[str, Any]) -> str:
    """生成人工查看用的污染摘要表和决策策略。"""

    lines = [
        "# English Mainline Contamination Report",
        "",
        f"Generated at: `{report['generated_at']}`",
        "",
        "| Audit | Queries | References | Excluded | Candidates | Unresolved | Max TF-IDF cosine |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for audit in report["audits"]:
        lines.append(
            f"| {audit['name']} | {audit['query_count']} | {audit['reference_count']} | "
            f"{audit['excluded_count']} | {audit['candidate_count']} | "
            f"{audit['unresolved_review_count']} | "
            f"{audit['maximum_tfidf_cosine']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Decision policy",
            "",
            "- Exact normalized matches are excluded.",
            "- A shared normalized contiguous span of 64 characters is excluded.",
            "- Fuzzy pairs require both high character TF-IDF cosine and character 5-gram overlap.",
            "- Threshold-triggered fuzzy candidates are conservatively excluded; their scores remain reviewable.",
            "- Full candidate details are stored in `contamination_candidates.jsonl`.",
            "",
        ]
    )
    return "\n".join(lines)


def _aggregate_sha256(file_manifests: Mapping[str, Mapping[str, Any]]) -> str:
    """将所有正式文件的名称、哈希和行数合成为数据版本指纹。"""

    digest = hashlib.sha256()
    for name, manifest in sorted(file_manifests.items()):
        digest.update(f"{name}\0{manifest['sha256']}\0{manifest['rows']}\n".encode())
    return digest.hexdigest()


def _audit_exclusions(audits: list[AuditResult]) -> list[dict[str, Any]]:
    """从候选对中提取真正被排除的记录及完整理由。"""

    excluded: list[dict[str, Any]] = []
    for audit in audits:
        for candidate in audit.candidates_as_dicts():
            if not str(candidate["decision"]).startswith("exclude_"):
                continue
            excluded.append(
                {
                    "audit": audit.name,
                    "id": candidate.pop("query_id"),
                    **candidate,
                }
            )
    return excluded


def prepare_english_mainline(config: DataPreparationConfig) -> dict[str, Any]:
    """准备全部 M1 数据、污染报告、质量报告和 manifest。"""

    repo_root = config.repo_root.resolve()
    processed_root = repo_root / "data" / "processed"
    manifests_root = repo_root / "data" / "manifests"
    reports_root = repo_root / "reports" / "data"

    # 先全部加载并完成 schema 转换，任何字段异常都会在写文件前失败。
    loaded = {key: _load_pinned_source(source) for key, source in SOURCES.items()}
    raw_sft_records = _convert_sft(
        loaded["huatuo_sft_en"], SOURCES["huatuo_sft_en"], config.max_sft_samples
    )
    raw_rl_records = _convert_rl(
        loaded["huatuo_verifiable"], SOURCES["huatuo_verifiable"], config.max_rl_samples
    )
    sft_records, sft_duplicate_exclusions = _deduplicate_records(
        raw_sft_records, "question", "sft_internal_dedup"
    )
    rl_records, rl_duplicate_exclusions = _deduplicate_records(
        raw_rl_records, "prompt", "rl_internal_dedup"
    )
    medqa_records = _convert_medqa(loaded["medqa_usmle"], SOURCES["medqa_usmle"])
    medmcqa_records = _convert_medmcqa(loaded["medmcqa"], SOURCES["medmcqa"])
    pubmedqa_records = _convert_pubmedqa(loaded["pubmedqa_labeled"], SOURCES["pubmedqa_labeled"])

    eval_audit_records = [
        *(_audit_record(record, "question") for record in medqa_records),
        *(_audit_record(record, "question") for record in medmcqa_records),
        *(
            AuditRecord(
                id=str(record["id"]),
                source=str(record["source"]),
                text=str(record.get("meta", {}).get("question_only", record["question"])),
            )
            for record in pubmedqa_records
        ),
    ]

    # SFT 先保护最终 benchmark；RL 再依次避开干净 SFT 和 benchmark。
    sft_vs_eval = promote_review_candidates_to_exclusions(audit_records(
        "sft_vs_final_eval",
        (_audit_record(record, "question") for record in sft_records),
        eval_audit_records,
        review_threshold=config.fuzzy_review_threshold,
        exclude_threshold=config.fuzzy_exclude_threshold,
    ))
    clean_sft = [record for record in sft_records if record["id"] not in sft_vs_eval.excluded_ids]

    rl_vs_sft = promote_review_candidates_to_exclusions(audit_records(
        "rl_vs_sft",
        (_audit_record(record, "prompt") for record in rl_records),
        (_audit_record(record, "question") for record in clean_sft),
        review_threshold=config.fuzzy_review_threshold,
        exclude_threshold=config.fuzzy_exclude_threshold,
        top_k=2,
    ))
    rl_without_sft = [record for record in rl_records if record["id"] not in rl_vs_sft.excluded_ids]

    rl_vs_eval = promote_review_candidates_to_exclusions(audit_records(
        "rl_vs_final_eval",
        (_audit_record(record, "prompt") for record in rl_without_sft),
        eval_audit_records,
        review_threshold=config.fuzzy_review_threshold,
        exclude_threshold=config.fuzzy_exclude_threshold,
    ))
    clean_rl = [record for record in rl_without_sft if record["id"] not in rl_vs_eval.excluded_ids]

    sft_train, sft_dev = _split_records(clean_sft, config.sft_dev_ratio, config.seed)
    rl_train, rl_dev = _split_records(clean_rl, config.rl_dev_ratio, config.seed)

    # 所有输出路径在一个位置声明，便于 manifest 与真实文件保持一一对应。
    output_records = {
        "sft_train": (processed_root / "sft" / "huatuo_o1_sft_en_train.jsonl", sft_train, "sft"),
        "sft_dev": (processed_root / "sft" / "huatuo_o1_sft_en_dev.jsonl", sft_dev, "sft"),
        "rl_train": (processed_root / "rl" / "huatuo_o1_verifiable_en_train.jsonl", rl_train, "rl"),
        "rl_dev": (processed_root / "rl" / "huatuo_o1_verifiable_en_dev.jsonl", rl_dev, "rl"),
        "medqa_test": (processed_root / "eval" / "medqa_usmle_test.jsonl", medqa_records, "mcq"),
        "medmcqa_validation": (
            processed_root / "eval" / "medmcqa_validation.jsonl",
            medmcqa_records,
            "mcq",
        ),
        "pubmedqa_labeled": (
            processed_root / "eval" / "pubmedqa_labeled_test.jsonl",
            pubmedqa_records,
            "mcq",
        ),
    }

    file_manifests: dict[str, dict[str, Any]] = {}
    for name, (path, records, schema) in output_records.items():
        row_count = _write_jsonl(path, records)
        if name.startswith("sft_"):
            sources = [SOURCES["huatuo_sft_en"]]
            exclusions = ["sft_vs_final_eval"]
        elif name.startswith("rl_"):
            sources = [SOURCES["huatuo_verifiable"]]
            exclusions = ["rl_vs_sft", "rl_vs_final_eval"]
        elif name == "medqa_test":
            sources = [SOURCES["medqa_usmle"]]
            exclusions = []
        elif name == "medmcqa_validation":
            sources = [SOURCES["medmcqa"]]
            exclusions = []
        else:
            sources = [SOURCES["pubmedqa_labeled"]]
            exclusions = []
        file_manifests[name] = _file_manifest(
            repo_root, path, row_count, schema, sources, exclusions
        )

    audits = [sft_vs_eval, rl_vs_sft, rl_vs_eval]
    candidates_path = reports_root / "contamination_candidates.jsonl"
    candidate_count = _write_audit_candidates(candidates_path, audits)

    contamination_report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            **asdict(config),
            "repo_root": str(config.repo_root),
        },
        "sources": {key: asdict(source) for key, source in SOURCES.items()},
        "audits": [audit.summary_dict() for audit in audits],
        "candidate_file": candidates_path.relative_to(repo_root).as_posix(),
        "candidate_rows": candidate_count,
        "rows": {
            "sft_before": len(raw_sft_records),
            "sft_after_internal_dedup": len(sft_records),
            "sft_after_eval_exclusion": len(clean_sft),
            "rl_before": len(raw_rl_records),
            "rl_after_internal_dedup": len(rl_records),
            "rl_after_sft_exclusion": len(rl_without_sft),
            "rl_after_eval_exclusion": len(clean_rl),
        },
    }
    report_json_path = reports_root / "contamination_report.json"
    report_md_path = reports_root / "contamination_report.md"
    _write_json(report_json_path, contamination_report)
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    report_md_path.write_text(_contamination_markdown(contamination_report), encoding="utf-8")

    excluded_records = [
        *sft_duplicate_exclusions,
        *rl_duplicate_exclusions,
        *_audit_exclusions(audits),
    ]
    excluded_ids_path = manifests_root / "excluded_ids.jsonl"
    _write_jsonl(excluded_ids_path, excluded_records)

    # 正式运行生成 token 报告；smoke test 可显式跳过以缩短耗时。
    quality_report_path: Path | None = None
    quality_report_md_path: Path | None = None
    quality_report: dict[str, Any] | None = None
    if config.generate_quality_report:
        quality_report = build_quality_report(
            {
                "sft_train": sft_train,
                "sft_dev": sft_dev,
                "rl_train": rl_train,
                "rl_dev": rl_dev,
                "medqa_test": medqa_records,
                "medmcqa_validation": medmcqa_records,
                "pubmedqa_labeled": pubmedqa_records,
            },
            seed=config.seed,
        )
        quality_report_path = reports_root / "data_quality_report.json"
        quality_report_md_path = reports_root / "data_quality_report.md"
        _write_json(quality_report_path, quality_report)
        quality_report_md_path.write_text(quality_report_markdown(quality_report), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "english_mainline_m1",
        "seed": config.seed,
        "base_model": {"repository": BASE_MODEL, "revision": BASE_MODEL_REVISION},
        "source_revisions": {key: source.revision for key, source in SOURCES.items()},
        "datasets": {
            "sft": {
                **asdict(SOURCES["huatuo_sft_en"]),
                "rows_before": len(raw_sft_records),
                "rows_after": len(clean_sft),
                "splits": {"train": len(sft_train), "dev": len(sft_dev)},
                "dedup_against": ["self", "medqa", "medmcqa", "pubmedqa"],
            },
            "rl": {
                **asdict(SOURCES["huatuo_verifiable"]),
                "rows_before": len(raw_rl_records),
                "rows_after": len(clean_rl),
                "splits": {"train": len(rl_train), "dev": len(rl_dev)},
                "dedup_against": [
                    "self",
                    "huatuo_sft_en",
                    "medqa",
                    "medmcqa",
                    "pubmedqa",
                ],
            },
            "evaluation": {
                "rows_before": len(medqa_records) + len(medmcqa_records) + len(pubmedqa_records),
                "rows_after": len(medqa_records) + len(medmcqa_records) + len(pubmedqa_records),
                "splits": {
                    "medqa_test": len(medqa_records),
                    "medmcqa_validation": len(medmcqa_records),
                    "pubmedqa_labeled": len(pubmedqa_records),
                },
            },
        },
        "files": file_manifests,
        "aggregate_sha256": _aggregate_sha256(file_manifests),
        "excluded_ids_file": excluded_ids_path.relative_to(repo_root).as_posix(),
        "contamination_report": report_json_path.relative_to(repo_root).as_posix(),
        "data_quality_report": (
            quality_report_path.relative_to(repo_root).as_posix() if quality_report_path else None
        ),
    }
    manifest_path = manifests_root / "english_mainline.json"
    _write_json(manifest_path, manifest)

    return {
        "manifest": str(manifest_path),
        "contamination_report_json": str(report_json_path),
        "contamination_report_markdown": str(report_md_path),
        "aggregate_sha256": manifest["aggregate_sha256"],
        "rows": contamination_report["rows"],
        "quality_report_json": str(quality_report_path) if quality_report_path else None,
        "quality_report_markdown": str(quality_report_md_path) if quality_report_md_path else None,
    }

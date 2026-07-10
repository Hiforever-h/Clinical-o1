"""正式 benchmark 加载、canonical schema 校验与 manifest 哈希门禁。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from medical_grpo.data.schema import to_mcq_sample
from medical_grpo.evaluation.config import EvaluationConfig
from medical_grpo.tracking.artifacts import sha256_file


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取完整 JSONL，并在解析错误中保留文件和行号。"""

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: JSON 解析失败：{exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: 每行必须是 JSON 对象")
            records.append(payload)
    return records


def load_evaluation_datasets(
    config: EvaluationConfig,
    repo_root: Path,
    selected_datasets: tuple[str, ...],
    max_samples_per_dataset: int | None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """校验冻结数据后加载所选数据集，并对每个集合固定截取前 N 题。"""

    repo_root = repo_root.resolve()
    manifest_path = config.resolve_path(repo_root, config.data.manifest_file)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    aggregate = str(manifest.get("aggregate_sha256", ""))
    if aggregate != config.data.expected_aggregate_sha256:
        raise ValueError(
            f"数据 aggregate SHA256 不一致：{aggregate} != {config.data.expected_aggregate_sha256}"
        )

    loaded: dict[str, list[dict[str, Any]]] = {}
    verified: dict[str, Any] = {}
    for dataset_name in selected_datasets:
        dataset_spec = config.data.datasets[dataset_name]
        path = config.resolve_path(repo_root, dataset_spec.path)
        metadata = manifest.get("files", {}).get(dataset_spec.manifest_key)
        if not isinstance(metadata, Mapping):
            raise ValueError(f"manifest 缺少 files.{dataset_spec.manifest_key}")
        expected_path = (repo_root / str(metadata["path"])).resolve()
        if path != expected_path:
            raise ValueError(f"{dataset_name} 路径与 manifest 不一致：{path} != {expected_path}")
        digest = sha256_file(path)
        if digest != metadata.get("sha256"):
            raise ValueError(f"{dataset_name} SHA256 不一致：{digest} != {metadata.get('sha256')}")

        raw_records = _read_jsonl(path)
        if len(raw_records) != int(metadata["rows"]):
            raise ValueError(f"{dataset_name} 行数不一致：{len(raw_records)} != {metadata['rows']}")
        # 每条数据都经过 canonical schema 往返，避免选项键或答案映射异常。
        records = [to_mcq_sample(record).to_dict() for record in raw_records]
        if max_samples_per_dataset is not None:
            records = records[:max_samples_per_dataset]
        if not records:
            raise ValueError(f"{dataset_name} 没有可评测记录")
        ids = [record["id"] for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{dataset_name} 存在重复 ID")
        loaded[dataset_name] = records
        verified[dataset_name] = {
            "path": str(path),
            "manifest_key": dataset_spec.manifest_key,
            "file_sha256": digest,
            "full_rows": len(raw_records),
            "selected_rows": len(records),
            "first_id": ids[0],
            "last_id": ids[-1],
        }
    return loaded, {
        "manifest_path": str(manifest_path),
        "aggregate_sha256": aggregate,
        "max_samples_per_dataset": max_samples_per_dataset,
        "datasets": verified,
    }

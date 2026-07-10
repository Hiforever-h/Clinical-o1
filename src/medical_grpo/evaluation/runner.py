"""评测运行编排、原子分片、断点恢复、合并和指标落盘。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping

from medical_grpo.evaluation.config import EvaluationConfig
from medical_grpo.evaluation.contract import build_evaluation_contract
from medical_grpo.evaluation.data import load_evaluation_datasets
from medical_grpo.evaluation.metrics import summarize_predictions
from medical_grpo.evaluation.modeling import (
    cuda_eval_preflight,
    generate_batch_predictions,
    load_evaluation_model,
)
from medical_grpo.tracking.runtime import write_runtime_snapshot


@dataclass(frozen=True)
class EvaluationOverrides:
    """CLI 对评测数据集、题数、模型和恢复行为的一次性覆盖。"""

    model_type: str
    selected_datasets: tuple[str, ...]
    selected_protocols: tuple[str, ...]
    max_samples_per_dataset: int | None
    adapter_path: Path | None = None
    run_id: str | None = None
    output_root: Path | None = None
    resume: bool = False
    allow_dirty: bool = False


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """原子写入 JSON，避免中断后留下不完整元数据。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    """先完整写入临时 JSONL，再原子替换正式分片或预测文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取已完成分片；恢复时任何损坏都作为硬错误处理。"""

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: 分片 JSON 损坏：{exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: 分片记录必须是对象")
            records.append(payload)
    return records


def _git_is_dirty(repo_root: Path) -> bool:
    """检测所有未提交、暂存和未跟踪文件。"""

    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def _check_disk(path: Path, minimum_free_gb: int) -> dict[str, Any]:
    """检查输出文件系统剩余空间，路径不存在时向上寻找现有父目录。"""

    parent = path.resolve()
    while not parent.exists():
        if parent.parent == parent:
            raise FileNotFoundError(f"无法找到 {path} 的现有父目录")
        parent = parent.parent
    usage = shutil.disk_usage(parent)
    free_gb = usage.free / 1024**3
    if free_gb < minimum_free_gb:
        raise RuntimeError(f"磁盘空间不足：{free_gb:.2f}GB < {minimum_free_gb}GB（{parent}）")
    return {
        "checked_path": str(parent),
        "total_gb": round(usage.total / 1024**3, 2),
        "free_gb": round(free_gb, 2),
        "minimum_free_gb": minimum_free_gb,
    }


def _make_run_id(config: EvaluationConfig, model_type: str) -> str:
    """生成包含模型类型、profile 和 UTC 时间的唯一 run ID。"""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{config.experiment_name}_{model_type}_{config.profile_name}_{timestamp}"


def _validate_existing_shard(
    path: Path,
    expected_records: list[Mapping[str, Any]],
    contract_sha256: str,
) -> list[dict[str, Any]]:
    """恢复时核对分片 ID 顺序和 contract，禁止跳过不兼容结果。"""

    records = _read_jsonl(path)
    expected_ids = [str(record["id"]) for record in expected_records]
    actual_ids = [str(record.get("id")) for record in records]
    if actual_ids != expected_ids:
        raise ValueError(f"分片 ID/顺序不一致：{path}")
    if any(record.get("evaluation_contract_sha256") != contract_sha256 for record in records):
        raise ValueError(f"分片 contract 不一致：{path}")
    return records


def _merge_predictions(
    shard_paths: list[Path],
    destination: Path,
    expected_ids: list[str],
) -> list[dict[str, Any]]:
    """按 shard 顺序合并预测，并验证无缺失、重复或乱序。"""

    merged = [record for path in shard_paths for record in _read_jsonl(path)]
    actual_ids = [str(record["id"]) for record in merged]
    if actual_ids != expected_ids:
        raise ValueError(f"合并预测与原始题目顺序不一致：{destination}")
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError(f"合并预测存在重复 ID：{destination}")
    _write_jsonl(destination, merged)
    return merged


def run_evaluation(
    config: EvaluationConfig,
    repo_root: Path,
    overrides: EvaluationOverrides,
) -> dict[str, Any]:
    """执行 Base/adapter 评测，并生成可恢复、可配对比较的完整 run。"""

    import torch

    repo_root = repo_root.resolve()
    if config.runtime.require_clean_git and not overrides.allow_dirty and _git_is_dirty(repo_root):
        raise RuntimeError("Git 工作区不干净；正式评测请先提交，或仅调试时传 --allow-dirty")
    if overrides.run_id and any(part in overrides.run_id for part in ("/", "\\", "..")):
        raise ValueError("run-id 不能包含路径分隔符或 ..")

    output_root = (
        overrides.output_root.resolve()
        if overrides.output_root is not None
        else config.resolve_path(repo_root, config.runtime.output_root)
    )
    run_id = overrides.run_id or _make_run_id(config, overrides.model_type)
    run_dir = output_root / run_id
    if run_dir.exists() and not overrides.resume:
        raise FileExistsError(f"run 目录已存在；如需继续请传 --resume：{run_dir}")
    if overrides.resume and not run_dir.is_dir():
        raise FileNotFoundError(f"恢复目录不存在：{run_dir}")

    disk_report = _check_disk(output_root, config.runtime.minimum_free_disk_gb)
    hardware_report = cuda_eval_preflight(config)
    datasets, data_report = load_evaluation_datasets(
        config,
        repo_root,
        overrides.selected_datasets,
        overrides.max_samples_per_dataset,
    )
    contract = build_evaluation_contract(
        config,
        overrides.selected_datasets,
        overrides.selected_protocols,
        overrides.max_samples_per_dataset,
        data_report,
    )
    contract_sha256 = str(contract["evaluation_contract_sha256"])

    run_dir.mkdir(parents=True, exist_ok=True)
    contract_path = run_dir / "evaluation_contract.json"
    if overrides.resume:
        existing_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing_contract.get("evaluation_contract_sha256") != contract_sha256:
            raise ValueError("恢复 run 的 evaluation contract 与当前参数不一致")
    else:
        _write_json(contract_path, contract)
        resolved = config.to_dict()
        resolved["cli_overrides"] = {
            "model_type": overrides.model_type,
            "adapter_path": str(overrides.adapter_path) if overrides.adapter_path else None,
            "selected_datasets": list(overrides.selected_datasets),
            "selected_protocols": list(overrides.selected_protocols),
            "max_samples_per_dataset": overrides.max_samples_per_dataset,
            "run_id": overrides.run_id,
        }
        _write_json(run_dir / "config_resolved.json", resolved)
        _write_json(run_dir / "data_manifest_verified.json", data_report)
        _write_json(run_dir / "hardware_preflight.json", hardware_report)
        _write_json(run_dir / "disk_preflight.json", disk_report)
        runtime = write_runtime_snapshot(repo_root, run_dir / "environment.json")
        _write_json(run_dir / "git_state.json", runtime["git"])
        (run_dir / "command.txt").write_text(shlex.join(sys.argv) + "\n", encoding="utf-8")

    model, tokenizer, model_identity = load_evaluation_model(
        config,
        overrides.model_type,
        overrides.adapter_path,
    )
    identity_path = run_dir / "model_identity.json"
    if overrides.resume and identity_path.is_file():
        existing_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing_identity != model_identity:
            raise ValueError("恢复 run 的模型或 adapter 身份与当前加载结果不一致")
    else:
        # 首次模型加载若在写身份前中断，resume 可以安全补写该文件。
        _write_json(run_dir / "model_identity.json", model_identity)

    all_metrics: dict[str, Any] = {}
    for dataset_name in overrides.selected_datasets:
        records = datasets[dataset_name]
        for protocol_name in overrides.selected_protocols:
            protocol = config.protocols[protocol_name]
            shard_dir = run_dir / "shards" / dataset_name / protocol_name
            shard_paths: list[Path] = []
            for shard_start in range(0, len(records), config.inference.shard_size):
                shard_records = records[shard_start : shard_start + config.inference.shard_size]
                shard_path = shard_dir / f"{shard_start:08d}.jsonl"
                shard_paths.append(shard_path)
                if shard_path.is_file():
                    if not overrides.resume:
                        raise FileExistsError(f"新 run 中意外存在分片：{shard_path}")
                    _validate_existing_shard(shard_path, shard_records, contract_sha256)
                    continue
                predictions = generate_batch_predictions(
                    model,
                    tokenizer,
                    shard_records,
                    dataset_name,
                    protocol_name,
                    protocol,
                    config.inference.max_input_length,
                    contract_sha256,
                    batch_id_start=shard_start // protocol.batch_size,
                )
                _write_jsonl(shard_path, predictions)

            prediction_path = run_dir / "predictions" / f"{dataset_name}_{protocol_name}.jsonl"
            merged = _merge_predictions(
                shard_paths,
                prediction_path,
                [str(record["id"]) for record in records],
            )
            metrics = summarize_predictions(merged)
            metrics.update({"dataset": dataset_name, "protocol": protocol_name})
            _write_json(run_dir / "metrics" / f"{dataset_name}_{protocol_name}.json", metrics)
            all_metrics[f"{dataset_name}/{protocol_name}"] = metrics

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "model_identity": model_identity,
        "evaluation_contract_sha256": contract_sha256,
        "selected_datasets": list(overrides.selected_datasets),
        "selected_protocols": list(overrides.selected_protocols),
        "max_samples_per_dataset": overrides.max_samples_per_dataset,
        "metrics": all_metrics,
        "gpu_peak_memory_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "gpu_peak_memory_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(run_dir / "summary.json", summary)
    return summary

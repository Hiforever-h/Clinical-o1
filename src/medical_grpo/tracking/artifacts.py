"""历史训练 ``outputs/`` 的不可变文件清单。"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


HASH_CHUNK_SIZE = 8 * 1024 * 1024
METRIC_FILENAMES = {"metrics.json", "train_results.json", "eval_results.json", "all_results.json"}


def sha256_file(path: Path) -> str:
    """分块计算 SHA256，避免一次把 checkpoint 大文件读入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_if_metric(path: Path) -> dict[str, Any] | None:
    """只抽取常见指标 JSON；损坏或非对象内容不会阻断整个清单。"""

    if path.name not in METRIC_FILENAMES:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def build_artifact_inventory(outputs_dir: Path) -> dict[str, Any]:
    """构造确定性的逐文件清单和整棵目录树指纹。

    ``generated_at`` 不进入树哈希；树哈希只依赖相对路径、大小和文件 SHA256，
    因而可以直接证明重构前后的历史产物是否完全一致。
    """

    outputs_dir = outputs_dir.resolve()
    if not outputs_dir.is_dir():
        raise FileNotFoundError(f"outputs directory does not exist: {outputs_dir}")

    files: list[dict[str, Any]] = []
    metric_snapshots: list[dict[str, Any]] = []
    tree_digest = hashlib.sha256()
    total_bytes = 0

    # 路径排序保证不同机器、不同文件系统枚举顺序下仍得到同一树哈希。
    for path in sorted(item for item in outputs_dir.rglob("*") if item.is_file()):
        relative_path = path.relative_to(outputs_dir).as_posix()
        size = path.stat().st_size
        file_digest = sha256_file(path)
        total_bytes += size
        entry = {"path": relative_path, "bytes": size, "sha256": file_digest}
        files.append(entry)
        tree_digest.update(f"{relative_path}\0{size}\0{file_digest}\n".encode())

        metric_payload = _load_json_if_metric(path)
        if metric_payload is not None:
            metric_snapshots.append({"path": relative_path, "payload": metric_payload})

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "classification": "immutable_historical_outputs",
        "lineage": "legacy_zh",
        "frozen": True,
        "outputs_dir": str(outputs_dir),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "tree_sha256": tree_digest.hexdigest(),
        "metric_snapshots": metric_snapshots,
        "files": files,
    }


def write_artifact_inventory(outputs_dir: Path, destination: Path) -> dict[str, Any]:
    """构造清单并通过临时文件原子替换目标。"""

    inventory = build_artifact_inventory(outputs_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return inventory

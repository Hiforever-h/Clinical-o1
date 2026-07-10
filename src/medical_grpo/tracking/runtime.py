"""为后续每次训练或评测记录可复现的运行环境快照。"""

from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

import torch


TRACKED_PACKAGES = (
    "accelerate",
    "bitsandbytes",
    "datasets",
    "huggingface-hub",
    "numpy",
    "peft",
    "pyyaml",
    "safetensors",
    "scikit-learn",
    "tensorboard",
    "torch",
    "transformers",
    "trl",
)


def _package_versions() -> dict[str, str | None]:
    """读取关键包版本；未安装的可选包显式记录为 ``null``。"""

    versions: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def _git(repo_root: Path, *args: str) -> str | None:
    """只执行只读 Git 命令，失败时返回 ``None`` 而不是修改仓库。"""

    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build_runtime_snapshot(repo_root: Path) -> dict[str, Any]:
    """收集 Python、依赖、加速器和非破坏性的 Git 状态。"""

    repo_root = repo_root.resolve()
    cuda_available = torch.cuda.is_available()
    devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    status = _git(repo_root, "status", "--short") or ""
    return {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": _package_versions(),
        },
        "accelerator": {
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": devices,
            "mps_available": bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ),
        },
        "git": {
            "commit": _git(repo_root, "rev-parse", "HEAD"),
            "branch": _git(repo_root, "branch", "--show-current"),
            "dirty": bool(status),
            "changed_paths": status.splitlines(),
        },
        "process": {
            "cwd": str(Path.cwd()),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        },
    }


def write_runtime_snapshot(repo_root: Path, destination: Path) -> dict[str, Any]:
    """原子写入运行环境快照，供每个 run 目录复用。"""

    snapshot = build_runtime_snapshot(repo_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return snapshot

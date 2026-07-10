"""与当前工作目录无关的仓库路径定位工具。"""

from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """向上查找同时包含 ``pyproject.toml`` 和 ``PLAN.md`` 的仓库根目录。"""

    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "PLAN.md").is_file():
            return candidate
    raise FileNotFoundError(f"无法从 {current} 定位 Clinical-o1 仓库根目录")

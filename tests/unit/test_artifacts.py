"""历史产物清单与目录树哈希的单元测试。"""

from __future__ import annotations

from pathlib import Path

from medical_grpo.tracking.artifacts import build_artifact_inventory


def test_artifact_tree_digest_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    """相同文件树哈希应稳定，任一文件内容变化都必须改变树哈希。"""

    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "a.txt").write_text("alpha", encoding="utf-8")
    nested = outputs / "nested"
    nested.mkdir()
    (nested / "b.txt").write_text("beta", encoding="utf-8")

    first = build_artifact_inventory(outputs)
    second = build_artifact_inventory(outputs)

    assert first["tree_sha256"] == second["tree_sha256"]
    assert first["lineage"] == "legacy_zh"
    assert first["frozen"] is True
    assert first["file_count"] == 2
    assert first["total_bytes"] == 9

    (nested / "b.txt").write_text("changed", encoding="utf-8")
    changed = build_artifact_inventory(outputs)
    assert changed["tree_sha256"] != first["tree_sha256"]

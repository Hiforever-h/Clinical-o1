"""评测 YAML、数据集选择和前 N 题覆盖的单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from medical_grpo.evaluation.config import (
    load_evaluation_config,
    normalize_dataset_selection,
    normalize_protocol_selection,
    resolve_max_samples,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/evaluation/qwen25_7b_mcq_4090.yaml"


def test_dataset_selection_supports_single_multiple_comma_and_all() -> None:
    """数据集选择应固定顺序去重，并兼容用户常用输入形式。"""

    assert normalize_dataset_selection(["medqa"]) == ("medqa",)
    assert normalize_dataset_selection(["pubmedqa", "medqa"]) == ("medqa", "pubmedqa")
    assert normalize_dataset_selection(["medqa,medmcqa"]) == ("medqa", "medmcqa")
    assert normalize_dataset_selection(["all"]) == ("medqa", "medmcqa", "pubmedqa")
    with pytest.raises(ValueError, match="all"):
        normalize_dataset_selection(["all", "medqa"])


def test_profile_and_cli_max_samples_are_per_dataset() -> None:
    """smoke 默认每集合 12 题，CLI 数值应覆盖 profile。"""

    smoke = load_evaluation_config(CONFIG_PATH, profile="smoke")
    full = load_evaluation_config(CONFIG_PATH, profile="full")

    assert resolve_max_samples(smoke, None) == 12
    assert resolve_max_samples(smoke, 5) == 5
    assert resolve_max_samples(full, None) is None
    assert normalize_protocol_selection("both") == ("direct", "cot")

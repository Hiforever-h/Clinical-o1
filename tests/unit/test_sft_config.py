from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from medical_grpo.sft.config import load_sft_config, make_run_id


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/sft/qwen25_7b_qlora_4090.yaml"


def test_smoke_profile_applies_4090_safe_overrides() -> None:
    """smoke 必须减小步数，但不能悄悄改变核心 QLoRA 契约。"""

    config = load_sft_config(CONFIG_PATH, profile="smoke")

    assert config.data.max_length == 2048
    assert config.quantization.quant_type == "nf4"
    assert config.gradient_accumulation_steps == 4
    assert config.effective_global_batch_size == 4
    assert config.eval_steps == config.save_steps == 5
    assert config.profile.max_steps == 10


def test_full_profile_has_stable_run_id() -> None:
    """run ID 同时包含实验、profile、seed 和 UTC 时间。"""

    config = load_sft_config(CONFIG_PATH, profile="full")
    now = datetime(2026, 7, 10, 3, 4, 5, tzinfo=timezone.utc)

    assert make_run_id(config, now) == (
        "qwen25_7b_huatuo_en_qlora_r16_full_seed42_20260710T030405Z"
    )
    assert config.effective_global_batch_size == 16
    assert config.profile.max_train_samples is None

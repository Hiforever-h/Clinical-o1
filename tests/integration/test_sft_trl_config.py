from __future__ import annotations

from pathlib import Path

import pytest

from medical_grpo.sft.config import load_sft_config


pytest.importorskip("trl")

from medical_grpo.sft.trainer import build_trl_sft_config
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_project_config_maps_to_trl_completion_only_training(tmp_path: Path) -> None:
    """防止升级或重构时误把 user prompt 纳入 loss。"""

    config = load_sft_config(
        REPO_ROOT / "configs/sft/qwen25_7b_qlora_4090.yaml",
        profile="smoke",
    )
    trl_config = build_trl_sft_config(
        config,
        tmp_path,
        run_name="test",
        hardware_agnostic=True,
    )

    assert trl_config.max_length == 2048
    assert trl_config.completion_only_loss is True
    assert trl_config.assistant_only_loss is False
    assert trl_config.packing is False
    assert trl_config.max_steps == 10
    assert trl_config.gradient_accumulation_steps == 4
    assert trl_config.eval_steps == trl_config.save_steps == 5


def test_trl_collator_masks_prompt_and_keeps_completion_labels() -> None:
    """直接测试固定 TRL 版本生成的 labels，而不只检查配置布尔值。"""

    collator = DataCollatorForLanguageModeling(
        pad_token_id=0,
        completion_only_loss=True,
    )
    batch = collator(
        [
            {
                "input_ids": [10, 11, 12, 13],
                "completion_mask": [0, 0, 1, 1],
            }
        ]
    )

    assert batch["labels"].tolist() == [[-100, -100, 12, 13]]

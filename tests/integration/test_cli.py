"""统一 CLI 子命令与关键参数解析的集成测试。"""

from __future__ import annotations

from medical_grpo.cli import build_parser


def test_cli_exposes_data_and_sft_commands() -> None:
    """所有已实现阶段都应从同一个 parser 进入并保留参数值。"""

    parser = build_parser()
    inventory = parser.parse_args(["inventory"])
    prepare_data = parser.parse_args(["prepare-data", "--max-sft-samples", "10", "--max-rl-samples", "20"])
    dry_run = parser.parse_args(["dry-run", "--component", "sft"])
    snapshot = parser.parse_args(["snapshot-runtime"])
    # Stage 3 两个入口分别覆盖无 GPU 检查和正式训练参数。
    sft_dry_run = parser.parse_args(["sft-dry-run", "--profile", "smoke"])
    train_sft = parser.parse_args(
        [
            "train-sft",
            "--profile",
            "pilot",
            "--run-id",
            "pilot-test",
            "--resume-from-checkpoint",
            "latest",
        ]
    )
    eval_dry_run = parser.parse_args(
        ["eval-dry-run", "--datasets", "medqa", "pubmedqa", "--max-samples", "20"]
    )
    evaluate = parser.parse_args(
        [
            "evaluate",
            "--model-type",
            "base",
            "--datasets",
            "medmcqa",
            "--protocol",
            "direct",
            "--max-samples",
            "50",
        ]
    )
    compare_eval = parser.parse_args(
        ["compare-eval", "--baseline", "base-run", "--candidate", "sft-run"]
    )

    assert inventory.command == "inventory"
    assert prepare_data.command == "prepare-data"
    assert prepare_data.max_sft_samples == 10
    assert prepare_data.max_rl_samples == 20
    assert dry_run.command == "dry-run"
    assert dry_run.component == "sft"
    assert snapshot.command == "snapshot-runtime"
    assert sft_dry_run.command == "sft-dry-run"
    assert sft_dry_run.profile == "smoke"
    assert train_sft.command == "train-sft"
    assert train_sft.profile == "pilot"
    assert train_sft.run_id == "pilot-test"
    assert train_sft.resume_from_checkpoint == "latest"
    assert eval_dry_run.datasets == ["medqa", "pubmedqa"]
    assert eval_dry_run.max_samples == 20
    assert evaluate.model_type == "base"
    assert evaluate.datasets == ["medmcqa"]
    assert evaluate.protocol == "direct"
    assert evaluate.max_samples == 50
    assert compare_eval.command == "compare-eval"

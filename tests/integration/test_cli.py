from __future__ import annotations

from medical_grpo.cli import build_parser


def test_cli_exposes_data_and_sft_commands() -> None:
    parser = build_parser()
    inventory = parser.parse_args(["inventory"])
    prepare_data = parser.parse_args(["prepare-data", "--max-sft-samples", "10", "--max-rl-samples", "20"])
    dry_run = parser.parse_args(["dry-run", "--component", "sft"])
    snapshot = parser.parse_args(["snapshot-runtime"])
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

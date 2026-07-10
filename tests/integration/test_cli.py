from __future__ import annotations

from medical_grpo.cli import build_parser


def test_cli_exposes_m0_m1_commands() -> None:
    parser = build_parser()
    inventory = parser.parse_args(["inventory"])
    prepare_data = parser.parse_args(["prepare-data", "--max-sft-samples", "10", "--max-rl-samples", "20"])
    dry_run = parser.parse_args(["dry-run", "--component", "sft"])
    snapshot = parser.parse_args(["snapshot-runtime"])

    assert inventory.command == "inventory"
    assert prepare_data.command == "prepare-data"
    assert prepare_data.max_sft_samples == 10
    assert prepare_data.max_rl_samples == 20
    assert dry_run.command == "dry-run"
    assert dry_run.component == "sft"
    assert snapshot.command == "snapshot-runtime"

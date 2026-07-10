"""Clinical-o1 的统一命令行入口。

脚本目录已经移除，所有已验收能力都从这里进入，避免出现多套参数和输出目录。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from medical_grpo.paths import find_repo_root
from medical_grpo.tracking.artifacts import write_artifact_inventory


DEFAULT_SFT_CONFIG = Path("configs/sft/qwen25_7b_qlora_4090.yaml")


def _add_inventory_command(subparsers: argparse._SubParsersAction) -> None:
    """注册历史 outputs 冻结清单命令。"""

    parser = subparsers.add_parser("inventory", help="逐文件冻结历史 outputs 的大小和 SHA256。")
    parser.add_argument("--outputs-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.set_defaults(command_func=_run_inventory)


def _add_prepare_data_command(subparsers: argparse._SubParsersAction) -> None:
    """注册英文 SFT/RL/评测数据准备命令。"""

    parser = subparsers.add_parser("prepare-data", help="准备英文数据并执行去重和污染审计。")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sft-dev-ratio", type=float, default=0.02)
    parser.add_argument("--rl-dev-ratio", type=float, default=0.02)
    parser.add_argument("--max-sft-samples", type=int, default=None)
    parser.add_argument("--max-rl-samples", type=int, default=None)
    parser.add_argument("--fuzzy-review-threshold", type=float, default=0.82)
    parser.add_argument("--fuzzy-exclude-threshold", type=float, default=0.93)
    parser.add_argument(
        "--skip-quality-report",
        action="store_true",
        help="跳过 Qwen tokenizer 长度统计，仅用于快速 smoke test。",
    )
    parser.set_defaults(command_func=_run_prepare_data)


def _add_dry_run_command(subparsers: argparse._SubParsersAction) -> None:
    """注册不加载 7B 模型的 M0/M1 合同检查命令。"""

    parser = subparsers.add_parser(
        "dry-run", help="运行当前已完成的 M0/M1 数据、SFT、RL 与评测合同检查。"
    )
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument(
        "--component", choices=("all", "data", "sft", "rl", "eval"), default="all"
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.set_defaults(command_func=_run_dry_run)


def _add_snapshot_command(subparsers: argparse._SubParsersAction) -> None:
    """注册运行环境和 Git 状态快照命令。"""

    parser = subparsers.add_parser(
        "snapshot-runtime", help="记录环境、加速器和 Git 状态，供实验复现。"
    )
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.set_defaults(command_func=_run_snapshot)


def _add_sft_dry_run_command(subparsers: argparse._SubParsersAction) -> None:
    """注册不加载 7B 权重的 Stage 3 SFT 检查命令。"""

    parser = subparsers.add_parser(
        "sft-dry-run",
        help="校验英文 SFT 数据哈希、token 边界和 TRL 参数映射。",
    )
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_SFT_CONFIG)
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="full")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.set_defaults(command_func=_run_sft_dry_run)


def _add_train_sft_command(subparsers: argparse._SubParsersAction) -> None:
    """注册 RTX 4090 单卡 QLoRA SFT 训练命令。"""

    parser = subparsers.add_parser(
        "train-sft",
        help="执行 smoke、pilot 或 full 单卡 QLoRA SFT。",
    )
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_SFT_CONFIG)
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), required=True)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="传 latest，或传相对于 run 目录的 checkpoint 路径。",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="仅调试时允许未提交代码；正式实验不要使用。",
    )
    parser.add_argument(
        "--skip-generations",
        action="store_true",
        help="跳过 SFT 前后固定样本生成，仅用于定位训练问题。",
    )
    parser.set_defaults(command_func=_run_train_sft)


def build_parser() -> argparse.ArgumentParser:
    """构造顶层解析器；子命令必须显式指定。"""

    parser = argparse.ArgumentParser(prog="clinical-o1", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_inventory_command(subparsers)
    _add_prepare_data_command(subparsers)
    _add_dry_run_command(subparsers)
    _add_snapshot_command(subparsers)
    _add_sft_dry_run_command(subparsers)
    _add_train_sft_command(subparsers)
    return parser


def _run_inventory(args: argparse.Namespace) -> int:
    """执行不可变产物清单生成。"""

    repo_root = find_repo_root()
    outputs_dir = (args.outputs_dir or repo_root / "outputs").resolve()
    output = (args.output or repo_root / "reports" / "artifact_inventory.json").resolve()
    inventory = write_artifact_inventory(outputs_dir, output)
    print(f"Inventory: {output}")
    print(f"Files: {inventory['file_count']} | Bytes: {inventory['total_bytes']}")
    print(f"Tree SHA256: {inventory['tree_sha256']}")
    return 0


def _run_prepare_data(args: argparse.Namespace) -> int:
    """延迟导入较重的数据依赖并执行完整 M1 管线。"""

    from medical_grpo.data.pipeline import DataPreparationConfig, prepare_english_mainline

    repo_root = (args.repo_root or find_repo_root()).resolve()
    config = DataPreparationConfig(
        repo_root=repo_root,
        seed=args.seed,
        sft_dev_ratio=args.sft_dev_ratio,
        rl_dev_ratio=args.rl_dev_ratio,
        max_sft_samples=args.max_sft_samples,
        max_rl_samples=args.max_rl_samples,
        fuzzy_review_threshold=args.fuzzy_review_threshold,
        fuzzy_exclude_threshold=args.fuzzy_exclude_threshold,
        generate_quality_report=not args.skip_quality_report,
    )
    summary = prepare_english_mainline(config)
    print(f"Prepared English mainline data under {repo_root / 'data' / 'processed'}")
    print(f"Contamination report: {summary['contamination_report_json']}")
    return 0


def _run_dry_run(args: argparse.Namespace) -> int:
    """执行合同检查，并可用临时文件原子写入 JSON 报告。"""

    from medical_grpo.dry_run import run_dry_checks

    repo_root = (args.repo_root or find_repo_root()).resolve()
    results = run_dry_checks(repo_root, component=args.component)
    if args.output is not None:
        output = args.output if args.output.is_absolute() else repo_root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


def _run_snapshot(args: argparse.Namespace) -> int:
    """保存当前机器和代码状态，不修改 Git 工作区。"""

    from medical_grpo.tracking.runtime import write_runtime_snapshot

    repo_root = (args.repo_root or find_repo_root()).resolve()
    output = (args.output or repo_root / "reports" / "runtime_environment.json").resolve()
    snapshot = write_runtime_snapshot(repo_root, output)
    print(f"Runtime snapshot: {output}")
    print(f"Git: {snapshot['git']['branch']} @ {snapshot['git']['commit']}")
    return 0


def _resolve_cli_path(repo_root: Path, path: Path) -> Path:
    """将 CLI 的相对路径统一解释为相对于仓库根目录。"""

    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _run_sft_dry_run(args: argparse.Namespace) -> int:
    """执行 Stage 3 数据、tokenizer 与 TRL 配置 dry-run。"""

    from medical_grpo.sft.config import load_sft_config
    from medical_grpo.sft.dry_run import run_sft_dry_run, write_dry_run_report

    repo_root = (args.repo_root or find_repo_root()).resolve()
    config_path = _resolve_cli_path(repo_root, args.config)
    config = load_sft_config(config_path, profile=args.profile)
    report = run_sft_dry_run(
        config,
        repo_root,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    output = args.output or Path(f"reports/sft/{args.profile}_dry_run.json")
    output = _resolve_cli_path(repo_root, output)
    write_dry_run_report(output, report)
    print(
        "SFT dry-run: "
        f"status={report['status']} profile={report['profile']} "
        f"train={report['rows']['train']} eval={report['rows']['eval']}"
    )
    print(
        "Token audit: "
        f"train_max={report['token_audit']['train']['max_input_tokens']} "
        f"eval_max={report['token_audit']['eval']['max_input_tokens']} "
        f"truncated={report['token_audit']['train']['truncated_rows'] + report['token_audit']['eval']['truncated_rows']}"
    )
    print(
        "TRL labels: "
        f"prompt_masked={report['trl_label_audit']['prompt_labels_all_minus_100']} "
        f"completion_supervised={report['trl_label_audit']['completion_labels_all_supervised']}"
    )
    print(f"SFT dry-run report: {output}")
    return 0


def _run_train_sft(args: argparse.Namespace) -> int:
    """延迟加载 GPU 依赖并执行 Stage 3 QLoRA SFT。"""

    from medical_grpo.sft.config import load_sft_config
    from medical_grpo.sft.trainer import TrainOverrides, run_sft_training

    repo_root = (args.repo_root or find_repo_root()).resolve()
    config_path = _resolve_cli_path(repo_root, args.config)
    output_root = (
        _resolve_cli_path(repo_root, args.output_root)
        if args.output_root is not None
        else None
    )
    config = load_sft_config(config_path, profile=args.profile)
    result = run_sft_training(
        config,
        repo_root,
        TrainOverrides(
            run_id=args.run_id,
            output_root=output_root,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
            max_steps=args.max_steps,
            resume_from_checkpoint=args.resume_from_checkpoint,
            allow_dirty=args.allow_dirty,
            skip_generations=args.skip_generations,
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 主入口，返回进程退出码。"""

    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.command_func(args))

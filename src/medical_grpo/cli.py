"""Clinical-o1 的统一命令行入口。

脚本目录已经移除，所有已验收能力都从这里进入，避免出现多套参数和输出目录。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from medical_grpo.paths import find_repo_root
from medical_grpo.tracking.artifacts import write_artifact_inventory


# 所有 SFT 子命令默认读取同一份 4090 基线，避免 CLI 与文档参数分叉。
DEFAULT_SFT_CONFIG = Path("configs/sft/qwen25_7b_qlora_4090.yaml")
# Stage 2 与所有后续模型共用这一份评测合同配置。
DEFAULT_EVAL_CONFIG = Path("configs/evaluation/qwen25_7b_mcq_4090.yaml")


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
    # dry-run 支持缩小样本数，但默认按所选 profile 审计对应数据规模。
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
    # profile 必须显式选择，防止无意间直接启动成本最高的 full run。
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_SFT_CONFIG)
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), required=True)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    # 样本/步数覆盖主要服务 smoke 与故障定位，正式 full 使用 YAML 默认值。
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


def _add_eval_dry_run_command(subparsers: argparse._SubParsersAction) -> None:
    """注册不加载 7B 权重的评测合同检查命令。"""

    parser = subparsers.add_parser(
        "eval-dry-run",
        help="检查所选 benchmark、前 N 题、Prompt、解析器和输入 token 长度。",
    )
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_EVAL_CONFIG)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help="可选 all、medqa、medmcqa、pubmedqa；多个名称用空格或逗号分隔。",
    )
    parser.add_argument("--protocol", choices=("direct", "cot", "both"), default="both")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="每个被选数据集只取固定前 N 题；不传则使用 profile 默认值。",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.set_defaults(command_func=_run_eval_dry_run)


def _add_evaluate_command(subparsers: argparse._SubParsersAction) -> None:
    """注册 Base 或 LoRA adapter 的正式 GPU 评测命令。"""

    parser = subparsers.add_parser(
        "evaluate",
        help="在所选 benchmark 的全部或前 N 题上执行可恢复评测。",
    )
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_EVAL_CONFIG)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--model-type", choices=("base", "adapter"), required=True)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help="可选 all、medqa、medmcqa、pubmedqa；多个名称用空格或逗号分隔。",
    )
    parser.add_argument("--protocol", choices=("direct", "cot", "both"), default="both")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="每个被选数据集只评测固定前 N 题。",
    )
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", help="恢复同一 run 中已完成的原子分片。")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="仅调试时允许未提交代码；正式评测不要使用。",
    )
    parser.set_defaults(command_func=_run_evaluate)


def _add_compare_eval_command(subparsers: argparse._SubParsersAction) -> None:
    """注册两个相同 contract 评测 run 的配对比较命令。"""

    parser = subparsers.add_parser(
        "compare-eval",
        help="比较 Base 与候选模型的逐题正确性变化和 McNemar 统计。",
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.set_defaults(command_func=_run_compare_eval)


def build_parser() -> argparse.ArgumentParser:
    """构造顶层解析器；子命令必须显式指定。"""

    parser = argparse.ArgumentParser(prog="clinical-o1", description=__doc__)
    # required=True 可避免用户漏写子命令时静默退出或执行错误默认操作。
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_inventory_command(subparsers)
    _add_prepare_data_command(subparsers)
    _add_dry_run_command(subparsers)
    _add_snapshot_command(subparsers)
    _add_sft_dry_run_command(subparsers)
    _add_train_sft_command(subparsers)
    _add_eval_dry_run_command(subparsers)
    _add_evaluate_command(subparsers)
    _add_compare_eval_command(subparsers)
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

    # 延迟导入 SFT 依赖，使 M0/M1 基础命令不强制安装完整训练栈。
    repo_root = (args.repo_root or find_repo_root()).resolve()
    config_path = _resolve_cli_path(repo_root, args.config)
    config = load_sft_config(config_path, profile=args.profile)
    report = run_sft_dry_run(
        config,
        repo_root,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    # 报告完整落盘，终端只打印最关键的行数、长度和 label mask 结果。
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

    # 在真正执行子命令前才导入 TRL/PEFT，普通数据命令可保持轻量启动。
    repo_root = (args.repo_root or find_repo_root()).resolve()
    config_path = _resolve_cli_path(repo_root, args.config)
    output_root = (
        _resolve_cli_path(repo_root, args.output_root)
        if args.output_root is not None
        else None
    )
    config = load_sft_config(config_path, profile=args.profile)
    # CLI 参数先封装为不可变 overrides，训练编排统一负责合法性校验。
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


def _resolve_eval_selection(args: argparse.Namespace, config: Any) -> tuple[tuple[str, ...], tuple[str, ...], int | None]:
    """统一解析数据集、协议和每数据集前 N 题限制。"""

    from medical_grpo.evaluation.config import (
        normalize_dataset_selection,
        normalize_protocol_selection,
        resolve_max_samples,
    )

    datasets = normalize_dataset_selection(args.datasets)
    protocols = normalize_protocol_selection(args.protocol)
    max_samples = resolve_max_samples(config, args.max_samples)
    return datasets, protocols, max_samples


def _run_eval_dry_run(args: argparse.Namespace) -> int:
    """执行评测数据、Prompt、解析器和输入长度的 CPU dry-run。"""

    from medical_grpo.evaluation.config import load_evaluation_config
    from medical_grpo.evaluation.dry_run import run_evaluation_dry_run, write_evaluation_dry_run

    repo_root = (args.repo_root or find_repo_root()).resolve()
    config = load_evaluation_config(_resolve_cli_path(repo_root, args.config), profile=args.profile)
    datasets, protocols, max_samples = _resolve_eval_selection(args, config)
    report = run_evaluation_dry_run(config, repo_root, datasets, protocols, max_samples)
    dataset_label = "-".join(datasets)
    limit_label = f"n{max_samples}" if max_samples is not None else "all"
    output = args.output or Path(
        f"reports/evaluation/{args.profile}_{dataset_label}_{args.protocol}_{limit_label}_dry_run.json"
    )
    output = _resolve_cli_path(repo_root, output)
    write_evaluation_dry_run(output, report)
    print(
        f"Eval dry-run: datasets={','.join(datasets)} protocols={','.join(protocols)} "
        f"max_samples_per_dataset={max_samples} predictions={report['total_predictions']}"
    )
    print(f"Contract: {report['evaluation_contract']['evaluation_contract_sha256']}")
    print(f"Report: {output}")
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    """延迟加载 GPU 依赖并执行 Base/adapter 正式评测。"""

    from medical_grpo.evaluation.config import load_evaluation_config
    from medical_grpo.evaluation.runner import EvaluationOverrides, run_evaluation

    repo_root = (args.repo_root or find_repo_root()).resolve()
    config = load_evaluation_config(_resolve_cli_path(repo_root, args.config), profile=args.profile)
    datasets, protocols, max_samples = _resolve_eval_selection(args, config)
    adapter_path = (
        _resolve_cli_path(repo_root, args.adapter_path)
        if args.adapter_path is not None
        else None
    )
    output_root = (
        _resolve_cli_path(repo_root, args.output_root)
        if args.output_root is not None
        else None
    )
    result = run_evaluation(
        config,
        repo_root,
        EvaluationOverrides(
            model_type=args.model_type,
            adapter_path=adapter_path,
            selected_datasets=datasets,
            selected_protocols=protocols,
            max_samples_per_dataset=max_samples,
            run_id=args.run_id,
            output_root=output_root,
            resume=args.resume,
            allow_dirty=args.allow_dirty,
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _run_compare_eval(args: argparse.Namespace) -> int:
    """执行相同 contract 的 Base/候选模型配对比较。"""

    from medical_grpo.evaluation.compare import compare_evaluation_runs, write_comparison

    repo_root = find_repo_root()
    baseline = _resolve_cli_path(repo_root, args.baseline)
    candidate = _resolve_cli_path(repo_root, args.candidate)
    report = compare_evaluation_runs(baseline, candidate)
    output = args.output or candidate / "comparison_to_baseline.json"
    output = _resolve_cli_path(repo_root, output)
    write_comparison(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Comparison report: {output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 主入口，返回进程退出码。"""

    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.command_func(args))

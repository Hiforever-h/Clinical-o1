"""不加载 7B 权重的 SFT tokenizer、mask 和配置 dry-run。"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
from typing import Any

from medical_grpo.sft.config import SFTExperimentConfig
from medical_grpo.sft.data import (
    audit_token_boundaries,
    build_hf_dataset,
    load_sft_records,
    verify_data_manifest,
)
from medical_grpo.sft.modeling import load_tokenizer
from medical_grpo.sft.trainer import build_trl_sft_config


def _version(package: str) -> str | None:
    """读取包版本；可选依赖缺失时记录 null 而不是中止检查。"""

    try:
        return version(package)
    except PackageNotFoundError:
        return None


def verify_trl_completion_labels(
    records: list[dict[str, Any]],
    tokenizer: Any,
    sample_size: int = 8,
) -> dict[str, Any]:
    """用 TRL 真实 collator 验证 prompt label 全为 -100。"""

    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    # 按 TRL 内部格式手工构造 input_ids 与 completion_mask。
    examples: list[dict[str, list[int]]] = []
    prompt_lengths: list[int] = []
    full_lengths: list[int] = []
    for record in records[: min(sample_size, len(records))]:
        prompt = [{"role": "user", "content": record["question"]}]
        completion = [record["messages"][1]]
        prompt_ids = tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
        )
        full_ids = tokenizer.apply_chat_template(
            prompt + completion,
            tokenize=True,
            add_generation_prompt=False,
        )
        # 若 prompt 不是完整对话的严格前缀，mask 切点就不可信，必须停止。
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(f"{record['id']}: TRL label 审计发现 prompt 前缀不匹配")
        # mask=0 对应 user prompt，mask=1 对应 assistant completion。
        examples.append(
            {
                "input_ids": full_ids,
                "completion_mask": [0] * len(prompt_ids)
                + [1] * (len(full_ids) - len(prompt_ids)),
            }
        )
        prompt_lengths.append(len(prompt_ids))
        full_lengths.append(len(full_ids))

    # 使用固定 TRL 版本的真实 collator，而不是只验证项目自己的模拟逻辑。
    collator = DataCollatorForLanguageModeling(
        pad_token_id=tokenizer.pad_token_id,
        completion_only_loss=True,
    )
    batch = collator(examples)
    supervised_tokens = 0
    for row, (prompt_length, full_length) in enumerate(
        zip(prompt_lengths, full_lengths, strict=True)
    ):
        labels = batch["labels"][row]
        # completion-only loss 要求所有 prompt label 都被替换成忽略值 -100。
        if not bool((labels[:prompt_length] == -100).all()):
            raise ValueError(f"row={row}: prompt token 未被 TRL collator 屏蔽")
        completion_labels = labels[prompt_length:full_length]
        # completion 区域必须全部保留监督信号，不能被错误 mask。
        if completion_labels.numel() == 0 or bool((completion_labels == -100).any()):
            raise ValueError(f"row={row}: completion label 被意外屏蔽")
        supervised_tokens += int(completion_labels.numel())
    return {
        "rows": len(examples),
        "prompt_labels_all_minus_100": True,
        "completion_labels_all_supervised": True,
        "supervised_tokens": supervised_tokens,
    }


def run_sft_dry_run(
    config: SFTExperimentConfig,
    repo_root: Path,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> dict[str, Any]:
    """验证正式数据、chat template、监督边界和 TRL 参数映射。"""

    repo_root = repo_root.resolve()
    # dry-run 与正式训练共用相同路径解析和数据哈希门禁。
    train_path = config.resolve_path(repo_root, config.data.train_file)
    eval_path = config.resolve_path(repo_root, config.data.eval_file)
    manifest_path = config.resolve_path(repo_root, config.data.manifest_file)
    manifest_report = verify_data_manifest(
        repo_root,
        manifest_path,
        config.data.expected_aggregate_sha256,
        train_path,
        eval_path,
    )
    # CLI 显式限制优先于 profile，方便快速检查自定义的小样本规模。
    effective_train_limit = (
        max_train_samples
        if max_train_samples is not None
        else config.profile.max_train_samples
    )
    effective_eval_limit = (
        max_eval_samples
        if max_eval_samples is not None
        else config.profile.max_eval_samples
    )
    train_records = load_sft_records(train_path, effective_train_limit)
    eval_records = load_sft_records(eval_path, effective_eval_limit)
    tokenizer = load_tokenizer(config)
    # 全量审计序列长度与 chat-template 前缀，但不加载 7B 模型权重。
    train_audit = audit_token_boundaries(
        train_records,
        tokenizer,
        config.data.max_length,
        inspect_samples=config.data.audit_sample_size,
    )
    eval_audit = audit_token_boundaries(
        eval_records,
        tokenizer,
        config.data.max_length,
        inspect_samples=min(config.data.audit_sample_size, len(eval_records)),
    )
    # 构造 Dataset 以验证列结构，并进一步交给 TRL collator 检查 labels。
    train_dataset = build_hf_dataset(train_records)
    eval_dataset = build_hf_dataset(eval_records)
    trl_label_audit = verify_trl_completion_labels(train_records, tokenizer)
    dry_run_dir = repo_root / ".dry-run/sft"
    # 本地 CPU/MPS 关闭 BF16/TF32 硬件校验，其余训练参数映射保持不变。
    trl_config = build_trl_sft_config(
        config,
        dry_run_dir,
        run_name="sft-dry-run",
        hardware_agnostic=True,
    )
    # smoke 使用显式 max_steps；pilot/full 按样本数、batch 和 epoch 推导。
    if config.profile.max_steps > 0:
        expected_steps = config.profile.max_steps
    else:
        updates_per_epoch = math.ceil(
            len(train_records) / config.effective_global_batch_size
        )
        expected_steps = math.ceil(updates_per_epoch * config.training.num_train_epochs)
    return {
        "status": "ok",
        "profile": config.profile_name,
        "manifest": manifest_report,
        "rows": {"train": len(train_records), "eval": len(eval_records)},
        "datasets": {
            "train_columns": train_dataset.column_names,
            "eval_columns": eval_dataset.column_names,
        },
        "token_audit": {
            "train": train_audit.to_dict(),
            "eval": eval_audit.to_dict(),
        },
        "trl_label_audit": trl_label_audit,
        "training_plan": {
            "micro_batch_size": config.training.per_device_train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "global_batch_size": config.effective_global_batch_size,
            "expected_optimizer_steps": expected_steps,
            "max_length": trl_config.max_length,
            "completion_only_loss": trl_config.completion_only_loss,
            "assistant_only_loss": trl_config.assistant_only_loss,
            "packing": trl_config.packing,
            "learning_rate": trl_config.learning_rate,
            "target_bf16": config.training.bf16,
            "target_tf32": config.training.tf32,
            "trl_config_hardware_agnostic": True,
        },
        "versions": {
            package: _version(package)
            for package in (
                "torch",
                "transformers",
                "datasets",
                "accelerate",
                "peft",
                "trl",
                "bitsandbytes",
            )
        },
    }


def write_dry_run_report(path: Path, report: dict[str, Any]) -> None:
    """原子保存 dry-run 报告。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    # 原子替换保证自动化读取方不会读到写了一半的 JSON。
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)

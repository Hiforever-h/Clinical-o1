"""Stage 3 QLoRA SFT 的训练编排、日志和产物落盘。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any, Mapping

from medical_grpo.sft.config import SFTExperimentConfig, make_run_id
from medical_grpo.sft.data import (
    audit_token_boundaries,
    build_hf_dataset,
    load_sft_records,
    verify_data_manifest,
)
from medical_grpo.sft.diagnostics import generate_diagnostic_samples, write_jsonl
from medical_grpo.sft.modeling import build_qlora_model, cuda_preflight, load_tokenizer
from medical_grpo.tracking.runtime import write_runtime_snapshot


@dataclass(frozen=True)
class TrainOverrides:
    run_id: str | None = None
    output_root: Path | None = None
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    max_steps: int | None = None
    resume_from_checkpoint: str | None = None
    allow_dirty: bool = False
    skip_generations: bool = False


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_text(path: Path, content: str) -> None:
    """原子写入短文本，供命令与人工备注等追踪文件复用。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def _git_is_dirty(repo_root: Path) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def _nearest_existing_parent(path: Path) -> Path:
    current = path.resolve()
    while not current.exists():
        if current.parent == current:
            raise FileNotFoundError(f"无法找到 {path} 的现有父目录")
        current = current.parent
    return current


def _check_disk(path: Path, minimum_free_gb: int) -> dict[str, Any]:
    parent = _nearest_existing_parent(path)
    usage = shutil.disk_usage(parent)
    free_gb = usage.free / 1024**3
    if free_gb < minimum_free_gb:
        raise RuntimeError(f"磁盘空间不足：{free_gb:.2f}GB < {minimum_free_gb}GB（{parent}）")
    return {
        "checked_path": str(parent),
        "total_gb": round(usage.total / 1024**3, 2),
        "free_gb": round(free_gb, 2),
        "minimum_free_gb": minimum_free_gb,
    }


def _resolve_resume(run_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    if value == "latest":
        checkpoints = sorted(
            (run_dir / "checkpoints").glob("checkpoint-*"),
            key=lambda path: int(path.name.rsplit("-", maxsplit=1)[-1]),
        )
        if not checkpoints:
            raise FileNotFoundError(f"没有可恢复 checkpoint：{run_dir / 'checkpoints'}")
        return checkpoints[-1]
    checkpoint = Path(value)
    if not checkpoint.is_absolute():
        checkpoint = (run_dir / checkpoint).resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint 不存在：{checkpoint}")
    return checkpoint


def build_trl_sft_config(
    config: SFTExperimentConfig,
    run_dir: Path,
    run_name: str,
    max_steps: int | None = None,
    hardware_agnostic: bool = False,
) -> Any:
    """将项目配置映射到固定版本 TRL 0.29.1 的 SFTConfig。

    ``hardware_agnostic`` 只供 CPU/MPS dry-run 使用，绕过 Transformers 对
    BF16/TF32 的本机硬件校验；正式训练始终保留目标 4090 参数。
    """

    from trl import SFTConfig

    effective_max_steps = config.profile.max_steps if max_steps is None else max_steps
    return SFTConfig(
        output_dir=str(run_dir / "checkpoints"),
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        num_train_epochs=config.training.num_train_epochs,
        max_steps=effective_max_steps,
        lr_scheduler_type=config.training.lr_scheduler_type,
        warmup_ratio=config.training.warmup_ratio,
        logging_steps=config.training.logging_steps,
        logging_first_step=True,
        logging_nan_inf_filter=False,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=config.training.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_safetensors=True,
        save_only_model=False,
        restore_callback_states_from_checkpoint=True,
        seed=config.training.seed,
        data_seed=config.training.data_seed,
        bf16=False if hardware_agnostic else config.training.bf16,
        fp16=False if hardware_agnostic else config.training.fp16,
        bf16_full_eval=False if hardware_agnostic else config.training.bf16,
        tf32=False if hardware_agnostic else config.training.tf32,
        optim=config.training.optim,
        gradient_checkpointing=config.training.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": config.training.use_reentrant},
        dataloader_num_workers=config.training.dataloader_num_workers,
        dataloader_prefetch_factor=2 if config.training.dataloader_num_workers > 0 else None,
        dataloader_persistent_workers=config.training.dataloader_num_workers > 0,
        dataloader_pin_memory=True,
        report_to=config.training.report_to or [],
        logging_dir=str(run_dir / "tensorboard"),
        run_name=run_name,
        include_tokens_per_second=True,
        include_num_input_tokens_seen=True,
        eval_on_start=config.training.eval_on_start,
        max_length=config.data.max_length,
        packing=config.data.packing,
        eval_packing=False,
        completion_only_loss=config.data.completion_only_loss,
        assistant_only_loss=config.data.assistant_only_loss,
        dataset_num_proc=min(4, config.training.dataloader_num_workers or 1),
        shuffle_dataset=True,
        remove_unused_columns=True,
    )


def _metrics_callback(path: Path) -> Any:
    """构造 JSONL 日志 callback，并在首次非有限 loss 时中止训练。"""

    import torch
    from transformers import TrainerCallback

    class JsonlMetricsCallback(TrainerCallback):
        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **_: Any) -> None:
            payload = {
                "logged_at": datetime.now(timezone.utc).isoformat(),
                "step": state.global_step,
                "epoch": state.epoch,
                **(logs or {}),
            }
            if torch.cuda.is_available():
                payload["gpu_max_memory_allocated_gb"] = round(
                    torch.cuda.max_memory_allocated() / 1024**3, 3
                )
                payload["gpu_max_memory_reserved_gb"] = round(
                    torch.cuda.max_memory_reserved() / 1024**3, 3
                )
            for key in ("loss", "eval_loss", "grad_norm"):
                value = payload.get(key)
                if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                    raise RuntimeError(f"step={state.global_step} 出现非有限 {key}={value}")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    return JsonlMetricsCallback()


def run_sft_training(
    config: SFTExperimentConfig,
    repo_root: Path,
    overrides: TrainOverrides | None = None,
) -> dict[str, Any]:
    """执行单卡 QLoRA SFT，并保存可复现的完整运行目录。"""

    from trl import SFTTrainer

    override = overrides or TrainOverrides()
    repo_root = repo_root.resolve()
    if override.run_id is not None and not re.fullmatch(r"[A-Za-z0-9._-]+", override.run_id):
        raise ValueError("run-id 只能包含英文字母、数字、点、下划线和连字符")
    if override.max_steps is not None and override.max_steps != -1 and override.max_steps <= 0:
        raise ValueError("--max-steps 必须为 -1 或正整数")
    for name, value in (
        ("--max-train-samples", override.max_train_samples),
        ("--max-eval-samples", override.max_eval_samples),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"{name} 必须为正整数")
    train_path = config.resolve_path(repo_root, config.data.train_file)
    eval_path = config.resolve_path(repo_root, config.data.eval_file)
    manifest_path = config.resolve_path(repo_root, config.data.manifest_file)
    output_root = (
        override.output_root.resolve()
        if override.output_root is not None
        else config.resolve_path(repo_root, config.runtime.output_root)
    )
    run_id = override.run_id or make_run_id(config)
    run_dir = output_root / run_id
    resume_checkpoint = _resolve_resume(run_dir, override.resume_from_checkpoint)
    if run_dir.exists() and resume_checkpoint is None:
        raise FileExistsError(f"run 目录已存在，禁止覆盖：{run_dir}")
    if config.runtime.require_clean_git and not override.allow_dirty and _git_is_dirty(repo_root):
        raise RuntimeError("Git 工作区不干净；正式训练请先提交，或仅调试时传 --allow-dirty")

    disk_report = _check_disk(output_root, config.runtime.minimum_free_disk_gb)
    hardware_report = cuda_preflight()
    manifest_report = verify_data_manifest(
        repo_root,
        manifest_path,
        config.data.expected_aggregate_sha256,
        train_path,
        eval_path,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = config.to_dict()
    resolved_config["cli_overrides"] = {
        "run_id": override.run_id,
        "output_root": str(override.output_root) if override.output_root else None,
        "max_train_samples": override.max_train_samples,
        "max_eval_samples": override.max_eval_samples,
        "max_steps": override.max_steps,
        "resume_from_checkpoint": override.resume_from_checkpoint,
        "allow_dirty": override.allow_dirty,
        "skip_generations": override.skip_generations,
    }
    _write_json(run_dir / "config_resolved.json", resolved_config)
    _write_json(run_dir / "hardware_preflight.json", hardware_report)
    _write_json(run_dir / "disk_preflight.json", disk_report)
    _write_json(run_dir / "data_manifest_verified.json", manifest_report)
    runtime_snapshot = write_runtime_snapshot(repo_root, run_dir / "environment.json")
    _write_json(run_dir / "git_state.json", runtime_snapshot["git"])
    _write_text(run_dir / "command.txt", shlex.join(sys.argv))
    shutil.copy2(manifest_path, run_dir / "source_data_manifest.json")

    max_train_samples = (
        override.max_train_samples
        if override.max_train_samples is not None
        else config.profile.max_train_samples
    )
    max_eval_samples = (
        override.max_eval_samples
        if override.max_eval_samples is not None
        else config.profile.max_eval_samples
    )
    train_records = load_sft_records(train_path, max_train_samples)
    eval_records = load_sft_records(eval_path, max_eval_samples)
    tokenizer = load_tokenizer(config)
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
    _write_json(
        run_dir / "token_audit.json",
        {"train": train_audit.to_dict(), "eval": eval_audit.to_dict()},
    )
    train_dataset = build_hf_dataset(train_records)
    eval_dataset = build_hf_dataset(eval_records)

    model, parameter_report = build_qlora_model(config)
    _write_json(run_dir / "trainable_parameters.json", parameter_report)
    generation_enabled = config.generation.enabled and not override.skip_generations
    samples_dir = run_dir / "samples"
    diagnostics: dict[str, Any] = {}
    pre_path = samples_dir / "pre_sft_generations.jsonl"
    if generation_enabled and not pre_path.exists():
        pre_samples, pre_metrics = generate_diagnostic_samples(
            model,
            tokenizer,
            eval_records,
            config.generation_sample_size,
            config.generation.max_new_tokens,
        )
        write_jsonl(pre_path, pre_samples)
        diagnostics["pre_sft"] = pre_metrics

    trl_config = build_trl_sft_config(
        config,
        run_dir,
        run_id,
        max_steps=override.max_steps,
    )
    trainer = SFTTrainer(
        model=model,
        args=trl_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[_metrics_callback(run_dir / "eval_history.jsonl")],
    )
    train_result = trainer.train(
        resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None
    )
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    _write_json(run_dir / "train_metrics.json", train_result.metrics)
    trainer.save_state()
    trainer.state.save_to_json(str(run_dir / "trainer_state.json"))
    final_eval_metrics = trainer.evaluate(metric_key_prefix="final_eval")
    trainer.log_metrics("final_eval", final_eval_metrics)
    trainer.save_metrics("final_eval", final_eval_metrics)
    _write_json(run_dir / "final_eval_metrics.json", final_eval_metrics)

    adapter_dir = run_dir / "best_adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    if generation_enabled:
        post_samples, post_metrics = generate_diagnostic_samples(
            trainer.model,
            tokenizer,
            eval_records,
            config.generation_sample_size,
            config.generation.max_new_tokens,
        )
        write_jsonl(samples_dir / "post_sft_generations.jsonl", post_samples)
        diagnostics["post_sft"] = post_metrics
    _write_json(run_dir / "generation_metrics.json", diagnostics)

    selected = {
        "run_id": run_id,
        "profile": config.profile_name,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "adapter_dir": str(adapter_dir),
        "train_rows": len(train_records),
        "eval_rows": len(eval_records),
        "global_step": trainer.state.global_step,
        "train_metrics": train_result.metrics,
        "final_eval_metrics": final_eval_metrics,
        "diagnostics": diagnostics,
    }
    _write_json(run_dir / "selected_checkpoint.json", selected)
    return selected

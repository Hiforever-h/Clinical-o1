"""Stage 3 SFT 配置读取、类型转换与门禁校验。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class ModelSpec:
    name_or_path: str
    revision: str
    trust_remote_code: bool = False
    attn_implementation: str = "sdpa"


@dataclass(frozen=True)
class DataSpec:
    train_file: str
    eval_file: str
    manifest_file: str
    expected_aggregate_sha256: str
    max_length: int = 2048
    packing: bool = False
    completion_only_loss: bool = True
    assistant_only_loss: bool = False
    audit_sample_size: int = 100


@dataclass(frozen=True)
class QuantizationSpec:
    load_in_4bit: bool = True
    quant_type: str = "nf4"
    use_double_quant: bool = True
    compute_dtype: str = "bfloat16"


@dataclass(frozen=True)
class LoraSpec:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str | list[str] = "all-linear"
    bias: str = "none"
    use_rslora: bool = False


@dataclass(frozen=True)
class TrainingSpec:
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    optim: str = "adamw_torch_fused"
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True
    gradient_checkpointing: bool = True
    use_reentrant: bool = False
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    eval_on_start: bool = True
    seed: int = 42
    data_seed: int = 42
    dataloader_num_workers: int = 4
    report_to: list[str] | None = None


@dataclass(frozen=True)
class GenerationSpec:
    enabled: bool = True
    sample_size: int = 64
    max_new_tokens: int = 1024


@dataclass(frozen=True)
class RuntimeSpec:
    output_root: str = "outputs/sft"
    minimum_free_disk_gb: int = 15
    require_clean_git: bool = True


@dataclass(frozen=True)
class ProfileSpec:
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    max_steps: int = -1
    gradient_accumulation_steps: int | None = None
    eval_steps: int | None = None
    save_steps: int | None = None
    generation_sample_size: int | None = None


@dataclass(frozen=True)
class SFTExperimentConfig:
    """经过校验的 SFT 实验配置及当前执行 profile。"""

    schema_version: int
    experiment_name: str
    model: ModelSpec
    data: DataSpec
    quantization: QuantizationSpec
    lora: LoraSpec
    training: TrainingSpec
    generation: GenerationSpec
    runtime: RuntimeSpec
    profiles: dict[str, ProfileSpec]
    profile_name: str
    source_path: Path

    @property
    def profile(self) -> ProfileSpec:
        return self.profiles[self.profile_name]

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.profile.gradient_accumulation_steps or self.training.gradient_accumulation_steps

    @property
    def eval_steps(self) -> int:
        return self.profile.eval_steps or self.training.eval_steps

    @property
    def save_steps(self) -> int:
        return self.profile.save_steps or self.training.save_steps

    @property
    def generation_sample_size(self) -> int:
        return self.profile.generation_sample_size or self.generation.sample_size

    @property
    def effective_global_batch_size(self) -> int:
        return self.training.per_device_train_batch_size * self.gradient_accumulation_steps

    def resolve_path(self, repo_root: Path, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (repo_root / path).resolve()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_path"] = str(self.source_path)
        payload["effective"] = {
            "profile": self.profile_name,
            "max_train_samples": self.profile.max_train_samples,
            "max_eval_samples": self.profile.max_eval_samples,
            "max_steps": self.profile.max_steps,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
            "generation_sample_size": self.generation_sample_size,
            "global_batch_size": self.effective_global_batch_size,
        }
        return payload


def _mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"配置段 {key!r} 必须是对象")
    return dict(value)


def _build_dataclass(cls: type[Any], payload: Mapping[str, Any], section: str) -> Any:
    try:
        return cls(**dict(payload))
    except TypeError as exc:
        raise ValueError(f"配置段 {section!r} 字段错误：{exc}") from exc


def _validate(config: SFTExperimentConfig) -> None:
    if config.schema_version != 1:
        raise ValueError(f"不支持的 schema_version={config.schema_version}")
    if not config.experiment_name.strip():
        raise ValueError("experiment_name 不能为空")
    if len(config.model.revision) != 40:
        raise ValueError("model.revision 必须是 40 位 commit SHA")
    if len(config.data.expected_aggregate_sha256) != 64:
        raise ValueError("data.expected_aggregate_sha256 必须是 64 位 SHA256")
    if config.data.max_length <= 0:
        raise ValueError("data.max_length 必须大于 0")
    if config.data.packing:
        raise ValueError("第一版 SFT 明确禁止 packing")
    if not config.data.completion_only_loss or config.data.assistant_only_loss:
        raise ValueError("必须使用 completion_only_loss=true 且 assistant_only_loss=false")
    if not config.quantization.load_in_4bit or config.quantization.quant_type != "nf4":
        raise ValueError("RTX 4090 基线必须使用 4-bit NF4")
    if config.quantization.compute_dtype != "bfloat16":
        raise ValueError("RTX 4090 基线 compute_dtype 必须是 bfloat16")
    if config.lora.rank <= 0 or config.lora.alpha <= 0:
        raise ValueError("LoRA rank/alpha 必须大于 0")
    if not 0.0 <= config.lora.dropout < 1.0:
        raise ValueError("LoRA dropout 必须位于 [0, 1)")
    if config.training.bf16 == config.training.fp16:
        raise ValueError("bf16 和 fp16 必须且只能启用一个")
    if config.training.per_device_train_batch_size != 1:
        raise ValueError("24GB RTX 4090 基线的 micro batch 必须为 1")
    if config.eval_steps != config.save_steps:
        raise ValueError("load_best_model_at_end 要求 eval_steps 与 save_steps 一致")
    if config.effective_global_batch_size <= 0:
        raise ValueError("有效全局 batch 必须大于 0")
    if config.generation.max_new_tokens <= 0:
        raise ValueError("generation.max_new_tokens 必须大于 0")
    if config.runtime.minimum_free_disk_gb < 10:
        raise ValueError("minimum_free_disk_gb 不得低于 10GB")
    for name, profile in config.profiles.items():
        if profile.max_steps != -1 and profile.max_steps <= 0:
            raise ValueError(f"profiles.{name}.max_steps 必须为 -1 或正整数")
        for field_name, value in (
            ("max_train_samples", profile.max_train_samples),
            ("max_eval_samples", profile.max_eval_samples),
            ("gradient_accumulation_steps", profile.gradient_accumulation_steps),
            ("eval_steps", profile.eval_steps),
            ("save_steps", profile.save_steps),
            ("generation_sample_size", profile.generation_sample_size),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"profiles.{name}.{field_name} 必须为正整数或 null")


def load_sft_config(path: Path, profile: str = "full") -> SFTExperimentConfig:
    """读取 YAML，并把 profile 覆盖解析成类型安全配置。"""

    source_path = path.resolve()
    payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"配置顶层必须是对象：{source_path}")
    raw_profiles = _mapping(payload, "profiles")
    profiles = {
        name: _build_dataclass(ProfileSpec, value, f"profiles.{name}")
        for name, value in raw_profiles.items()
        if isinstance(value, Mapping)
    }
    if profile not in profiles:
        raise ValueError(f"未知 profile={profile!r}，可选值：{sorted(profiles)}")
    config = SFTExperimentConfig(
        schema_version=int(payload.get("schema_version", 0)),
        experiment_name=str(payload.get("experiment_name", "")),
        model=_build_dataclass(ModelSpec, _mapping(payload, "model"), "model"),
        data=_build_dataclass(DataSpec, _mapping(payload, "data"), "data"),
        quantization=_build_dataclass(
            QuantizationSpec, _mapping(payload, "quantization"), "quantization"
        ),
        lora=_build_dataclass(LoraSpec, _mapping(payload, "lora"), "lora"),
        training=_build_dataclass(TrainingSpec, _mapping(payload, "training"), "training"),
        generation=_build_dataclass(
            GenerationSpec, _mapping(payload, "generation"), "generation"
        ),
        runtime=_build_dataclass(RuntimeSpec, _mapping(payload, "runtime"), "runtime"),
        profiles=profiles,
        profile_name=profile,
        source_path=source_path,
    )
    _validate(config)
    return config


def make_run_id(config: SFTExperimentConfig, now: datetime | None = None) -> str:
    """生成不会复用历史目录的 UTC run ID。"""

    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{config.experiment_name}_{config.profile_name}_seed{config.training.seed}_{timestamp}"

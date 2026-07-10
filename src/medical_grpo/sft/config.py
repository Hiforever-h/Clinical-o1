"""Stage 3 SFT 配置读取、类型转换与门禁校验。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class ModelSpec:
    """基础模型位置、不可变 revision 与注意力实现。"""

    # 训练时必须同时给出仓库名和 commit SHA，不能依赖会漂移的 main 分支。
    name_or_path: str
    revision: str
    trust_remote_code: bool = False
    attn_implementation: str = "sdpa"


@dataclass(frozen=True)
class DataSpec:
    """SFT 数据文件、数据指纹和序列处理规则。"""

    # manifest 和 aggregate SHA 把本次训练绑定到 M1 已验收的数据版本。
    train_file: str
    eval_file: str
    manifest_file: str
    expected_aggregate_sha256: str
    # 当前数据最大 1451 tokens，2048 足够容纳全部样本并节省显存。
    max_length: int = 2048
    packing: bool = False
    # prompt-completion 数据只监督 assistant completion，user prompt 不参与 loss。
    completion_only_loss: bool = True
    assistant_only_loss: bool = False
    audit_sample_size: int = 100


@dataclass(frozen=True)
class QuantizationSpec:
    """bitsandbytes 4-bit QLoRA 量化参数。"""

    load_in_4bit: bool = True
    quant_type: str = "nf4"
    use_double_quant: bool = True
    # 4090 原生支持 BF16，用它作为量化矩阵运算的计算精度。
    compute_dtype: str = "bfloat16"


@dataclass(frozen=True)
class LoraSpec:
    """PEFT LoRA adapter 的容量和目标层配置。"""

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    # all-linear 覆盖 attention 与 MLP 线性层，符合 QLoRA 推荐做法。
    target_modules: str | list[str] = "all-linear"
    bias: str = "none"
    use_rslora: bool = False


@dataclass(frozen=True)
class TrainingSpec:
    """与 profile 无关的训练、精度、日志和 DataLoader 参数。"""

    # 24GB 显存下 micro batch 固定为 1，有效 batch 由梯度累积放大。
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
    # BF16 与 FP16 互斥；TF32 只加速支持的 CUDA 矩阵乘法。
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True
    # 非重入 checkpointing 是当前 PyTorch/Transformers 更稳妥的组合。
    gradient_checkpointing: bool = True
    use_reentrant: bool = False
    # eval/save 步长必须一致，才能安全加载 dev loss 最优 checkpoint。
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    eval_on_start: bool = True
    seed: int = 42
    data_seed: int = 42
    dataloader_num_workers: int = 4
    # 正式环境使用 TensorBoard，同时训练器另存 JSONL 标量日志。
    report_to: list[str] | None = None


@dataclass(frozen=True)
class GenerationSpec:
    """训练前后固定 dev 样本生成诊断参数。"""

    enabled: bool = True
    sample_size: int = 64
    max_new_tokens: int = 1024


@dataclass(frozen=True)
class RuntimeSpec:
    """输出目录、磁盘余量和代码状态门禁。"""

    output_root: str = "outputs/sft"
    minimum_free_disk_gb: int = 15
    require_clean_git: bool = True


@dataclass(frozen=True)
class ProfileSpec:
    """smoke、pilot、full 对数据规模和日志频率的局部覆盖。"""

    # None 表示使用全部数据，-1 表示由 epoch 数推导总训练步数。
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
        """返回 CLI 选中的 profile 配置。"""

        return self.profiles[self.profile_name]

    @property
    def gradient_accumulation_steps(self) -> int:
        """优先使用 profile 覆盖，否则使用正式训练默认值。"""

        return self.profile.gradient_accumulation_steps or self.training.gradient_accumulation_steps

    @property
    def eval_steps(self) -> int:
        """返回当前 profile 生效的评估间隔。"""

        return self.profile.eval_steps or self.training.eval_steps

    @property
    def save_steps(self) -> int:
        """返回当前 profile 生效的 checkpoint 保存间隔。"""

        return self.profile.save_steps or self.training.save_steps

    @property
    def generation_sample_size(self) -> int:
        """返回当前 profile 用于前后生成诊断的样本数。"""

        return self.profile.generation_sample_size or self.generation.sample_size

    @property
    def effective_global_batch_size(self) -> int:
        """单卡有效 batch 等于 micro batch 乘梯度累积步数。"""

        return self.training.per_device_train_batch_size * self.gradient_accumulation_steps

    def resolve_path(self, repo_root: Path, value: str) -> Path:
        """把 YAML 相对路径稳定解释为相对于仓库根目录。"""

        path = Path(value)
        return path if path.is_absolute() else (repo_root / path).resolve()

    def to_dict(self) -> dict[str, Any]:
        """序列化原始配置，并附加 profile 解析后的最终生效值。"""

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
    """读取必需的 YAML 对象段，避免把列表或标量误当配置。"""

    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"配置段 {key!r} 必须是对象")
    return dict(value)


def _build_dataclass(cls: type[Any], payload: Mapping[str, Any], section: str) -> Any:
    """将单个 YAML 配置段转换为 dataclass，并改善字段错误信息。"""

    try:
        return cls(**dict(payload))
    except TypeError as exc:
        raise ValueError(f"配置段 {section!r} 字段错误：{exc}") from exc


def _validate(config: SFTExperimentConfig) -> None:
    """执行 4090 基线的强约束校验，拒绝静默改变实验定义。"""

    # 先检查版本、模型和数据指纹，保证实验输入可复现。
    if config.schema_version != 1:
        raise ValueError(f"不支持的 schema_version={config.schema_version}")
    if not config.experiment_name.strip():
        raise ValueError("experiment_name 不能为空")
    if len(config.model.revision) != 40:
        raise ValueError("model.revision 必须是 40 位 commit SHA")
    if len(config.data.expected_aggregate_sha256) != 64:
        raise ValueError("data.expected_aggregate_sha256 必须是 64 位 SHA256")

    # 数据策略固定为不 packing、只监督 completion；违反时直接拒绝运行。
    if config.data.max_length <= 0:
        raise ValueError("data.max_length 必须大于 0")
    if config.data.packing:
        raise ValueError("第一版 SFT 明确禁止 packing")
    if not config.data.completion_only_loss or config.data.assistant_only_loss:
        raise ValueError("必须使用 completion_only_loss=true 且 assistant_only_loss=false")

    # 量化和 LoRA 参数决定显存占用与可训练参数范围，不能由 profile 改写。
    if not config.quantization.load_in_4bit or config.quantization.quant_type != "nf4":
        raise ValueError("RTX 4090 基线必须使用 4-bit NF4")
    if config.quantization.compute_dtype != "bfloat16":
        raise ValueError("RTX 4090 基线 compute_dtype 必须是 bfloat16")
    if config.lora.rank <= 0 or config.lora.alpha <= 0:
        raise ValueError("LoRA rank/alpha 必须大于 0")
    if not 0.0 <= config.lora.dropout < 1.0:
        raise ValueError("LoRA dropout 必须位于 [0, 1)")

    # 精度、batch 和 eval/save 对齐是单卡稳定训练及最优模型恢复的前提。
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

    # 逐个 profile 检查覆盖值，避免 0 步、0 样本等看似成功的空训练。
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
    # profile 独立解析，便于 smoke/pilot 只覆盖必要字段而复用同一主配置。
    raw_profiles = _mapping(payload, "profiles")
    profiles = {
        name: _build_dataclass(ProfileSpec, value, f"profiles.{name}")
        for name, value in raw_profiles.items()
        if isinstance(value, Mapping)
    }
    if profile not in profiles:
        raise ValueError(f"未知 profile={profile!r}，可选值：{sorted(profiles)}")
    # 每个配置段都通过 dataclass 构造，从而拒绝拼写错误或未知字段。
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

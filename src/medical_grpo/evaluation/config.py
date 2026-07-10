"""Stage 2 评测 YAML 的类型转换、数据集选择和强约束校验。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


SUPPORTED_DATASETS = ("medqa", "medmcqa", "pubmedqa")
SUPPORTED_PROTOCOLS = ("direct", "cot")


@dataclass(frozen=True)
class EvalModelSpec:
    """BF16 基础模型位置、revision 与注意力实现。"""

    name_or_path: str
    revision: str
    trust_remote_code: bool = False
    attn_implementation: str = "sdpa"
    dtype: str = "bfloat16"


@dataclass(frozen=True)
class EvalDatasetSpec:
    """一个 benchmark 在 manifest 和本地 processed 目录中的定位。"""

    manifest_key: str
    path: str


@dataclass(frozen=True)
class EvalDataSpec:
    """冻结数据清单、aggregate SHA 与三个 benchmark 配置。"""

    manifest_file: str
    expected_aggregate_sha256: str
    datasets: dict[str, EvalDatasetSpec]


@dataclass(frozen=True)
class InferenceSpec:
    """序列长度、原子分片和 GPU 预检参数。"""

    max_input_length: int = 2048
    shard_size: int = 64
    seed: int = 42
    minimum_vram_gb: float = 23.0


@dataclass(frozen=True)
class ProtocolSpec:
    """单个生成协议的最大输出长度和固定 batch size。"""

    max_new_tokens: int
    batch_size: int


@dataclass(frozen=True)
class EvalProfileSpec:
    """smoke/full 对每个数据集最大题数的覆盖。"""

    # None 表示评测该数据集的全部记录。
    max_samples_per_dataset: int | None = None


@dataclass(frozen=True)
class EvalRuntimeSpec:
    """评测输出位置、磁盘余量和 clean Git 门禁。"""

    output_root: str = "outputs/evaluation"
    minimum_free_disk_gb: int = 5
    require_clean_git: bool = True


@dataclass(frozen=True)
class EvaluationConfig:
    """经过校验的评测配置及当前 profile。"""

    schema_version: int
    experiment_name: str
    model: EvalModelSpec
    data: EvalDataSpec
    inference: InferenceSpec
    protocols: dict[str, ProtocolSpec]
    profiles: dict[str, EvalProfileSpec]
    runtime: EvalRuntimeSpec
    profile_name: str
    source_path: Path

    @property
    def profile(self) -> EvalProfileSpec:
        """返回 CLI 当前选择的 smoke 或 full profile。"""

        return self.profiles[self.profile_name]

    def resolve_path(self, repo_root: Path, value: str) -> Path:
        """将 YAML 相对路径统一解释为相对于仓库根目录。"""

        path = Path(value)
        return path.resolve() if path.is_absolute() else (repo_root / path).resolve()

    def to_dict(self) -> dict[str, Any]:
        """转换为可写入 run 目录的完整配置字典。"""

        payload = asdict(self)
        payload["source_path"] = str(self.source_path)
        payload["profile_name"] = self.profile_name
        return payload


def _mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    """读取必需的 YAML 对象段，并拒绝列表或标量。"""

    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"配置段 {key!r} 必须是对象")
    return dict(value)


def _construct(cls: type[Any], payload: Mapping[str, Any], section: str) -> Any:
    """构造 dataclass，并把未知字段错误关联到具体配置段。"""

    try:
        return cls(**dict(payload))
    except TypeError as exc:
        raise ValueError(f"配置段 {section!r} 字段错误：{exc}") from exc


def normalize_dataset_selection(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """规范化 CLI 数据集选择，支持 all、空格列表和逗号列表。"""

    expanded: list[str] = []
    for value in values:
        expanded.extend(part.strip().lower() for part in value.split(",") if part.strip())
    if not expanded or "all" in expanded:
        if len(expanded) > 1:
            raise ValueError("datasets 使用 all 时不能再混入其他数据集")
        return SUPPORTED_DATASETS
    unknown = sorted(set(expanded) - set(SUPPORTED_DATASETS))
    if unknown:
        raise ValueError(f"未知评测数据集：{unknown}；可选值为 {SUPPORTED_DATASETS}")
    # 按固定全局顺序去重，避免 CLI 输入顺序改变 contract 和输出顺序。
    return tuple(name for name in SUPPORTED_DATASETS if name in set(expanded))


def normalize_protocol_selection(value: str) -> tuple[str, ...]:
    """把 direct/cot/both 转换为固定顺序的协议元组。"""

    normalized = value.strip().lower()
    if normalized == "both":
        return SUPPORTED_PROTOCOLS
    if normalized not in SUPPORTED_PROTOCOLS:
        raise ValueError(f"未知 protocol={value!r}")
    return (normalized,)


def resolve_max_samples(config: EvaluationConfig, cli_value: int | None) -> int | None:
    """CLI 前 N 题限制优先于 profile；限制按每个数据集分别应用。"""

    value = cli_value if cli_value is not None else config.profile.max_samples_per_dataset
    if value is not None and value <= 0:
        raise ValueError("max-samples 必须为正整数")
    return value


def _validate(config: EvaluationConfig) -> None:
    """冻结公平评测所需的模型、数据、精度和生成协议。"""

    if config.schema_version != 1:
        raise ValueError(f"不支持 schema_version={config.schema_version}")
    if not config.experiment_name.strip():
        raise ValueError("experiment_name 不能为空")
    if len(config.model.revision) != 40:
        raise ValueError("model.revision 必须是 40 位 commit SHA")
    if config.model.dtype != "bfloat16":
        raise ValueError("公平评测基线固定使用 bfloat16，不允许 4-bit 或 fp16")
    if len(config.data.expected_aggregate_sha256) != 64:
        raise ValueError("expected_aggregate_sha256 必须是 64 位 SHA256")
    if tuple(config.data.datasets) != SUPPORTED_DATASETS:
        raise ValueError(f"data.datasets 必须按顺序包含 {SUPPORTED_DATASETS}")
    if tuple(config.protocols) != SUPPORTED_PROTOCOLS:
        raise ValueError(f"protocols 必须按顺序包含 {SUPPORTED_PROTOCOLS}")
    if config.inference.max_input_length <= 0 or config.inference.shard_size <= 0:
        raise ValueError("max_input_length 和 shard_size 必须为正整数")
    for name, protocol in config.protocols.items():
        if protocol.max_new_tokens <= 0 or protocol.batch_size <= 0:
            raise ValueError(f"protocols.{name} 的 max_new_tokens/batch_size 必须为正整数")
    for name, profile in config.profiles.items():
        if profile.max_samples_per_dataset is not None and profile.max_samples_per_dataset <= 0:
            raise ValueError(f"profiles.{name}.max_samples_per_dataset 必须为正整数或 null")


def load_evaluation_config(path: Path, profile: str = "full") -> EvaluationConfig:
    """读取 YAML，解析嵌套数据集/协议并应用 profile。"""

    source_path = path.resolve()
    payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"配置顶层必须是对象：{source_path}")

    raw_data = _mapping(payload, "data")
    raw_datasets = _mapping(raw_data, "datasets")
    datasets = {
        name: _construct(EvalDatasetSpec, value, f"data.datasets.{name}")
        for name, value in raw_datasets.items()
        if isinstance(value, Mapping)
    }
    data = EvalDataSpec(
        manifest_file=str(raw_data.get("manifest_file", "")),
        expected_aggregate_sha256=str(raw_data.get("expected_aggregate_sha256", "")),
        datasets=datasets,
    )

    raw_protocols = _mapping(payload, "protocols")
    protocols = {
        name: _construct(ProtocolSpec, value, f"protocols.{name}")
        for name, value in raw_protocols.items()
        if isinstance(value, Mapping)
    }
    raw_profiles = _mapping(payload, "profiles")
    profiles = {
        name: _construct(EvalProfileSpec, value, f"profiles.{name}")
        for name, value in raw_profiles.items()
        if isinstance(value, Mapping)
    }
    if profile not in profiles:
        raise ValueError(f"未知 profile={profile!r}，可选值：{sorted(profiles)}")

    config = EvaluationConfig(
        schema_version=int(payload.get("schema_version", 0)),
        experiment_name=str(payload.get("experiment_name", "")),
        model=_construct(EvalModelSpec, _mapping(payload, "model"), "model"),
        data=data,
        inference=_construct(InferenceSpec, _mapping(payload, "inference"), "inference"),
        protocols=protocols,
        profiles=profiles,
        runtime=_construct(EvalRuntimeSpec, _mapping(payload, "runtime"), "runtime"),
        profile_name=profile,
        source_path=source_path,
    )
    _validate(config)
    return config

"""SFT 数据加载、prompt-completion 转换和 label 边界审计。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from datasets import Dataset

from medical_grpo.data.schema import to_sft_sample
from medical_grpo.tracking.artifacts import sha256_file


@dataclass(frozen=True)
class TokenAuditSummary:
    rows: int
    min_input_tokens: int
    mean_input_tokens: float
    p95_input_tokens: int
    max_input_tokens: int
    min_supervised_tokens: int
    mean_supervised_tokens: float
    max_supervised_tokens: int
    truncated_rows: int
    empty_completion_rows: int
    prefix_mismatch_rows: int
    max_length: int
    inspected_samples: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL，并在错误信息中保留文件名和行号。"""

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: JSON 解析失败：{exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: 每行必须是 JSON 对象")
            records.append(payload)
    return records


def load_sft_records(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    """逐条通过 canonical schema，并可为 smoke/pilot 截取固定前 N 条。"""

    raw_records = read_jsonl(path)
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples 必须大于 0")
        raw_records = raw_records[:max_samples]
    records: list[dict[str, Any]] = []
    for index, record in enumerate(raw_records, start=1):
        try:
            records.append(to_sft_sample(record).to_dict())
        except ValueError as exc:
            raise ValueError(f"{path}:{index}: {exc}") from exc
    if not records:
        raise ValueError(f"SFT 文件为空：{path}")
    return records


def to_prompt_completion(record: Mapping[str, Any]) -> dict[str, Any]:
    """把 canonical 单轮对话拆成 TRL conversational prompt-completion。"""

    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError(f"{record.get('id')}: 当前 SFT 只支持单轮 user/assistant")
    user, assistant = messages
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        raise ValueError(f"{record.get('id')}: messages 角色顺序必须是 user → assistant")
    assistant_content = str(assistant.get("content", ""))
    if "## Thinking" not in assistant_content or "## Final Response" not in assistant_content:
        raise ValueError(f"{record.get('id')}: assistant 缺少 Huatuo 输出标题")
    return {
        "id": str(record["id"]),
        "source": str(record["source"]),
        "prompt": [{"role": "user", "content": str(user.get("content", ""))}],
        "completion": [{"role": "assistant", "content": assistant_content}],
    }


def build_hf_dataset(records: Iterable[Mapping[str, Any]]) -> Dataset:
    """构造 SFTTrainer 原生支持的 prompt-completion Dataset。"""

    return Dataset.from_list([to_prompt_completion(record) for record in records])


def verify_data_manifest(
    repo_root: Path,
    manifest_path: Path,
    expected_aggregate_sha256: str,
    train_path: Path,
    eval_path: Path,
) -> dict[str, Any]:
    """在训练前重算 train/dev 哈希，禁止使用被替换过的数据。"""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_aggregate = str(manifest.get("aggregate_sha256", ""))
    if actual_aggregate != expected_aggregate_sha256:
        raise ValueError(
            "数据 aggregate SHA256 不匹配："
            f"{actual_aggregate} != {expected_aggregate_sha256}"
        )
    checks: dict[str, Any] = {}
    for key, path in (("sft_train", train_path), ("sft_dev", eval_path)):
        metadata = manifest.get("files", {}).get(key)
        if not isinstance(metadata, Mapping):
            raise ValueError(f"manifest 缺少 files.{key}")
        expected_path = (repo_root / str(metadata["path"])).resolve()
        if expected_path != path.resolve():
            raise ValueError(f"{key} 路径与 manifest 不一致：{path} != {expected_path}")
        digest = sha256_file(path)
        if digest != metadata["sha256"]:
            raise ValueError(f"{key} SHA256 不一致：{digest} != {metadata['sha256']}")
        checks[key] = {
            "path": str(path),
            "rows": int(metadata["rows"]),
            "sha256": digest,
        }
    return {"aggregate_sha256": actual_aggregate, "files": checks}


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        length += 1
    return length


def audit_token_boundaries(
    records: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    inspect_samples: int = 100,
    batch_size: int = 128,
) -> TokenAuditSummary:
    """模拟 TRL 的 prompt/completion 边界，证明 prompt 不参与 loss 且没有截断。"""

    prepared = [to_prompt_completion(record) for record in records]
    input_lengths: list[int] = []
    supervised_lengths: list[int] = []
    truncated_rows = 0
    empty_rows = 0
    mismatch_rows = 0
    inspected: list[dict[str, Any]] = []

    for start in range(0, len(prepared), batch_size):
        batch = prepared[start : start + batch_size]
        prompt_texts = [
            tokenizer.apply_chat_template(
                item["prompt"], tokenize=False, add_generation_prompt=True
            )
            for item in batch
        ]
        full_texts = [
            tokenizer.apply_chat_template(
                item["prompt"] + item["completion"],
                tokenize=False,
                add_generation_prompt=False,
            )
            for item in batch
        ]
        prompt_ids = tokenizer(
            prompt_texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
        full_ids = tokenizer(
            full_texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]

        for item, one_prompt_ids, one_full_ids in zip(
            batch, prompt_ids, full_ids, strict=True
        ):
            prefix_length = _common_prefix_length(one_prompt_ids, one_full_ids)
            supervised_length = len(one_full_ids) - prefix_length
            input_lengths.append(len(one_full_ids))
            supervised_lengths.append(supervised_length)
            truncated_rows += int(len(one_full_ids) > max_length)
            empty_rows += int(supervised_length <= 0)
            mismatch_rows += int(prefix_length != len(one_prompt_ids))
            if len(inspected) < inspect_samples:
                inspected.append(
                    {
                        "id": item["id"],
                        "input_tokens": len(one_full_ids),
                        "prompt_tokens": prefix_length,
                        "supervised_tokens": supervised_length,
                        "supervised_preview": tokenizer.decode(
                            one_full_ids[prefix_length : prefix_length + 96],
                            skip_special_tokens=False,
                        ),
                    }
                )

    summary = TokenAuditSummary(
        rows=len(records),
        min_input_tokens=min(input_lengths),
        mean_input_tokens=round(sum(input_lengths) / len(input_lengths), 2),
        p95_input_tokens=_percentile(input_lengths, 0.95),
        max_input_tokens=max(input_lengths),
        min_supervised_tokens=min(supervised_lengths),
        mean_supervised_tokens=round(sum(supervised_lengths) / len(supervised_lengths), 2),
        max_supervised_tokens=max(supervised_lengths),
        truncated_rows=truncated_rows,
        empty_completion_rows=empty_rows,
        prefix_mismatch_rows=mismatch_rows,
        max_length=max_length,
        inspected_samples=inspected,
    )
    if summary.truncated_rows:
        raise ValueError(f"发现 {summary.truncated_rows} 条记录超过 max_length={max_length}")
    if summary.empty_completion_rows:
        raise ValueError(f"发现 {summary.empty_completion_rows} 条 completion 没有监督 token")
    if summary.prefix_mismatch_rows:
        raise ValueError(f"发现 {summary.prefix_mismatch_rows} 条 chat template 前缀不一致")
    return summary

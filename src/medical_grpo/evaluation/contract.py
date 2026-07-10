"""公平评测 contract 构造与稳定 SHA256 计算。"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any

from medical_grpo.evaluation.config import EvaluationConfig
from medical_grpo.evaluation.parsing import PARSER_VERSION
from medical_grpo.evaluation.prompts import prompt_contract


def build_evaluation_contract(
    config: EvaluationConfig,
    selected_datasets: tuple[str, ...],
    selected_protocols: tuple[str, ...],
    max_samples_per_dataset: int | None,
    data_report: dict[str, Any],
) -> dict[str, Any]:
    """构造不含模型权重身份、但覆盖全部公平比较条件的合同。"""

    # 合同只保留与结果相关的稳定字段，不写机器相关的绝对路径。
    portable_datasets = {
        name: {
            key: value
            for key, value in report.items()
            if key != "path"
        }
        for name, report in data_report["datasets"].items()
    }
    contract = {
        "schema_version": 1,
        "base_model": config.model.name_or_path,
        "base_revision": config.model.revision,
        "tokenizer_revision": config.model.revision,
        "dtype": config.model.dtype,
        "attn_implementation": config.model.attn_implementation,
        "selected_datasets": list(selected_datasets),
        "selected_protocols": list(selected_protocols),
        "max_samples_per_dataset": max_samples_per_dataset,
        "max_input_length": config.inference.max_input_length,
        "shard_size": config.inference.shard_size,
        "seed": config.inference.seed,
        "protocols": {
            name: {
                "max_new_tokens": config.protocols[name].max_new_tokens,
                "batch_size": config.protocols[name].batch_size,
                "do_sample": False,
                "num_beams": 1,
            }
            for name in selected_protocols
        },
        "prompt": prompt_contract(),
        "parser_version": PARSER_VERSION,
        "data": {
            "aggregate_sha256": data_report["aggregate_sha256"],
            "datasets": portable_datasets,
        },
    }
    encoded = json.dumps(contract, ensure_ascii=False, sort_keys=True).encode("utf-8")
    contract["evaluation_contract_sha256"] = sha256(encoded).hexdigest()
    return contract

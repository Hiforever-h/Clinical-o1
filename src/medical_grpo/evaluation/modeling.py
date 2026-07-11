"""RTX 4090 BF16 Base/LoRA adapter 加载与确定性批量生成。"""

from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping

from medical_grpo.evaluation.config import EvaluationConfig, ProtocolSpec
from medical_grpo.evaluation.metrics import repetition_ratio
from medical_grpo.evaluation.parsing import parse_mcq_answer
from medical_grpo.evaluation.prompts import build_messages
from medical_grpo.tracking.artifacts import sha256_file


def cuda_eval_preflight(config: EvaluationConfig) -> dict[str, Any]:
    """拒绝在错误 PyTorch/CUDA、低显存或不支持 BF16 的设备上评测。"""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("evaluate 需要 CUDA GPU；本地只能运行 eval-dry-run")
    if not torch.__version__.startswith("2.8.0"):
        raise RuntimeError(f"正式评测要求 torch 2.8.0，当前为 {torch.__version__}")
    if torch.version.cuda != "12.8":
        raise RuntimeError(f"正式评测要求 PyTorch CUDA runtime 12.8，当前为 {torch.version.cuda}")
    device_index = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(device_index)
    properties = torch.cuda.get_device_properties(device_index)
    total_vram_gb = properties.total_memory / 1024**3
    if total_vram_gb < config.inference.minimum_vram_gb:
        raise RuntimeError(
            f"GPU 显存不足：{total_vram_gb:.2f}GB < {config.inference.minimum_vram_gb:.2f}GB"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 GPU/CUDA/PyTorch 组合不支持 BF16")
    return {
        "device_index": device_index,
        "device_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_vram_gb": round(total_vram_gb, 2),
        "bf16_supported": True,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }


def load_eval_tokenizer(config: EvaluationConfig) -> Any:
    """从固定 Base revision 加载 tokenizer，并配置生成所需的左 padding。"""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name_or_path,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("tokenizer 同时缺少 pad_token 和 eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    # decoder-only 模型做批量 generation 必须左 padding，保证新增 token 对齐。
    tokenizer.padding_side = "left"
    return tokenizer


def _adapter_tree_sha256(adapter_path: Path) -> str:
    """对 adapter 关键文件按相对路径和内容构造稳定目录哈希。"""

    hasher = sha256()
    candidates = sorted(
        path
        for path in adapter_path.rglob("*")
        if path.is_file() and path.suffix in {".json", ".safetensors", ".model"}
    )
    if not candidates:
        raise FileNotFoundError(f"adapter 目录没有可识别的权重或配置：{adapter_path}")
    for path in candidates:
        relative = path.relative_to(adapter_path).as_posix()
        hasher.update(relative.encode("utf-8"))
        hasher.update(sha256_file(path).encode("ascii"))
    return hasher.hexdigest()


def load_evaluation_model(
    config: EvaluationConfig,
    model_type: str,
    adapter_path: Path | None,
) -> tuple[Any, Any, dict[str, Any]]:
    """加载 BF16 Base，必要时挂载只读 PEFT adapter，并返回模型身份。"""

    import torch
    from transformers import AutoModelForCausalLM, set_seed

    if model_type not in {"base", "adapter"}:
        raise ValueError("model-type 必须是 base 或 adapter")
    if model_type == "adapter" and adapter_path is None:
        raise ValueError("model-type=adapter 时必须提供 --adapter-path")
    if model_type == "base" and adapter_path is not None:
        raise ValueError("model-type=base 时不能提供 --adapter-path")

    # 即使 greedy 不采样也固定全部随机源，避免未来生成参数扩展后结果漂移。
    set_seed(config.inference.seed)
    device_index = int(os.environ.get("LOCAL_RANK", "0"))
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
        dtype=torch.bfloat16,
        device_map={"": device_index},
        attn_implementation=config.model.attn_implementation,
        low_cpu_mem_usage=True,
    )
    identity: dict[str, Any] = {
        "model_type": model_type,
        "base_model": config.model.name_or_path,
        "base_revision": config.model.revision,
        "dtype": config.model.dtype,
    }
    if model_type == "adapter":
        from peft import PeftModel

        resolved_adapter = adapter_path.resolve()  # type: ignore[union-attr]
        if not resolved_adapter.is_dir():
            raise FileNotFoundError(f"adapter 目录不存在：{resolved_adapter}")
        model = PeftModel.from_pretrained(model, str(resolved_adapter), is_trainable=False)
        identity.update(
            {
                "adapter_path": str(resolved_adapter),
                "adapter_tree_sha256": _adapter_tree_sha256(resolved_adapter),
            }
        )
    model.config.use_cache = True
    model.eval()
    tokenizer = load_eval_tokenizer(config)
    return model, tokenizer, identity


def _trim_generated_ids(ids: list[int], eos_token_id: int | None, pad_token_id: int | None) -> list[int]:
    """在首个 EOS 处截断，并移除 batch generation 补齐的尾部 pad。"""

    if eos_token_id is not None and eos_token_id in ids:
        ids = ids[: ids.index(eos_token_id) + 1]
    while ids and pad_token_id is not None and ids[-1] == pad_token_id:
        ids.pop()
    return ids


def generate_batch_predictions(
    model: Any,
    tokenizer: Any,
    records: list[Mapping[str, Any]],
    dataset_name: str,
    protocol_name: str,
    protocol: ProtocolSpec,
    max_input_length: int,
    contract_sha256: str,
    batch_id_start: int,
    progress_callback: Callable[[int], None] | None = None,
) -> list[dict[str, Any]]:
    """按固定 batch size 做 greedy generation，并生成完整逐题审计记录。

    progress_callback 由运行编排层传入，每完成一个 batch 后按实际题数更新进度条。
    这里不直接依赖 tqdm，便于在测试或其他调用场景中替换进度展示方式。
    """

    import torch

    predictions: list[dict[str, Any]] = []
    device = next(model.parameters()).device
    for batch_start in range(0, len(records), protocol.batch_size):
        batch_records = records[batch_start : batch_start + protocol.batch_size]
        messages = [build_messages(record, protocol_name) for record in batch_records]
        prompt_texts = [
            tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=True)
            for item in messages
        ]
        encoded = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        )
        prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        too_long = [
            str(record["id"])
            for record, length in zip(batch_records, prompt_lengths, strict=True)
            if int(length) > max_input_length
        ]
        if too_long:
            raise ValueError(f"发现输入超过 max_input_length={max_input_length}：{too_long[:10]}")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        input_width = int(encoded["input_ids"].shape[1])

        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=protocol.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(
                f"{dataset_name}/{protocol_name} batch_size={protocol.batch_size} 发生 OOM；"
                "请先修改并重新冻结配置，不要在正式 run 中自动改变 batch"
            ) from exc
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        batch_id = batch_id_start + batch_start // protocol.batch_size

        for row, (record, prompt_text, prompt_length) in enumerate(
            zip(batch_records, prompt_texts, prompt_lengths, strict=True)
        ):
            completion_ids = _trim_generated_ids(
                generated[row, input_width:].tolist(),
                tokenizer.eos_token_id,
                tokenizer.pad_token_id,
            )
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            parsed = parse_mcq_answer(completion, record["options"], protocol_name)
            predictions.append(
                {
                    "id": str(record["id"]),
                    "dataset": dataset_name,
                    "protocol": protocol_name,
                    "source": str(record["source"]),
                    "question": str(record["question"]),
                    "options": dict(record["options"]),
                    "gold_answer": str(record["answer"]),
                    "prompt": prompt_text,
                    "prompt_sha256": sha256(prompt_text.encode("utf-8")).hexdigest(),
                    "raw_completion": completion,
                    **parsed.to_dict(),
                    "correct": parsed.parsed_answer == str(record["answer"]),
                    "prompt_tokens": int(prompt_length),
                    "completion_tokens": len(completion_ids),
                    "hit_max_new_tokens": len(completion_ids) >= protocol.max_new_tokens,
                    "repetition_4gram_ratio": round(repetition_ratio(completion), 6),
                    "batch_id": batch_id,
                    "batch_size": len(batch_records),
                    "batch_generation_seconds": round(elapsed, 6),
                    "estimated_sample_seconds": round(elapsed / len(batch_records), 6),
                    "evaluation_contract_sha256": contract_sha256,
                    "meta": dict(record.get("meta", {})),
                }
            )
        # 必须在当前 batch 的所有预测均完成解析和记录构造后再推进，避免进度领先于实际结果。
        if progress_callback is not None:
            progress_callback(len(batch_records))
    return predictions

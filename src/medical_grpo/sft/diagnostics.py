"""SFT 前后固定样本生成与格式退化诊断。"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any


def _format_valid(text: str) -> bool:
    thinking = text.find("## Thinking")
    final = text.find("## Final Response")
    return thinking >= 0 and final > thinking


def _final_response(text: str) -> str:
    marker = "## Final Response"
    return text.split(marker, maxsplit=1)[1].strip() if marker in text else ""


def _repetition_ratio(text: str, n: int = 4) -> float:
    tokens = re.findall(r"[A-Za-z0-9]+|[^\W\s]", text.lower(), flags=re.UNICODE)
    if len(tokens) < n * 2:
        return 0.0
    ngrams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(ngrams)


def generate_diagnostic_samples(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    sample_size: int,
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """对固定 dev 前 N 条做 greedy generation，并计算可解释的格式指标。"""

    import torch

    selected = records[: min(sample_size, len(records))]
    results: list[dict[str, Any]] = []
    original_use_cache = getattr(model.config, "use_cache", False)
    was_training = model.training
    model.config.use_cache = True
    model.eval()
    device = next(model.parameters()).device
    try:
        with torch.inference_mode():
            for record in selected:
                prompt = [{"role": "user", "content": record["question"]}]
                prompt_text = tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
                encoded = {key: value.to(device) for key, value in encoded.items()}
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                completion_ids = generated[0, encoded["input_ids"].shape[1] :]
                completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
                repetition = _repetition_ratio(completion)
                results.append(
                    {
                        "id": record["id"],
                        "question": record["question"],
                        "completion": completion,
                        "format_valid": _format_valid(completion),
                        "final_response_nonempty": bool(_final_response(completion)),
                        "repetition_4gram_ratio": round(repetition, 6),
                        "abnormal_repetition": repetition > 0.20,
                        "output_tokens": int(completion_ids.numel()),
                        "hit_max_new_tokens": int(completion_ids.numel()) >= max_new_tokens,
                    }
                )
    finally:
        model.config.use_cache = original_use_cache
        model.train(was_training)
    total = len(results)
    metrics = {
        "rows": total,
        "format_compliance": sum(item["format_valid"] for item in results) / total if total else 0.0,
        "nonempty_final_response": (
            sum(item["final_response_nonempty"] for item in results) / total if total else 0.0
        ),
        "abnormal_repetition_rate": (
            sum(item["abnormal_repetition"] for item in results) / total if total else 0.0
        ),
        "max_token_hit_rate": (
            sum(item["hit_max_new_tokens"] for item in results) / total if total else 0.0
        ),
    }
    return results, metrics


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """原子写入生成样本，防止长时间生成中断后误读半文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)

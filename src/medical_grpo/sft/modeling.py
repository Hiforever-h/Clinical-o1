"""RTX 4090 单卡 QLoRA 模型构造与硬件预检。"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import os
from typing import Any

from medical_grpo.sft.config import SFTExperimentConfig


def cuda_preflight(minimum_vram_gb: float = 23.0) -> dict[str, Any]:
    """拒绝在 CPU、低显存或不支持 BF16 的设备上误启动正式训练。"""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("train-sft 需要 CUDA GPU；本地仅可运行 sft-dry-run")
    if not torch.__version__.startswith("2.8.0"):
        raise RuntimeError(f"正式基线要求 torch 2.8.0，当前为 {torch.__version__}")
    if torch.version.cuda != "12.8":
        raise RuntimeError(f"正式基线要求 PyTorch CUDA runtime 12.8，当前为 {torch.version.cuda}")
    device_index = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(device_index)
    properties = torch.cuda.get_device_properties(device_index)
    total_vram_gb = properties.total_memory / 1024**3
    if total_vram_gb < minimum_vram_gb:
        raise RuntimeError(
            f"GPU 显存不足：{total_vram_gb:.2f}GB < {minimum_vram_gb:.2f}GB"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 GPU/CUDA/PyTorch 组合不支持 BF16")
    try:
        bnb_version = version("bitsandbytes")
    except PackageNotFoundError as exc:
        raise RuntimeError("未安装 bitsandbytes，请安装项目的 sft 依赖") from exc
    if bnb_version != "0.49.2":
        raise RuntimeError(f"正式基线要求 bitsandbytes 0.49.2，当前为 {bnb_version}")
    return {
        "device_index": device_index,
        "device_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_vram_gb": round(total_vram_gb, 2),
        "bf16_supported": True,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "bitsandbytes_version": bnb_version,
    }


def load_tokenizer(config: SFTExperimentConfig) -> Any:
    """从固定模型 revision 加载 tokenizer，并确保存在 pad token。"""

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
    tokenizer.padding_side = "right"
    return tokenizer


def build_qlora_model(config: SFTExperimentConfig) -> tuple[Any, dict[str, Any]]:
    """加载 4-bit Base，执行 k-bit 预处理并挂载 all-linear LoRA。"""

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    dtype_by_name = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    compute_dtype = dtype_by_name.get(config.quantization.compute_dtype)
    if compute_dtype is None:
        raise ValueError(f"不支持 compute_dtype={config.quantization.compute_dtype}")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=config.quantization.load_in_4bit,
        bnb_4bit_quant_type=config.quantization.quant_type,
        bnb_4bit_use_double_quant=config.quantization.use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    device_index = int(os.environ.get("LOCAL_RANK", "0"))
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
        quantization_config=quantization_config,
        dtype=compute_dtype,
        device_map={"": device_index},
        attn_implementation=config.model.attn_implementation,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config.training.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": config.training.use_reentrant},
    )
    peft_config = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias=config.lora.bias,
        use_rslora=config.lora.use_rslora,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    trainable_parameters = 0
    total_parameters = 0
    unexpected_trainable: list[str] = []
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total_parameters += count
        if parameter.requires_grad:
            trainable_parameters += count
            if "lora_" not in name:
                unexpected_trainable.append(name)
    if not trainable_parameters:
        raise RuntimeError("LoRA 没有产生任何可训练参数")
    if unexpected_trainable:
        raise RuntimeError(f"发现非 LoRA 可训练参数：{unexpected_trainable[:20]}")
    report = {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_ratio": trainable_parameters / total_parameters,
        "target_modules": config.lora.target_modules,
        "unexpected_trainable_parameters": unexpected_trainable,
    }
    return model, report

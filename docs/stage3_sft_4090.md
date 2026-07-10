# Stage 3：RTX 4090 单卡 English QLoRA SFT 操作手册

本文是 Stage 3 的服务器执行手册。目标机器为单张 RTX 4090 24GB、系统盘约 30GB、数据盘约 50GB。训练只使用 HuatuoGPT-o1 英文 SFT train/dev，不读取 MedQA、MedMCQA 或 PubMedQA。

## 1. 固定软件栈

| 组件 | 固定版本 | 说明 |
| --- | --- | --- |
| Python | 3.10 | 与项目 `pyproject.toml` 一致 |
| PyTorch | 2.8.0 | 使用官方 CUDA 12.8 wheel |
| Transformers | 4.57.6 | 固定 Qwen/Trainer 行为 |
| datasets | 5.0.0 | 固定数据映射行为 |
| TRL | 0.29.1 | 固定 `SFTTrainer` 接口 |
| PEFT | 0.19.1 | QLoRA adapter |
| Accelerate | 1.14.0 | 单 GPU 设备编排 |
| bitsandbytes | 0.49.2 | 4-bit NF4 |

不安装完整 CUDA Toolkit。PyTorch wheel 自带训练所需的 CUDA 12.8 runtime，宿主机只需安装能够支持 CUDA 12.8 的 NVIDIA 驱动。这样可减少系统盘占用，也避免系统 CUDA 与 PyTorch runtime 混用。

## 2. 磁盘布局

把代码、虚拟环境、模型缓存、pip 缓存、临时文件和训练输出全部放到 50GB 数据盘。以下假定数据盘挂载为 `/data`；租卡平台路径不同则统一替换该前缀。

```bash
export PROJECT_ROOT=/data/Clinical-o1
export VENV_ROOT=/data/venvs/clinical-o1
export HF_HOME=/data/cache/huggingface
export PIP_CACHE_DIR=/data/cache/pip
export TMPDIR=/data/tmp

mkdir -p /data/venvs "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR"
```

空间预算建议：模型缓存预留 16–18GB，Python/CUDA wheels 与虚拟环境预留 10–12GB，数据与仓库预留 2GB，checkpoint、adapter 和日志预留 8–10GB，始终保留至少 10–15GB 可用空间。不要在服务器上合并 7B 全量模型；Stage 3 只保存 LoRA adapter。

## 3. 创建环境

Ubuntu 服务器可使用平台自带的 Python 3.10，或先安装 `python3.10-venv`。在项目根目录执行：

```bash
python3.10 -m venv "$VENV_ROOT"
source "$VENV_ROOT/bin/activate"
python -m pip install --upgrade pip setuptools wheel

python -m pip install torch==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[sft,dev]'
```

安装后先清除重复 wheel 缓存，给 checkpoint 留空间：

```bash
python -m pip cache purge
```

每次新 shell 都要重新导出第 2 节环境变量并激活虚拟环境。建议把这些命令写入实例自己的启动脚本，但不要提交带平台密钥的文件。

## 4. 准备冻结数据

`data/processed/` 是可再生成的大文件目录，不提交 Git。把本机已验收的 134MB `data/processed/` 上传到服务器项目目录，或在服务器从固定 Hugging Face revisions 重建：

```bash
cd "$PROJECT_ROOT"
clinical-o1 prepare-data
clinical-o1 dry-run --component data
git restore data/manifests reports/data
clinical-o1 sft-dry-run --profile full
git status --short
```

二选一即可，优先上传已验收数据以节省租卡时间。重建命令会刷新带生成时间的版本化报告，所以在数据合同检查通过后恢复仓库内冻结的 manifest/report；被 Git 忽略的 processed JSONL 不会被删除。最后一次 `git status --short` 必须无输出，正式训练不接受 dirty Git。

无论采用哪种方式，后续 `sft-dry-run` 都会重算 train/dev 文件 SHA256，并要求 aggregate SHA 严格等于 `80619feea7de6575e376a7cc9ae144c7e5a8328ec65e52aec20e9dd645db980d`。不要手工编辑 JSONL。

## 5. CUDA 与依赖预检

```bash
nvidia-smi
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("BF16:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY

clinical-o1 sft-dry-run --profile full
```

预期：GPU 为 RTX 4090，CUDA 可用，BF16 为 `True`；full dry-run 报告 19,147/391 行、无截断、无空 completion、无 chat-template 前缀错位，并生成 `reports/sft/full_dry_run.json`。

## 6. 固定训练配置

正式参数全部来自 `configs/sft/qwen25_7b_qlora_4090.yaml`：

- Qwen2.5-7B-Instruct 固定 commit revision；
- 4-bit NF4、double quantization、BF16 compute；
- LoRA `r=16`、`alpha=32`、`dropout=0.05`、`all-linear`；
- `max_length=2048`、不 packing、只对 completion 计算 loss；
- micro batch 1、full 梯度累积 16、有效 batch 16；
- 1 epoch、学习率 `1e-4`、cosine、warmup ratio 0.03；
- gradient checkpointing、SDPA、BF16、TF32；
- TensorBoard 与本地 JSONL 双重日志；
- dev loss 选择最优 checkpoint，最多保留 3 个训练 checkpoint。

选择 2048 而非 4096，是因为冻结数据最大仅 1,451 tokens。当前实现遇到任何超长样本会直接终止，不会静默截断 CoT。

## 7. 执行顺序

建议在 `tmux` 中运行。四步必须依次通过，不要直接启动 full。

### 7.1 Smoke 前半程

先运行 5 个 optimizer steps，验证前向、反向、评估和 checkpoint 保存：

```bash
clinical-o1 train-sft \
  --profile smoke \
  --run-id qwen25_7b_huatuo_en_sft_smoke_seed42 \
  --max-steps 5
```

### 7.2 Smoke 断点恢复

从刚才的 `checkpoint-5` 恢复并训练到 10 steps，以真实验证断点恢复：

```bash
clinical-o1 train-sft \
  --profile smoke \
  --run-id qwen25_7b_huatuo_en_sft_smoke_seed42 \
  --max-steps 10 \
  --resume-from-checkpoint latest
```

验收：loss、eval loss、grad norm 均为有限值；存在 `checkpoint-10`、`best_adapter/`、`trainer_state.json`、SFT 前后固定生成和本地指标日志；显存没有 OOM。

### 7.3 Pilot

```bash
clinical-o1 train-sft \
  --profile pilot \
  --run-id qwen25_7b_huatuo_en_sft_pilot_seed42
```

Pilot 使用 1,000 条 train 与完整 391 条 dev，约 63 个 optimizer steps。验收重点是 loss 趋势、周期评估、最优 checkpoint、保存空间、前后格式合规率和生成是否出现异常重复。

### 7.4 Full

```bash
clinical-o1 train-sft \
  --profile full \
  --run-id qwen25_7b_huatuo_en_sft_full_seed42
```

Full 使用 19,147 条 train、391 条 dev，1 epoch、有效 batch 16，预计约 1,197 个 optimizer steps。实际耗时以服务器吞吐为准，不根据估算提前释放实例。

## 8. 监控与中断处理

另开一个终端：

```bash
watch -n 2 nvidia-smi
tensorboard --logdir "$PROJECT_ROOT/outputs/sft" --bind_all --port 6006
```

若 SSH 断开但进程仍在 tmux 中，重新连接即可。若训练进程中断，使用相同 `run-id` 和 `--resume-from-checkpoint latest` 恢复。不要删除已有 checkpoint，也不要新建同名 run 后从头训练。

出现以下任一情况应停止并排查：NaN/Inf、CUDA OOM、dev loss 持续恶化、固定生成格式明显崩坏、大量输出打满 1,024 new tokens、异常重复率明显升高、剩余磁盘低于 10GB。

## 9. Run 目录与备份

每个 run 位于 `outputs/sft/<run-id>/`，至少包含：

```text
config_resolved.json
command.txt
environment.json
git_state.json
hardware_preflight.json
disk_preflight.json
data_manifest_verified.json
token_audit.json
trainable_parameters.json
eval_history.jsonl
train_metrics.json
final_eval_metrics.json
generation_metrics.json
selected_checkpoint.json
trainer_state.json
best_adapter/
checkpoints/
samples/
tensorboard/
```

实例释放前，至少备份 full run 的 `best_adapter/`、`selected_checkpoint.json`、全部 JSON/JSONL、TensorBoard 日志、源数据 manifest 和必要的 trainer checkpoint。只有确认断点恢复不再需要时，才可删除非最优 checkpoint 缓解磁盘压力。

## 10. Stage 3 退出门槛

由于 Base benchmark 按项目决定延期，Stage 3 先采用内部门禁：训练/dev loss 有限、无 OOM/NaN、completion-only token 审计通过、格式合规率不低于 95%、Final Response 非空率不低于 99%、无明显重复或长度退化。

进入 GRPO 前仍需补齐同一评测协议下的 Base 与 SFT benchmark 对比。如果 SFT 在 MedQA/MedMCQA 相比 Base 下降超过 2 个百分点，则暂停 RL，优先检查 prompt 协议、学习率和灾难性遗忘。

## 11. 官方参考

- [PyTorch 2.8.0 CUDA 12.8 安装命令](https://pytorch.org/get-started/previous-versions/)
- [bitsandbytes 安装与 CUDA 支持矩阵](https://huggingface.co/docs/bitsandbytes/installation)
- [TRL SFTTrainer 与 completion-only 数据格式](https://huggingface.co/docs/trl/en/sft_trainer)
- [PEFT 量化训练与 QLoRA 指南](https://huggingface.co/docs/peft/developer_guides/quantization)

# Stage 2：Base 全量评测基线操作手册

Stage 2 的实际 Base 全量运行仍按项目计划延期，但评测代码和协议已经冻结。Base、SFT、GRPO、DAPO 必须复用同一配置和 `evaluation_contract_sha256`，否则工具会拒绝生成配对比较。

## 1. 支持的数据集和题数限制

`--datasets` 可以选择：

- `medqa`
- `medmcqa`
- `pubmedqa`
- `all`
- 多个集合，例如 `--datasets medqa medmcqa`
- 逗号形式，例如 `--datasets medqa,pubmedqa`

`--max-samples N` 表示每个被选择的数据集固定取前 N 题，不随机打乱。例如：

```bash
# 只检查 MedQA 前 20 题，direct/cot 各生成 20 条预测。
clinical-o1 eval-dry-run \
  --datasets medqa \
  --max-samples 20 \
  --protocol both

# 选择 MedQA 和 MedMCQA，每个集合评测前 100 题。
clinical-o1 eval-dry-run \
  --datasets medqa medmcqa \
  --max-samples 100 \
  --protocol direct
```

同一个 `--max-samples` 对每个集合分别生效，因此第二个示例总计检查 200 条预测。

## 2. 环境

评测复用 Stage 3 的 Python 3.10、PyTorch 2.8.0 + CUDA 12.8 环境。独立安装评测依赖可执行：

```bash
python -m pip install torch==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[evaluation,dev]'
```

Base 和 adapter 统一以 BF16 加载，不使用 4-bit。RTX 4090 24GB 可以容纳 Qwen2.5-7B BF16 权重和评测 KV cache。

## 3. CPU dry-run

在租卡前可运行：

```bash
clinical-o1 eval-dry-run \
  --profile smoke \
  --datasets all \
  --protocol both
```

smoke profile 默认每个数据集前 12 题。dry-run 不加载 7B 权重，但会：

- 重算 manifest aggregate SHA 和所选数据文件 SHA；
- 逐条通过 canonical MCQ schema；
- 构造真实 direct/cot Prompt；
- 使用 Qwen tokenizer 检查输入长度；
- 执行严格答案解析器的合成对抗检查；
- 生成完整公平评测 contract。

`--profile full` 且不传 `--max-samples` 时，会审计全部 6,456 题。

## 4. Base 小规模评测

正式全量前先验证 GPU、分片和耗时：

```bash
clinical-o1 evaluate \
  --model-type base \
  --profile full \
  --datasets medqa \
  --protocol both \
  --max-samples 20 \
  --run-id qwen25_7b_base_medqa_first20
```

这里只评测 MedQA 前 20 题，direct/cot 共 40 条预测。`--max-samples` 会进入 contract，因此不能和全量 run 直接比较。

## 5. 选择单个正式 benchmark

```bash
# 只跑完整 MedQA。
clinical-o1 evaluate \
  --model-type base \
  --datasets medqa \
  --protocol both \
  --run-id qwen25_7b_base_medqa_full

# 只跑完整 MedMCQA。
clinical-o1 evaluate \
  --model-type base \
  --datasets medmcqa \
  --protocol both \
  --run-id qwen25_7b_base_medmcqa_full

# 只跑完整 PubMedQA。
clinical-o1 evaluate \
  --model-type base \
  --datasets pubmedqa \
  --protocol both \
  --run-id qwen25_7b_base_pubmedqa_full
```

## 6. 三个数据集全量评测

```bash
clinical-o1 evaluate \
  --model-type base \
  --profile full \
  --datasets all \
  --protocol both \
  --run-id qwen25_7b_base_all_full
```

该命令生成 12,912 条预测：6,456 题乘 direct/cot 两种协议。

## 7. Adapter 评测

SFT 完成后使用完全相同的数据集、题数和 protocol：

```bash
clinical-o1 evaluate \
  --model-type adapter \
  --adapter-path outputs/sft/<sft-run-id>/best_adapter \
  --profile full \
  --datasets all \
  --protocol both \
  --run-id qwen25_7b_sft_all_full
```

评测器会记录 adapter 关键文件的目录树 SHA256，不会把另一个 adapter 误当成同一模型恢复。

## 8. 中断恢复

预测按 64 题写入原子分片。进程中断后使用完全相同参数并增加 `--resume`：

```bash
clinical-o1 evaluate \
  --model-type base \
  --profile full \
  --datasets all \
  --protocol both \
  --run-id qwen25_7b_base_all_full \
  --resume
```

恢复时会校验：

- evaluation contract；
- 模型或 adapter 身份；
- 分片题目 ID 和顺序；
- 每条预测携带的 contract SHA。

任一项不一致都会停止，不会把两个配置的结果拼在一起。

## 9. 配对比较

```bash
clinical-o1 compare-eval \
  --baseline outputs/evaluation/qwen25_7b_base_all_full \
  --candidate outputs/evaluation/qwen25_7b_sft_all_full
```

只有两个 run 的 `evaluation_contract_sha256` 完全相同时才允许比较。报告包含 Accuracy 差值、配对 bootstrap 95% 区间、Base-only/SFT-only 正确数和 exact McNemar p-value。

## 10. Run 目录

```text
outputs/evaluation/<run-id>/
├── config_resolved.json
├── evaluation_contract.json
├── model_identity.json
├── data_manifest_verified.json
├── hardware_preflight.json
├── environment.json
├── shards/
├── predictions/
├── metrics/
└── summary.json
```

逐题预测保存完整 Prompt、原始输出、解析答案、解析方式、正确性、格式状态、token 数、batch 耗时和原始 meta，便于后续分析能力变化。

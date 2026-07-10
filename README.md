# Clinical-o1

基于 `Qwen/Qwen2.5-7B-Instruct` 的英文医疗复杂推理对齐项目。主路线为：

`English QLoRA SFT → PRM/Verifier → GRPO → DAPO → final evaluation`

完整路线和阶段门禁见 [PLAN.md](PLAN.md)。目标成绩不是已取得结果；只有可由本仓库训练日志与逐题评测复现的数字，才能写入最终结论。

## 当前状态

截至 2026-07-10：

- M0/M1 已完成：工程骨架、英文 SFT/RL/评测数据、固定 revision、去重和 benchmark 污染报告均已落盘。
- M2 Base 全量评测由项目负责人决定延期，未标记为完成；最终模型对比前仍需补跑。
- M3 Stage 3 代码已实现并通过本地 19 项测试，等待 RTX 4090 服务器执行 `smoke → resume → pilot → full`。
- GRPO/DAPO 尚未实现，必须在 SFT 训练及门禁完成后渐进落地。

## 冻结数据

| 用途 | 数据源 | 最终划分 |
| --- | --- | ---: |
| SFT | `FreedomIntelligence/medical-o1-reasoning-SFT` / `en` | train 19,147；dev 391 |
| RL | `FreedomIntelligence/medical-o1-verifiable-problem` | train 24,734；dev 505 |
| Eval | MedQA-USMLE | test 1,273 |
| Eval | MedMCQA | validation 4,183 |
| Eval | PubMedQA `pqa_labeled` | test 1,000 |

SFT 的 Qwen chat-template 长度 P95 为 905 tokens，最大值为 1,451，因此 Stage 3 固定 `max_length=2048`。数据 revision、行数和 SHA256 见 [英文数据清单](data/manifests/english_mainline.json)。

## Stage 3 快速入口

目标环境为 Python 3.10、PyTorch 2.8.0 + CUDA 12.8 runtime、TRL 0.29.1。完整的 30GB 系统盘 / 50GB 数据盘部署和训练顺序见 [RTX 4090 SFT 操作手册](docs/stage3_sft_4090.md)。

```bash
clinical-o1 sft-dry-run --profile full
clinical-o1 train-sft --profile smoke --run-id sft_smoke_seed42
clinical-o1 train-sft --profile pilot --run-id sft_pilot_seed42
clinical-o1 train-sft --profile full --run-id sft_full_seed42
```

统一配置位于 [qwen25_7b_qlora_4090.yaml](configs/sft/qwen25_7b_qlora_4090.yaml)。每次训练会先检查数据哈希、Git 状态、CUDA/BF16、显存和剩余磁盘；正式 run 不允许覆盖同名目录。

注意：`data/processed/` 不提交 Git。部署服务器时需上传本机已验收目录，或先执行 `clinical-o1 prepare-data` 从固定 revisions 重建；操作手册已包含该步骤。

## 本地验收

```bash
conda activate clinical-o1-grpo
python -m ruff check src tests
python -m pytest
clinical-o1 dry-run --component all
clinical-o1 sft-dry-run --profile smoke
```

`sft-dry-run` 不加载 7B 权重，可在 CPU/MPS 机器验证数据、chat template、监督边界和 TRL 参数映射；`train-sft` 强制要求至少 23GB CUDA 显存与 BF16 支持。

## 关键产物

- [PLAN.md](PLAN.md)：从数据到 DAPO 的唯一权威实施计划。
- [Stage 3 配置](configs/sft/qwen25_7b_qlora_4090.yaml)：4090 单卡 QLoRA 的唯一参数源。
- [RTX 4090 SFT 操作手册](docs/stage3_sft_4090.md)：环境安装、磁盘布局、smoke、断点恢复、pilot 和 full 命令。
- [英文数据清单](data/manifests/english_mainline.json)：数据源 revision、划分、行数和 SHA256。
- [污染报告](reports/data/contamination_report.md)：训练数据与最终 benchmark 的污染审计。
- [数据质量报告](reports/data/data_quality_report.md)：schema、抽样和 token 长度统计。

## 工程纪律

- `outputs/` 是本机生成目录且不提交 Git；每个新实验必须写入唯一的 `outputs/<stage>/<run_id>/`。
- 租用实例释放前，必须把通过门禁的 adapter、配置、trainer state、日志与样本生成结果同步到持久存储。
- MedQA、MedMCQA、PubMedQA 最终集合不得用于 SFT 早停、Reward 选择或 RL 超参选择。
- GRPO 通过稳定性和评测门禁后才实现 DAPO；缺少 Dynamic Sampling 时不得称为完整 DAPO。

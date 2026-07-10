# Clinical-o1

基于 `Qwen/Qwen2.5-7B-Instruct` 的英文医疗复杂推理对齐项目。主路线为：

`Base baseline → English QLoRA SFT → PRM/Verifier → GRPO → DAPO → final evaluation`

完整设计、阶段门禁和实验纪律见 [PLAN.md](PLAN.md)。目标指标不是已取得结果；只有可由本仓库产物复现的实测数字才会写入最终结论。

当前版本：`v0.1.0`（第一版）。

## 当前状态

截至 2026-07-10，已完成 M0/M1，尚未启动新的 7B 英文训练：

- M0 工程冻结：历史 `outputs/` 被标记为 `legacy_zh` 并逐文件计算 SHA256；清理 3 个 Finder `.DS_Store` 后保留 147 个有效文件、1,680,922,329 bytes，树哈希为 `6cd42bde0195d874089e2b5549d30c364454defe585f30a2b2069fa298b3d9dc`。模型、adapter、checkpoint 和评测结果均保留。
- M0 工程骨架：提供可安装包、统一 CLI、当前阶段最小依赖、运行环境快照、Ruff/Pytest 和 CPU-only dry-run。
- M1 英文数据：使用 HuatuoGPT-o1 官方 English SFT 与 verifiable RL 数据，固定源 revision，完成规范化、集合内去重、SFT/RL 互斥和 benchmark 污染排除。
- M1 评测数据：MedQA test 1,273、MedMCQA validation 4,183、PubMedQA labeled 1,000；只作为最终评测数据，不参与训练。
- M1 数据门禁：51,233 条最终记录 schema 错误为 0；SFT/RL 内部重复为 0；训练与最终评测之间未决污染候选为 0。

下一阶段是 M2：先冻结 Qwen2.5-7B-Instruct 的 English Base 全量评测基线，再决定是否进入 English SFT。旧中文数据链路以及未经门禁的 SFT、评测、GRPO、DAPO 代码已删除；后续阶段到达对应里程碑时重新实现。

## 最终数据

| 用途 | 数据源 | 最终划分 |
| --- | --- | ---: |
| SFT | `FreedomIntelligence/medical-o1-reasoning-SFT` / `en` | train 19,147；dev 391 |
| RL | `FreedomIntelligence/medical-o1-verifiable-problem` | train 24,734；dev 505 |
| Eval | MedQA-USMLE | test 1,273 |
| Eval | MedMCQA | validation 4,183 |
| Eval | PubMedQA `pqa_labeled` | test 1,000 |

SFT 的 Qwen chat-template 长度 P95 为 905 tokens，最大值 1,451；当前数据没有超过 `max_length=4096` 的样本。详细 revision、行数和文件哈希见 [data/manifests/english_mainline.json](data/manifests/english_mainline.json)。

## 快速验收

使用当前项目环境：

```bash
conda activate clinical-o1-grpo
export PYTHONPATH=src

python -m medical_grpo inventory
python -m medical_grpo dry-run --component all
python -m medical_grpo snapshot-runtime
ruff check .
pytest -q
```

从固定的 Hugging Face revisions 重新生成 M1 数据：

```bash
python -m medical_grpo prepare-data
```

该命令会覆盖 `data/processed/{sft,rl,eval}`、`data/manifests` 和 `reports/data` 中由管线生成的文件，不会写入 `outputs/`。

## 关键产物

- [PLAN.md](PLAN.md)：English SFT → PRM → GRPO → DAPO 的唯一权威计划。
- [reports/artifact_inventory.json](reports/artifact_inventory.json)：不可变历史训练产物清单。
- [reports/runtime_environment.json](reports/runtime_environment.json)：当前 Python、依赖、硬件和 Git 状态。
- [data/manifests/english_mainline.json](data/manifests/english_mainline.json)：数据源 revision、划分、行数和 SHA256。
- [data/manifests/excluded_ids.jsonl](data/manifests/excluded_ids.jsonl)：所有去重和污染排除决定。
- [reports/data/contamination_report.md](reports/data/contamination_report.md)：污染审计摘要。
- [reports/data/data_quality_report.md](reports/data/data_quality_report.md)：schema、抽样和 token 长度报告。

## 工程边界

- `outputs/` 永久保留；任何重构和数据准备不得删除或覆盖历史文件。
- 新实验必须写入 `outputs/<stage>/<run_id>/`，并保存配置、数据 manifest、环境、Git 状态和标量日志。
- GRPO 通过稳定性和评测门禁后才实现完整 DAPO；缺少 Dynamic Sampling 时不得称为完整 DAPO。
- MedQA、MedMCQA、PubMedQA 最终集合不得用于超参选择、Reward 选择或早停。

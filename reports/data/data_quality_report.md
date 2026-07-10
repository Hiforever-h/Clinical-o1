# English Mainline Data Quality Report

Generated at: `2026-07-10T03:03:12.320477+00:00`

Tokenizer: `Qwen/Qwen2.5-7B-Instruct` at `a09a35458c702b33eeacc393d103063234e8bc28`

| Dataset | Rows | Invalid | Duplicate prompts | Chat/prompt P95 tokens | Over 4096 |
| --- | ---: | ---: | ---: | ---: | ---: |
| sft_train | 19147 | 0 | 0 | 905 | 0 |
| sft_dev | 391 | 0 | 0 | 872 | 0 |
| rl_train | 24734 | 0 | 0 | 85 | 0 |
| rl_dev | 505 | 0 | 0 | 87 | 0 |
| medqa_test | 1273 | 0 | 0 | 334 | 0 |
| medmcqa_validation | 4183 | 0 | 20 | 52 | 0 |
| pubmedqa_labeled | 1000 | 0 | 0 | 512 | 0 |

The pipeline validates every row. The `sampled_ids` arrays in the JSON report freeze a deterministic sample of up to 100 rows per split for manual inspection.
Official evaluation rows are retained unchanged; duplicate counts there are reported, not removed.

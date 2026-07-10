# English Mainline Contamination Report

Generated at: `2026-07-10T03:02:57.690976+00:00`

| Audit | Queries | References | Excluded | Candidates | Unresolved | Max TF-IDF cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sft_vs_final_eval | 19664 | 6456 | 126 | 126 | 0 | 0.8868 |
| rl_vs_sft | 40276 | 19538 | 14978 | 14978 | 0 | 1.0000 |
| rl_vs_final_eval | 25298 | 6456 | 59 | 59 | 0 | 0.9581 |

## Decision policy

- Exact normalized matches are excluded.
- A shared normalized contiguous span of 64 characters is excluded.
- Fuzzy pairs require both high character TF-IDF cosine and character 5-gram overlap.
- Threshold-triggered fuzzy candidates are conservatively excluded; their scores remain reviewable.
- Full candidate details are stored in `contamination_candidates.jsonl`.

"""数据源选项适配、确定性划分和去重逻辑的单元测试。"""

from __future__ import annotations

from medical_grpo.data.pipeline import _deduplicate_records, _normalize_options, _split_records


def test_normalize_options_supports_hf_shapes() -> None:
    """字典、HF label/text 和 key/value 三类选项结构应归一化一致。"""

    assert _normalize_options({"A": "alpha", "B": "beta"}) == {"A": "alpha", "B": "beta"}
    assert _normalize_options({"label": ["A", "B"], "text": ["alpha", "beta"]}) == {
        "A": "alpha",
        "B": "beta",
    }
    assert _normalize_options([{"key": "A", "value": "alpha"}, {"key": "B", "value": "beta"}]) == {
        "A": "alpha",
        "B": "beta",
    }


def test_split_records_is_deterministic_and_does_not_mutate_inputs() -> None:
    """相同 seed 必须产生相同划分，且不能原地修改输入记录。"""

    records = [{"id": str(index), "split": "train"} for index in range(10)]

    train_a, dev_a = _split_records(records, dev_ratio=0.2, seed=42)
    train_b, dev_b = _split_records(records, dev_ratio=0.2, seed=42)

    assert train_a == train_b
    assert dev_a == dev_b
    assert len(train_a) == 8
    assert len(dev_a) == 2
    assert all(record["split"] == "train" for record in records)


def test_deduplicate_records_keeps_first_and_records_lineage() -> None:
    """标准化重复题保留首条，并记录被排除样本的来源关系。"""

    records = [
        {"id": "first", "question": "What is first-line therapy?"},
        {"id": "duplicate", "question": "WHAT is first line therapy!"},
        {"id": "different", "question": "Which receptor does atropine block?"},
    ]

    retained, excluded = _deduplicate_records(records, "question", "sft_internal_dedup")

    assert [record["id"] for record in retained] == ["first", "different"]
    assert excluded == [
        {
            "audit": "sft_internal_dedup",
            "id": "duplicate",
            "decision": "exclude_duplicate_normalized_prompt",
            "reference_id": "first",
        }
    ]

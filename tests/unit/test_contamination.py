"""训练数据与 benchmark 污染审计规则的单元测试。"""

from __future__ import annotations

from medical_grpo.data.contamination import (
    AuditRecord,
    audit_records,
    char_ngram_jaccard,
    has_common_contiguous_span,
    normalize_for_audit,
    promote_review_candidates_to_exclusions,
)


def test_normalization_and_common_span() -> None:
    """大小写和标点差异不应绕过标准化及连续片段检测。"""

    left = "A patient has sudden chest pain, severe dyspnea, and hypotension after surgery."
    right = "A PATIENT has sudden chest pain; severe dyspnea and hypotension after surgery!"
    assert normalize_for_audit(left) == normalize_for_audit(right)
    assert has_common_contiguous_span(normalize_for_audit(left), normalize_for_audit(right), span=32)
    assert char_ngram_jaccard(normalize_for_audit(left), normalize_for_audit(right)) == 1.0


def test_exact_contamination_is_excluded() -> None:
    """与保护集完全相同的训练问题必须直接排除。"""

    query = [AuditRecord("train-1", "sft", "What is the treatment for bacterial meningitis?")]
    reference = [AuditRecord("eval-1", "medqa", "What is the treatment for bacterial meningitis?")]

    result = audit_records("test", query, reference)

    assert result.excluded_ids == {"train-1"}
    assert result.counts["exclude_exact"] == 1
    assert result.candidates[0].decision == "exclude_exact"


def test_clean_question_is_retained() -> None:
    """语义和文本均无关的问题不应被污染规则误删。"""

    query = [AuditRecord("train-1", "sft", "Which receptor is blocked by atropine in bradycardia?")]
    reference = [AuditRecord("eval-1", "medqa", "How is an open tibial fracture initially managed?")]

    result = audit_records("test", query, reference, review_threshold=0.95, exclude_threshold=0.99)

    assert result.excluded_ids == set()
    assert result.counts["clean"] == 1


def test_review_candidates_can_be_conservatively_excluded() -> None:
    """模糊复核候选可按 M1 保守策略提升为排除项。"""

    query = [AuditRecord("train-1", "rl", "What is the earliest manifestation of Cushing syndrome?")]
    reference = [
        AuditRecord(
            "eval-1",
            "medmcqa",
            "Which is the earliest manifestation of Cushing syndrome?",
        )
    ]
    result = audit_records("test", query, reference, review_threshold=0.50, exclude_threshold=0.99)

    promoted = promote_review_candidates_to_exclusions(result)

    assert promoted.excluded_ids == {"train-1"}
    assert promoted.counts["review"] == 0
    assert promoted.counts["exclude_fuzzy_conservative"] == 1
    assert promoted.candidates[0].decision == "exclude_fuzzy_conservative"

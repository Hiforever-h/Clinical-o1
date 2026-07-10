"""训练题与最终 benchmark 之间的重叠和污染审计。

审计依次使用标准化精确匹配、连续 64 字符、字符 5-gram Jaccard 和
字符 TF-IDF 近邻。所有阈值触发项都会落盘，便于人工复核排除原因。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import re
import unicodedata
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class AuditRecord:
    """污染检查所需的最小记录，避免把完整 CoT 放入近邻矩阵。"""

    id: str
    source: str
    text: str


@dataclass(frozen=True)
class MatchCandidate:
    """一对可疑训练题/保护集题目及其各层相似度。"""

    query_id: str
    query_source: str
    reference_id: str
    reference_source: str
    exact_match: bool
    common_64_chars: bool
    tfidf_cosine: float
    char_5gram_jaccard: float
    decision: str
    query_excerpt: str
    reference_excerpt: str


@dataclass
class AuditResult:
    """一次审计的计数、候选样本和最终排除集合。"""

    name: str
    query_count: int
    reference_count: int
    excluded_ids: set[str]
    candidates: list[MatchCandidate]
    counts: dict[str, int]
    maximum_tfidf_cosine: float

    def summary_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "query_count": self.query_count,
            "reference_count": self.reference_count,
            "excluded_count": len(self.excluded_ids),
            "candidate_count": len(self.candidates),
            "unresolved_review_count": self.counts.get("review", 0),
            "counts": self.counts,
            "maximum_tfidf_cosine": self.maximum_tfidf_cosine,
            "method": {
                "normalization": "NFKC + lowercase + Unicode alphanumeric characters only",
                "exact": True,
                "common_span": 64,
                "char_ngram_jaccard_n": 5,
                "sparse_embedding": "character TF-IDF (3-5 grams) cosine nearest neighbors",
            },
        }

    def candidates_as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(candidate) for candidate in self.candidates]


def promote_review_candidates_to_exclusions(result: AuditResult) -> AuditResult:
    """将达到模糊阈值的候选保守排除，同时保留完整分数供复核。

    M1 的原则是宁可少量损失训练题，也不能让最终评测题或 SFT 近似题进入 RL。
    """

    promoted: list[MatchCandidate] = []
    promoted_count = 0
    for candidate in result.candidates:
        if candidate.decision == "review":
            result.excluded_ids.add(candidate.query_id)
            promoted.append(replace(candidate, decision="exclude_fuzzy_conservative"))
            promoted_count += 1
        else:
            promoted.append(candidate)
    result.candidates = promoted
    result.counts["review"] = result.counts.get("review", 0) - promoted_count
    result.counts["exclude_fuzzy_conservative"] = promoted_count
    return result


def normalize_for_audit(text: str) -> str:
    """执行 NFKC、小写化并只保留各语言字母和数字。"""

    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    return "".join(character for character in normalized if character.isalnum())


def _char_ngrams(text: str, n: int = 5) -> set[str]:
    if len(text) < n:
        return {text} if text else set()
    return {text[index : index + n] for index in range(len(text) - n + 1)}


def char_ngram_jaccard(left: str, right: str, n: int = 5) -> float:
    """计算字符 n-gram 集合的 Jaccard，相比词切分更适合医学拼写变体。"""

    left_ngrams = _char_ngrams(left, n=n)
    right_ngrams = _char_ngrams(right, n=n)
    union = left_ngrams | right_ngrams
    if not union:
        return 0.0
    return len(left_ngrams & right_ngrams) / len(union)


def has_common_contiguous_span(left: str, right: str, span: int = 64) -> bool:
    """判断两个标准化文本是否共享指定长度的连续片段。"""

    if len(left) < span or len(right) < span:
        return False
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    windows = {shorter[index : index + span] for index in range(len(shorter) - span + 1)}
    return any(longer[index : index + span] in windows for index in range(len(longer) - span + 1))


def _decision(
    exact_match: bool,
    common_span: bool,
    cosine: float,
    jaccard: float,
    review_threshold: float,
    exclude_threshold: float,
) -> str:
    """按从严格到宽松的顺序给一对候选分配审计决定。"""

    if exact_match:
        return "exclude_exact"
    if common_span:
        return "exclude_common_64"
    if cosine >= exclude_threshold and jaccard >= 0.50:
        return "exclude_fuzzy_high"
    if cosine >= review_threshold or jaccard >= 0.60:
        return "review"
    return "clean"


def audit_records(
    name: str,
    query_records: Iterable[AuditRecord],
    reference_records: Iterable[AuditRecord],
    *,
    review_threshold: float = 0.82,
    exclude_threshold: float = 0.93,
    top_k: int = 3,
    batch_size: int = 512,
) -> AuditResult:
    """将待训练记录与保护集进行批量污染审计。

    精确匹配使用全局索引；连续片段与模糊指标只检查字符 TF-IDF 最近邻，
    从而让数万条长医学题的全量审计可以在普通 CPU 上完成。
    """

    queries = list(query_records)
    references = list(reference_records)
    if not queries or not references:
        return AuditResult(name, len(queries), len(references), set(), [], {}, 0.0)

    query_texts = [normalize_for_audit(record.text) for record in queries]
    reference_texts = [normalize_for_audit(record.text) for record in references]
    # 精确索引独立于 TF-IDF top-k，保证完全相同的问题不会因近邻截断而漏掉。
    exact_reference_map: dict[str, list[int]] = {}
    for index, text in enumerate(reference_texts):
        exact_reference_map.setdefault(text, []).append(index)

    # 限制特征数并使用 float32，控制全量字符向量的内存占用。
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=1 if len(references) < 20 else 2,
        max_features=200_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    reference_matrix = vectorizer.fit_transform(reference_texts)
    neighbor_count = min(max(1, top_k), len(references))
    neighbors = NearestNeighbors(n_neighbors=neighbor_count, metric="cosine", algorithm="brute", n_jobs=-1)
    neighbors.fit(reference_matrix)

    excluded_ids: set[str] = set()
    candidates: list[MatchCandidate] = []
    counts = {
        "exclude_exact": 0,
        "exclude_common_64": 0,
        "exclude_fuzzy_high": 0,
        "review": 0,
        "clean": 0,
    }
    maximum_cosine = 0.0

    # 查询分批转换和检索，避免一次构造完整 query 稀疏矩阵。
    for batch_start in range(0, len(queries), batch_size):
        batch_end = min(batch_start + batch_size, len(queries))
        batch_matrix = vectorizer.transform(query_texts[batch_start:batch_end])
        distances, indices = neighbors.kneighbors(batch_matrix, return_distance=True)

        for local_index, query_index in enumerate(range(batch_start, batch_end)):
            query = queries[query_index]
            query_text = query_texts[query_index]
            best_candidate: MatchCandidate | None = None

            exact_indices = exact_reference_map.get(query_text, [])
            candidate_indices = list(indices[local_index])
            for exact_index in exact_indices:
                if exact_index not in candidate_indices:
                    candidate_indices.insert(0, exact_index)

            distance_by_index = {
                int(reference_index): float(distance)
                for reference_index, distance in zip(indices[local_index], distances[local_index], strict=True)
            }

            for reference_index in candidate_indices:
                reference_index = int(reference_index)
                reference = references[reference_index]
                reference_text = reference_texts[reference_index]
                exact_match = query_text == reference_text
                cosine = 1.0 if exact_match else 1.0 - distance_by_index.get(reference_index, 1.0)
                cosine = max(0.0, min(1.0, cosine))
                common_span = has_common_contiguous_span(query_text, reference_text, span=64)
                jaccard = char_ngram_jaccard(query_text, reference_text, n=5)
                decision = _decision(
                    exact_match,
                    common_span,
                    cosine,
                    jaccard,
                    review_threshold,
                    exclude_threshold,
                )
                candidate = MatchCandidate(
                    query_id=query.id,
                    query_source=query.source,
                    reference_id=reference.id,
                    reference_source=reference.source,
                    exact_match=exact_match,
                    common_64_chars=common_span,
                    tfidf_cosine=round(cosine, 6),
                    char_5gram_jaccard=round(jaccard, 6),
                    decision=decision,
                    query_excerpt=re.sub(r"\s+", " ", query.text).strip()[:300],
                    reference_excerpt=re.sub(r"\s+", " ", reference.text).strip()[:300],
                )
                if best_candidate is None or (
                    candidate.decision != "clean",
                    candidate.tfidf_cosine,
                    candidate.char_5gram_jaccard,
                ) > (
                    best_candidate.decision != "clean",
                    best_candidate.tfidf_cosine,
                    best_candidate.char_5gram_jaccard,
                ):
                    best_candidate = candidate

            if best_candidate is None:
                counts["clean"] += 1
                continue

            maximum_cosine = max(maximum_cosine, best_candidate.tfidf_cosine)
            counts[best_candidate.decision] += 1
            if best_candidate.decision.startswith("exclude_"):
                excluded_ids.add(query.id)
            if best_candidate.decision != "clean":
                candidates.append(best_candidate)

    if sum(counts.values()) != len(queries):
        raise AssertionError("contamination audit counts do not add up")
    if not math.isfinite(maximum_cosine):
        maximum_cosine = 0.0

    return AuditResult(
        name=name,
        query_count=len(queries),
        reference_count=len(references),
        excluded_ids=excluded_ids,
        candidates=candidates,
        counts=counts,
        maximum_tfidf_cosine=maximum_cosine,
    )

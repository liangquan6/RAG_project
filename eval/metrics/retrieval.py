"""
检索层评估指标

- Hit@K：标准参考片段是否出现在 Top-K 检索结果中
- MRR@K：第一个命中结果的倒数排名
"""

import re
from difflib import SequenceMatcher


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "", text)
    return text


def text_similarity(reference: str, retrieved: str) -> float:
    """计算两段文本的相似度（0~1）。"""
    ref = _normalize(reference)
    ret = _normalize(retrieved)
    if not ref or not ret:
        return 0.0
    if ref in ret or ret in ref:
        return 1.0
    return SequenceMatcher(None, ref, ret).ratio()


def _is_match(reference: str, retrieved: str, threshold: float) -> bool:
    return text_similarity(reference, retrieved) >= threshold


def hit_at_k(
    retrieved: list[str],
    references: list[str],
    k: int,
    threshold: float = 0.25,
) -> bool:
    """任一 reference 在 Top-K 中命中即返回 True。"""
    top_k = retrieved[:k]
    for reference in references:
        if any(_is_match(reference, doc, threshold) for doc in top_k):
            return True
    return False


def mrr_at_k(
    retrieved: list[str],
    references: list[str],
    k: int,
    threshold: float = 0.25,
) -> float:
    """返回第一个命中位置的 reciprocal rank，未命中为 0。"""
    for rank, doc in enumerate(retrieved[:k], start=1):
        if any(_is_match(reference, doc, threshold) for reference in references):
            return 1.0 / rank
    return 0.0

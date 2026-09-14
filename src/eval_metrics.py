"""Ranking metrics — the only "truth" the ranker is scored against.

Every metric here takes a list of relevance labels *ordered by the model's
predicted score* (index 0 = the item the model ranked first) and returns a
scalar. Keep these tiny and readable: if you can't reproduce NDCG@k by hand on
a 3-item example, you can't defend the ranker's numbers in an interview.
"""
from __future__ import annotations
import numpy as np


def dcg_at_k(rels: list[float], k: int) -> float:
    """Discounted Cumulative Gain. gain = 2**rel - 1, discount = 1/log2(rank+1)."""
    rels = np.asarray(rels[:k], dtype=float)
    if rels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float(np.sum((2 ** rels - 1) * discounts))


def ndcg_at_k(ranked_rels: list[float], k: int) -> float:
    """NDCG = DCG(model order) / DCG(perfect order). 1.0 = perfect ranking."""
    ideal = sorted(ranked_rels, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(ranked_rels, k) / idcg


def recall_at_k(ranked_rels: list[float], k: int, n_positives: int) -> float:
    """Fraction of all relevant items that appear in the top k."""
    if n_positives == 0:
        return 0.0
    hits = sum(1 for r in ranked_rels[:k] if r > 0)
    return hits / n_positives


def average_precision_at_k(ranked_rels: list[float], k: int) -> float:
    """AP@k with binary relevance (rel>0 counts as a hit)."""
    hits, score = 0, 0.0
    for i, r in enumerate(ranked_rels[:k], start=1):
        if r > 0:
            hits += 1
            score += hits / i
    denom = min(sum(1 for r in ranked_rels if r > 0), k)
    return score / denom if denom else 0.0


def mean_metrics(per_user_ranked_rels: dict[str, list[float]], k: int = 10) -> dict[str, float]:
    """Average NDCG@k / Recall@k / MAP@k across users. Input: {user: rels ordered by model}."""
    ndcgs, recalls, aps = [], [], []
    for rels in per_user_ranked_rels.values():
        n_pos = sum(1 for r in rels if r > 0)
        if n_pos == 0:
            continue  # user with no relevant item in candidate set — undefined, skip
        ndcgs.append(ndcg_at_k(rels, k))
        recalls.append(recall_at_k(rels, k, n_pos))
        aps.append(average_precision_at_k(rels, k))
    return {
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"Recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"MAP@{k}": float(np.mean(aps)) if aps else 0.0,
        "n_users_scored": len(ndcgs),
    }

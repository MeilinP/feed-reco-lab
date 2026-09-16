"""Stage 2 — Learning to Rank (LambdaMART via XGBoost XGBRanker).

The retriever hands the ranker a short candidate list per user; the ranker
orders it to maximise NDCG.

Training data is grouped BY USER (that is what makes it *ranking*, not regression):
  * a user's engaged items get graded relevance  (rating>=4 -> 2, rating==3 -> 1, else 0)
  * plus sampled un-engaged items as hard negatives (relevance 0)
XGBoost's `rank:ndcg` objective (LambdaMART) optimises pairwise order within each
group, weighted by how much a swap would move NDCG. Features are the same 28
Yelp features from features.py — the ranker learns which of them separate liked
from disliked items.
"""
from __future__ import annotations
import numpy as np
import xgboost as xgb
from collections import defaultdict


def _relevance(rating: float) -> int:
    return 2 if rating >= 4 else 1 if rating >= 3 else 0


class LambdaMARTRanker:
    def __init__(self, feat_fn, feature_names, n_neg_per_user=40, seed=42):
        """feat_fn(user_id, item_id) -> list[float]. In the two-stage pipeline this
        is the 28 Yelp features PLUS the CF retrieval score, so the ranker blends
        content signal with the retriever's personalization signal."""
        self.feat_fn = feat_fn
        self.feature_names = feature_names
        self.n_neg = n_neg_per_user
        self.rng = np.random.default_rng(seed)
        self.model = None

    def fit(self, train_rows, all_items, item_weights=None, neg_source_fn=None):
        """Negatives control what task the ranker actually learns.

        neg_source_fn(user) -> [items]  : HARD-NEGATIVE MINING. Pass the retriever's
            own candidates so the ranker trains on the exact distribution it will
            rerank at serve time (retrieved-but-not-liked items). This is what makes
            a two-stage reranker work — otherwise it learns to beat easy random
            negatives and then flails when asked to reorder CF-similar candidates.
        item_weights                    : fallback — sample negatives ~ this
            distribution over all_items (e.g. popularity) when no retriever is given.
        """
        user_pos = defaultdict(list)                 # user -> [(item, rating)]
        for u, b, r in train_rows:
            user_pos[u].append((b, float(r)))
        all_items = np.asarray(all_items)
        if item_weights is not None:
            item_weights = np.asarray(item_weights, dtype=float)
            item_weights = item_weights / item_weights.sum()

        X, y, groups = [], [], []
        for u, items in user_pos.items():
            seen = {b for b, _ in items}
            rows = [(b, _relevance(r)) for b, r in items]
            if neg_source_fn is not None:
                negs = [b for b in neg_source_fn(u) if b not in seen][:self.n_neg]
            else:
                negs = [b for b in self.rng.choice(all_items, size=min(self.n_neg, len(all_items)),
                                                   replace=False, p=item_weights) if b not in seen]
            rows += [(b, 0) for b in negs]
            for b, rel in rows:
                X.append(self.feat_fn(u, b)); y.append(rel)
            groups.append(len(rows))

        dtrain = xgb.DMatrix(np.asarray(X, dtype=float), label=np.asarray(y, dtype=float))
        dtrain.set_group(groups)
        params = {
            "objective": "rank:ndcg", "eval_metric": "ndcg@10",
            "eta": 0.1, "max_depth": 4, "min_child_weight": 5,
            "subsample": 0.8, "colsample_bytree": 0.6,
            "lambda": 2.0, "alpha": 0.1, "seed": 42,
        }
        self.model = xgb.train(params, dtrain, num_boost_round=150)
        return self

    def score(self, user_id, item_ids) -> np.ndarray:
        X = np.asarray([self.feat_fn(user_id, b) for b in item_ids], dtype=float)
        return self.model.predict(xgb.DMatrix(X))

    def rank(self, user_id, item_ids) -> list[str]:
        scores = self.score(user_id, item_ids)
        return [item_ids[j] for j in np.argsort(-scores)]

    def feature_importance(self) -> list[tuple[str, float]]:
        gain = self.model.get_score(importance_type="gain")     # {'f0': .., ...}
        pairs = [(self.feature_names[int(k[1:])], v) for k, v in gain.items()]
        return sorted(pairs, key=lambda x: -x[1])

"""Stage 1 — candidate generation (retrieval).

Two retrievers:
  * PopularityRetriever  — the honest baseline. Recommends the globally most
    interacted-with businesses. Any real system must beat this.
  * ItemCFRetriever      — item-based collaborative filtering (same idea as your
    DSCI 553 CF term): items are similar if the same users engaged with both;
    a user's candidates are items similar to their history. Binary implicit
    signal (engaged = rated), cosine item-item similarity.

Retrieval optimises RECALL (get the good items into a short candidate list), not
final order — that is stage 2's job (the LTR ranker).
"""
from __future__ import annotations
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix


class PopularityRetriever:
    def __init__(self, train_rows):
        self.pop = defaultdict(int)
        for _, b, _r in train_rows:
            self.pop[b] += 1
        self.ranked = [b for b, _ in sorted(self.pop.items(), key=lambda x: -x[1])]

    def candidates(self, user_id, seen: set, n: int) -> list[str]:
        out = [b for b in self.ranked if b not in seen]
        return out[:n]

    def scores_for(self, user_id, item_ids) -> np.ndarray:
        return np.array([self.pop.get(b, 0) for b in item_ids], dtype=float)


class ItemCFRetriever:
    def __init__(self, train_rows):
        users = sorted({u for u, _, _ in train_rows})
        items = sorted({b for _, b, _ in train_rows})
        self.u_idx = {u: i for i, u in enumerate(users)}
        self.i_idx = {b: j for j, b in enumerate(items)}
        self.items = items
        self.user_hist = defaultdict(list)

        rows, cols = [], []
        for u, b, _ in train_rows:
            rows.append(self.u_idx[u]); cols.append(self.i_idx[b])
            self.user_hist[u].append(b)
        M = csr_matrix((np.ones(len(rows)), (rows, cols)),
                       shape=(len(users), len(items)))          # users x items, binary
        # cosine item-item similarity = normalized(M)^T @ normalized(M)
        col_norms = np.sqrt(np.asarray(M.multiply(M).sum(axis=0)).ravel())
        col_norms[col_norms == 0] = 1.0
        self._M = M
        self._inv_norm = 1.0 / col_norms                        # (items,)
        self._pop = np.asarray(M.sum(axis=0)).ravel()
        self._cache = {}                                        # user -> score vector

    def _item_scores_for_user(self, user_id) -> np.ndarray:
        if user_id in self._cache:
            return self._cache[user_id]
        v = self._compute_scores(user_id)
        self._cache[user_id] = v
        return v

    def _compute_scores(self, user_id) -> np.ndarray:
        hist = self.user_hist.get(user_id, [])
        if not hist:
            return self._pop * self._inv_norm * 0.0            # cold user -> no CF signal
        hist_idx = [self.i_idx[b] for b in hist if b in self.i_idx]
        if not hist_idx:
            return np.zeros(len(self.items))
        # sim(history_items, all_items) summed over the user's history
        hist_vec = self._M[:, hist_idx]                         # users x |hist|
        # co-occurrence: (all_items x users) @ (users x |hist|) -> all_items x |hist|
        cooc = self._M.T @ hist_vec                             # sparse items x |hist|
        cooc = np.asarray(cooc.sum(axis=1)).ravel()             # items
        return cooc * self._inv_norm                            # cosine-normalised

    def candidates(self, user_id, seen: set, n: int) -> list[str]:
        scores = self._item_scores_for_user(user_id)
        order = np.argsort(-scores)
        out = []
        for j in order:
            b = self.items[j]
            if b in seen or scores[j] <= 0:
                continue
            out.append(b)
            if len(out) >= n:
                break
        return out

    def scores_for(self, user_id, item_ids) -> np.ndarray:
        scores = self._item_scores_for_user(user_id)
        return np.array([scores[self.i_idx[b]] if b in self.i_idx else 0.0
                         for b in item_ids], dtype=float)

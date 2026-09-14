"""Contextual multi-armed bandit + off-policy evaluation.

This is the JD's "reinforcement learning / multi-armed bandit / balancing
exploration and exploitation" line — the one thing your portfolio was fully
missing.

Framing: at each step a user arrives with K candidate items (arms). A policy picks
ONE to show and observes reward = click (1 if the user would rate it >=4). The
policy sees only the reward of the arm it showed (bandit feedback), so it must
explore to learn.

Grounded, not arbitrary: the "would click" probabilities come from a logistic
CTR model fit on YOUR real relevance labels (rating>=4) over the 28 Yelp
features. So theta* is estimated from data; the bandits try to recover it online.

Policies:
  RandomPolicy   — uniform (this is also the *logging* policy for off-policy eval)
  GreedyOracle   — knows theta*, upper bound
  LinUCB         — theta_hat·x + alpha * sqrt(xᵀA⁻¹x)   (optimism under uncertainty)
  LinTS          — Thompson: sample theta ~ N(A⁻¹b, A⁻¹), act greedily on the sample

Off-policy evaluation (evaluate a new policy from logs of the random policy):
  Replay (Li et al. 2011) — keep a logged step only if the target policy would
                            pick the same arm; average their rewards (unbiased
                            because logging is uniform).
  IPS                     — weight each logged reward by 1{π picks logged arm}/propensity.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ---------- ground-truth reward environment (fit from real labels) ----------
class CTREnvironment:
    def __init__(self, train_rows, feature_store, seed=42):
        X, y = [], []
        for u, b, r in train_rows:
            X.append(feature_store.build(u, b))
            y.append(1 if float(r) >= 4 else 0)
        self.scaler = StandardScaler().fit(X)
        self.clf = LogisticRegression(max_iter=1000, C=1.0).fit(self.scaler.transform(X), y)
        self.fs = feature_store
        self.rng = np.random.default_rng(seed)
        self.d = len(feature_store.build(train_rows[0][0], train_rows[0][1]))

    def context(self, user_id, item_id) -> np.ndarray:
        return self.scaler.transform([self.fs.build(user_id, item_id)])[0]

    def prob(self, ctx: np.ndarray) -> float:
        return float(self.clf.predict_proba([ctx])[0, 1])

    def reward(self, ctx: np.ndarray) -> int:
        return int(self.rng.random() < self.prob(ctx))


# ---------- policies ----------
class RandomPolicy:
    name = "Random (logging)"
    def __init__(self, d, rng): self.rng = rng
    def choose(self, contexts): return int(self.rng.integers(len(contexts)))
    def update(self, ctx, reward): pass


class GreedyOracle:
    name = "Greedy oracle (upper bound)"
    def __init__(self, env): self.env = env
    def choose(self, contexts): return int(np.argmax([self.env.prob(c) for c in contexts]))
    def update(self, ctx, reward): pass


class LinUCB:
    name = "LinUCB"
    def __init__(self, d, alpha=0.6):
        self.A = np.identity(d); self.b = np.zeros(d); self.alpha = alpha
    def _theta(self): return np.linalg.solve(self.A, self.b)
    def choose(self, contexts):
        Ainv = np.linalg.inv(self.A); theta = Ainv @ self.b
        ucb = [float(theta @ x + self.alpha * np.sqrt(x @ Ainv @ x)) for x in contexts]
        return int(np.argmax(ucb))
    def update(self, ctx, reward):
        self.A += np.outer(ctx, ctx); self.b += reward * ctx


class LinTS:
    name = "LinTS (Thompson)"
    def __init__(self, d, v=0.3, rng=None):
        self.A = np.identity(d); self.b = np.zeros(d); self.v2 = v * v
        self.rng = rng or np.random.default_rng(0)
    def choose(self, contexts):
        Ainv = np.linalg.inv(self.A); mu = Ainv @ self.b
        theta = self.rng.multivariate_normal(mu, self.v2 * Ainv)
        return int(np.argmax([theta @ x for x in contexts]))
    def update(self, ctx, reward):
        self.A += np.outer(ctx, ctx); self.b += reward * ctx


# ---------- context stream ----------
def _make_rounds(env, train_rows, all_items, T, K, seed=7):
    """Each round: a user + K candidate arms (their history + random items)."""
    rng = np.random.default_rng(seed)
    users = list({u for u, _, _ in train_rows})
    user_hist = {}
    for u, b, _ in train_rows:
        user_hist.setdefault(u, []).append(b)
    all_items = np.asarray(all_items)
    rounds = []
    for _ in range(T):
        u = users[int(rng.integers(len(users)))]
        hist = user_hist.get(u, [])
        pick = list(rng.choice(hist, size=min(2, len(hist)), replace=False)) if hist else []
        pick += list(rng.choice(all_items, size=K - len(pick), replace=False))
        rng.shuffle(pick)
        contexts = np.asarray([env.context(u, b) for b in pick[:K]])
        rounds.append(contexts)
    return rounds


# ---------- online run (regret / cumulative reward) ----------
def run_online(env, policy, rounds):
    total, oracle_total = 0, 0
    for contexts in rounds:
        probs = [env.prob(c) for c in contexts]
        a = policy.choose(contexts)
        r = int(env.rng.random() < probs[a])
        policy.update(contexts[a], r)
        total += r
        oracle_total += max(probs)                 # expected reward of best arm
    return {"avg_reward": total / len(rounds),
            "regret_per_round": (oracle_total - total) / len(rounds)}


# ---------- off-policy evaluation from random-policy logs ----------
def make_logs(env, rounds, seed=11):
    rng = np.random.default_rng(seed)
    logs = []
    for contexts in rounds:
        a = int(rng.integers(len(contexts)))
        r = int(rng.random() < env.prob(contexts[a]))
        logs.append({"contexts": contexts, "action": a, "reward": r,
                     "propensity": 1.0 / len(contexts)})
    return logs


def ope_replay_and_ips(target_policy, logs):
    """Evaluate target_policy on random-policy logs. Returns (replay, ips) value estimates."""
    kept, kept_reward = 0, 0
    ips_num = 0.0
    for e in logs:
        a = target_policy.choose(e["contexts"])
        if a == e["action"]:
            kept += 1; kept_reward += e["reward"]
            target_policy.update(e["contexts"][a], e["reward"])   # learn only from matched steps
        ips_num += (1.0 if a == e["action"] else 0.0) * e["reward"] / e["propensity"]
    return {"replay_value": kept_reward / kept if kept else 0.0,
            "replay_matched": kept,
            "ips_value": ips_num / len(logs)}

"""Generate a small dataset in the EXACT Yelp schema the DSCI 553 project reads.

Why synthetic: the real Yelp dump is multi-GB and can't live in this repo. This
generator emits the same files (yelp_train.csv, yelp_val.csv, business.json,
user.json, checkin.json, photo.json, tip.json) with the same field names, so the
whole pipeline runs out-of-the-box AND swaps to real Yelp with zero code changes
(just point --data_dir at the real folder and skip this script).

The data is NOT noise: ratings come from a latent-factor model
(rating ≈ user_taste · business_profile + biases), so collaborative filtering and
a learned ranker genuinely beat a popularity baseline. That is the whole point —
if the signal were random, NDCG would be flat and the demo would prove nothing.
"""
from __future__ import annotations
import argparse, json, os, random
import numpy as np

GLOBAL_MEAN = 3.6          # Yelp ratings skew high
D = 10                     # latent dimensions


def _rating_from_affinity(affinity: float) -> int:
    r = GLOBAL_MEAN + affinity + np.random.normal(0, 0.45)
    return int(min(5, max(1, round(r))))


def generate(data_dir: str, n_users=2000, n_biz=600, seed=42):
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    random.seed(seed); np.random.seed(seed)

    # ---- latent profiles (the hidden "why") ----
    user_taste = rng.normal(0, 1, size=(n_users, D))
    biz_profile = rng.normal(0, 1, size=(n_biz, D))
    user_bias = rng.normal(0, 0.4, size=n_users)
    biz_bias = rng.normal(0, 1.0, size=n_biz)   # business quality: an independent,
    #   observable signal (surfaces in biz_stars) that CF alone can't fully capture,
    #   so the hybrid ranker (CF + content) can beat CF-only.
    biz_pop = rng.lognormal(mean=0.0, sigma=1.0, size=n_biz)   # exposure ~ popularity
    biz_pop /= biz_pop.sum()

    user_ids = [f"u{idx:05d}" for idx in range(n_users)]
    biz_ids = [f"b{idx:05d}" for idx in range(n_biz)]

    # ---- sample interactions ----
    # Exposure is mostly TASTE-DRIVEN (people browse categories they like) with a
    # little popularity. This is what lets collaborative filtering beat a
    # popularity baseline: co-liking patterns carry real taste structure. Pure
    # popularity exposure would make popularity a near-unbeatable baseline (and it
    # nearly is in raw web data — a real, documented recsys pitfall).
    aff_core = (user_taste @ biz_profile.T) / np.sqrt(D)          # (users, biz)
    train_rows, val_rows = [], []
    biz_rating_sum = np.zeros(n_biz); biz_rating_cnt = np.zeros(n_biz)
    for u in range(n_users):
        n_inter = int(rng.integers(15, 60))
        logits = 1.3 * (aff_core[u] + biz_bias)                   # taste + quality
        p_exp = np.exp(logits - logits.max()); p_exp /= p_exp.sum()
        p_exp = 0.8 * p_exp + 0.2 * biz_pop                       # blend mild popularity
        p_exp /= p_exp.sum()
        exposed = rng.choice(n_biz, size=min(n_inter, n_biz), replace=False, p=p_exp)
        for b in exposed:
            aff = 1.6 * float(aff_core[u, b]) + user_bias[u] + biz_bias[b]
            rating = _rating_from_affinity(aff)
            biz_rating_sum[b] += rating; biz_rating_cnt[b] += 1
            # 70/30 train/val split per interaction
            (train_rows if rng.random() < 0.7 else val_rows).append((user_ids[u], biz_ids[b], rating))

    with open(os.path.join(data_dir, "yelp_train.csv"), "w") as f:
        f.write("user_id,business_id,stars\n")
        for uid, bid, r in train_rows:
            f.write(f"{uid},{bid},{r}\n")
    with open(os.path.join(data_dir, "yelp_val.csv"), "w") as f:
        f.write("user_id,business_id,stars\n")
        for uid, bid, r in val_rows:
            f.write(f"{uid},{bid},{r}\n")

    # ---- business.json (observable attributes, correlated with latent quality) ----
    def price_bucket(x):  # 1..4
        return int(min(4, max(1, round(2.5 + x))))
    with open(os.path.join(data_dir, "business.json"), "w") as f:
        for b in range(n_biz):
            avg = biz_rating_sum[b] / biz_rating_cnt[b] if biz_rating_cnt[b] else GLOBAL_MEAN
            rc = int(biz_rating_cnt[b] * rng.uniform(3, 8)) + 5
            f.write(json.dumps({
                "business_id": biz_ids[b],
                "stars": round(avg * 2) / 2,                      # Yelp stars are half-steps
                "review_count": rc,
                "attributes": {
                    "RestaurantsPriceRange2": str(price_bucket(biz_bias[b])),
                    "BusinessAcceptsCreditCards": str(rng.random() > 0.15),
                    "ByAppointmentOnly": str(rng.random() > 0.85),
                    "RestaurantsReservations": str(rng.random() > 0.6),
                    "RestaurantsTableService": str(rng.random() > 0.4),
                    "WheelchairAccessible": str(rng.random() > 0.2),
                },
            }) + "\n")

    # ---- user.json ----
    with open(os.path.join(data_dir, "user.json"), "w") as f:
        for u in range(n_users):
            rc = int(rng.integers(1, 300))
            elite_years = rng.integers(0, 6)
            f.write(json.dumps({
                "user_id": user_ids[u],
                "review_count": rc,
                "average_stars": round(GLOBAL_MEAN + user_bias[u] + rng.normal(0, 0.2), 2),
                "fans": int(max(0, rng.normal(rc / 20, 3))),
                "useful": int(max(0, rng.normal(rc, rc / 2))),
                "funny": int(max(0, rng.normal(rc / 2, rc / 3))),
                "cool": int(max(0, rng.normal(rc / 2, rc / 3))),
                "friends": ",".join(str(x) for x in range(int(max(0, rng.normal(30, 20))))) or "None",
                "elite": ",".join(str(2018 + i) for i in range(elite_years)) or "None",
                "compliment_hot": int(max(0, rng.normal(5, 5))),
                "compliment_more": int(max(0, rng.normal(2, 2))),
                "compliment_plain": int(max(0, rng.normal(5, 5))),
                "compliment_cool": int(max(0, rng.normal(5, 5))),
                "compliment_funny": int(max(0, rng.normal(5, 5))),
                "compliment_writer": int(max(0, rng.normal(2, 2))),
                "compliment_photos": int(max(0, rng.normal(2, 2))),
                "yelping_since": f"{int(rng.integers(2008, 2022))}-01-01",
            }) + "\n")

    # ---- side files (checkin / photo / tip), correlated with popularity ----
    with open(os.path.join(data_dir, "checkin.json"), "w") as f:
        for b in range(n_biz):
            n = int(max(0, rng.normal(biz_rating_cnt[b], 5)))
            f.write(json.dumps({"business_id": biz_ids[b],
                                "date": ",".join(["2021-01-01 00:00:00"] * n)}) + "\n")
    with open(os.path.join(data_dir, "photo.json"), "w") as f:
        for b in range(n_biz):
            for _ in range(int(max(0, rng.normal(biz_rating_cnt[b] / 3, 3)))):
                label = rng.choice(["food", "inside", "outside", "drink", "menu"], p=[.5, .2, .1, .1, .1])
                f.write(json.dumps({"business_id": biz_ids[b], "label": str(label)}) + "\n")
    with open(os.path.join(data_dir, "tip.json"), "w") as f:
        for b in range(n_biz):
            for _ in range(int(max(0, rng.normal(biz_rating_cnt[b] / 4, 2)))):
                f.write(json.dumps({"business_id": biz_ids[b],
                                    "likes": int(max(0, rng.normal(0.5, 1)))}) + "\n")

    print(f"[make_synthetic_yelp] users={n_users} biz={n_biz} "
          f"train={len(train_rows)} val={len(val_rows)} -> {data_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/synthetic")
    ap.add_argument("--n_users", type=int, default=2000)
    ap.add_argument("--n_biz", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    generate(a.data_dir, a.n_users, a.n_biz, a.seed)

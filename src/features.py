"""The 28-feature builder — ported from the DSCI 553 `competition.py` implementation and cleaned up.

Same 13 business + 10 user + 5 interaction features as that project. Reusing
them here means the ranker trains on features with known, well-understood
semantics rather than a new black box. The only change: plain-Python/dict
loading instead of Spark RDDs, so it runs anywhere without a cluster. Feature
semantics are otherwise identical.

FEATURE_NAMES is the source of truth for column order (feeds NDCG feature-importance).
"""
from __future__ import annotations
import json, os

FEATURE_NAMES = [
    # business (13)
    "biz_stars", "biz_review_count", "price_range", "credit_cards", "by_appointment",
    "reservations", "table_service", "wheelchair", "checkin_count", "photo_count",
    "food_photo_ratio", "tip_count", "avg_tip_likes",
    # user (10)
    "usr_review_count", "usr_average_stars", "fans", "useful", "funny", "cool",
    "friends_count", "elite_years", "compliment_total", "yelping_days",
    # interaction (5)
    "star_difference", "review_ratio", "user_activity", "business_popularity",
    "normalized_star_product",
]


def _f(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def _i(v, d=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return d


def _binary(attrs, key, d=0):
    if not attrs:
        return d
    v = attrs.get(key)
    return 1 if v == "True" else 0 if v == "False" else d


def _count_csv(s):
    return 0 if not s or s == "None" else len(s.split(","))


def _yelping_days(since, ref_year=2025):
    if not since:
        return 1500
    try:
        return min((ref_year - int(since.split("-")[0])) * 365, 6000)
    except Exception:
        return 1500


class YelpFeatureStore:
    """Loads the Yelp files once, then answers build(user, biz) in O(1)."""

    DEFAULT_BIZ = {"stars": 3.5, "review_count": 30, "price_range": 2, "credit_cards": 1,
                   "by_appointment": 0, "reservations": 0, "table_service": 1, "wheelchair": 1}
    DEFAULT_USR = {"review_count": 15, "average_stars": 3.7, "fans": 1, "useful": 5, "funny": 2,
                   "cool": 2, "friends_count": 10, "elite_years": 0, "compliment_total": 2,
                   "yelping_days": 1500}
    COMPLIMENTS = ["compliment_hot", "compliment_more", "compliment_profile", "compliment_cute",
                   "compliment_list", "compliment_note", "compliment_plain", "compliment_cool",
                   "compliment_funny", "compliment_writer", "compliment_photos"]

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.biz, self.usr = {}, {}
        self.checkin, self.photo_count, self.photo_ratio = {}, {}, {}
        self.tip_count, self.tip_likes = {}, {}
        self.user_activity, self.biz_popularity = {}, {}
        self._load()

    def _jsonl(self, name):
        path = os.path.join(self.data_dir, name)
        if not os.path.exists(path):
            return
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def _load(self):
        for b in self._jsonl("business.json"):
            a = b.get("attributes") or {}
            self.biz[b["business_id"]] = {
                "stars": _f(b.get("stars"), 3.5), "review_count": _f(b.get("review_count"), 30),
                "price_range": _i(a.get("RestaurantsPriceRange2"), 2),
                "credit_cards": _binary(a, "BusinessAcceptsCreditCards", 1),
                "by_appointment": _binary(a, "ByAppointmentOnly", 0),
                "reservations": _binary(a, "RestaurantsReservations", 0),
                "table_service": _binary(a, "RestaurantsTableService", 1),
                "wheelchair": _binary(a, "WheelchairAccessible", 1),
            }
        for u in self._jsonl("user.json"):
            self.usr[u["user_id"]] = {
                "review_count": _f(u.get("review_count"), 15),
                "average_stars": _f(u.get("average_stars"), 3.7),
                "fans": _f(u.get("fans"), 1), "useful": _f(u.get("useful"), 5),
                "funny": _f(u.get("funny"), 2), "cool": _f(u.get("cool"), 2),
                "friends_count": _count_csv(u.get("friends")),
                "elite_years": _count_csv(u.get("elite")),
                "compliment_total": sum(_f(u.get(k, 0)) for k in self.COMPLIMENTS),
                "yelping_days": _yelping_days(u.get("yelping_since")),
            }
        for c in self._jsonl("checkin.json"):
            self.checkin[c["business_id"]] = _count_csv(c.get("date"))
        food, total = {}, {}
        for p in self._jsonl("photo.json"):
            bid = p["business_id"]
            total[bid] = total.get(bid, 0) + 1
            if p.get("label") == "food":
                food[bid] = food.get(bid, 0) + 1
        self.photo_count = total
        self.photo_ratio = {b: food.get(b, 0) / t if t else 0.5 for b, t in total.items()}
        likes_sum, likes_cnt = {}, {}
        for t in self._jsonl("tip.json"):
            bid = t["business_id"]
            self.tip_count[bid] = self.tip_count.get(bid, 0) + 1
            likes_sum[bid] = likes_sum.get(bid, 0) + _i(t.get("likes"), 0)
            likes_cnt[bid] = likes_cnt.get(bid, 0) + 1
        self.tip_likes = {b: likes_sum[b] / likes_cnt[b] for b in likes_sum}

    def set_activity(self, user_activity: dict, biz_popularity: dict):
        """Interaction counts from the training split (set by the pipeline)."""
        self.user_activity = user_activity
        self.biz_popularity = biz_popularity

    def build(self, user_id: str, business_id: str) -> list[float]:
        b = self.biz.get(business_id, self.DEFAULT_BIZ)
        u = self.usr.get(user_id, self.DEFAULT_USR)
        checkin = self.checkin.get(business_id, 0)
        photo_count = self.photo_count.get(business_id, 5)
        food_ratio = self.photo_ratio.get(business_id, 0.5)
        tip_count = self.tip_count.get(business_id, 3)
        avg_tip_likes = self.tip_likes.get(business_id, 0.5)
        star_diff = abs(u["average_stars"] - b["stars"])
        review_ratio = u["review_count"] / (b["review_count"] + 1)
        user_activity = self.user_activity.get(user_id, 10)
        biz_popularity = self.biz_popularity.get(business_id, 30)
        norm_star_product = (u["average_stars"] * b["stars"]) / 25.0
        return [
            b["stars"], b["review_count"], b["price_range"], b["credit_cards"],
            b["by_appointment"], b["reservations"], b["table_service"], b["wheelchair"],
            checkin, photo_count, food_ratio, tip_count, avg_tip_likes,
            u["review_count"], u["average_stars"], u["fans"], u["useful"], u["funny"],
            u["cool"], u["friends_count"], u["elite_years"], u["compliment_total"], u["yelping_days"],
            star_diff, review_ratio, user_activity, biz_popularity, norm_star_product,
        ]

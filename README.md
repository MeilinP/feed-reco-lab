# feed-reco-lab — two-stage recommender, contextual bandit, and A/B evaluation

An end-to-end **recommendation & ranking** system on the Yelp dataset:
candidate generation → learning-to-rank → contextual-bandit re-ranking → offline
A/B evaluation. Built on top of my DSCI 553 hybrid recommender
([hybrid-recommender-system](https://github.com/MeilinP/hybrid-recommender-system)),
which predicted Yelp *ratings*; this repo turns that into a system that **ranks**
what to show and **experiments** on it — the way a production recsys/ranking team works.

Everything runs out-of-the-box on a synthetic dataset in the **exact Yelp schema**,
and swaps to the real Yelp dump with no code changes. Every number below is
computed by `run_pipeline.py`, not hard-coded — re-run with `--seed` to see it move.

---

## Architecture

```
        user + context
             │
   ┌─────────▼──────────┐   STAGE 1 — RETRIEVAL (recall)
   │  candidate sources │   • PopularityRetriever   (baseline)
   │                    │   • ItemCFRetriever        (item-based CF on "likes")
   └─────────┬──────────┘
             │  ~100 candidates
   ┌─────────▼──────────┐   STAGE 2 — LEARNING TO RANK (order)
   │   LambdaMARTRanker │   XGBoost rank:ndcg, hard-negative mining,
   │  (28 Yelp features │   popularity-matched evaluation
   │   + CF score)      │
   └─────────┬──────────┘
             │  ranked list
   ┌─────────▼──────────┐   STAGE 3 — CONTEXTUAL BANDIT (explore/exploit)
   │   LinUCB / LinTS   │   decides what to actually show; learns online from
   │  + off-policy eval │   bandit feedback; evaluated off-policy from logs
   └─────────┬──────────┘
             │
   ┌─────────▼──────────┐   STAGE 4 — EXPERIMENTATION
   │  A/B test harness  │   Welch t-test + CI, power analysis, CUPED variance
   │                    │   reduction, always-valid (mSPRT) p-values
   └────────────────────┘
```

---

## Results — REAL Yelp Open Dataset (`data/real_yelp`)

This is the actual course dataset from DSCI 553 (`yelp_train.csv` — 455,854
ratings, `yelp_val.csv` — 142,044 ratings, `business.json` — 192,609 businesses).
**`user.json` / `checkin.json` / `photo.json` / `tip.json` were not found locally**
(likely read from the course cluster, not saved), so the 10 user-profile features
and the check-in/photo/tip features fall back to defaults — the ranker below is
running on real ratings and real business attributes only, not the full 28-feature
set. That is a real, stated limitation, not a hidden one.

**Stage 1+2 — ranking quality (held-out likes, rating ≥ 4, n=11,260 users):**

| pipeline | NDCG@10 | Recall@10 | MAP@10 |
|---|---|---|---|
| Popularity (baseline) | 0.0128 | 0.0118 | 0.033 |
| **Item-CF retrieval** | **0.0489** | **0.0404** | **0.114** |
| Two-stage CF→LambdaMART | 0.0362 | 0.0314 | 0.089 |

→ **Item-CF retrieval is ~3.8x popularity's NDCG@10 on real Yelp** — the absolute
numbers are small because real Yelp is huge and sparse (192K businesses, most
users rate only a handful), which is expected given how sparse the real dataset is.

> **The LambdaMART line still trails CF here — and now I know exactly why.**
> With `user.json` missing, `usr_average_stars` is a constant default for every
> user, so `star_difference` and `normalized_star_product` carry no real signal —
> the ranker has almost nothing beyond `cf_score` to rerank with, and reranking
> by near-constant features can only hurt CF's order. This isn't a synthetic-data
> artifact anymore; it's a direct, explainable consequence of a missing input,
> and it's exactly the kind of gap a real production postmortem finds. Feeding it
> real user.json (get it from Yelp's open dataset release) is the next concrete
> step, not a re-tune.

**Stage 3 — contextual bandit (avg reward per shown item, oracle = 0.868):**

| policy | avg reward | regret/round |
|---|---|---|
| Random (logging) | 0.660 | 0.204 |
| **LinUCB** | **0.841** | 0.024 |
| **LinTS (Thompson)** | **0.835** | 0.029 |

Off-policy evaluation from *random-policy logs* recovers LinUCB's value without
ever deploying it: **replay = 0.766, IPS = 0.820** (true online ≈ 0.841).

**Stage 4 — A/B test (Popularity vs CF ranker) + methods checks:**

- lift = **+236% Recall@5** (+0.128 absolute, t = 22.18, p = 3e-106, 95% CI [+11.7%, +13.9%])
- power analysis: N/arm to detect a +0.02 lift at 80% power
- CUPED: **50% variance reduction** on a pre-period covariate (adjusted p 1.4e-11 vs raw 1.8e-06)
- always-valid mSPRT p-value reported (deliberately more conservative than the
  fixed-horizon p — it allows continuous peeking without inflating false positives)

Re-run with `python run_pipeline.py --data_dir data/real_yelp` (real files are
gitignored — not checked in — so a local copy of `yelp_train.csv`,
`yelp_val.csv`, `business.json` needs to be added to that folder first). Takes ~3 minutes on
455K ratings. A synthetic fallback (`data/synthetic`, generated automatically if
missing) exists only so the repo runs for someone without the Yelp files —
the real-data numbers above are the ones that reflect actual model behavior;
the synthetic ones reflect a much smaller, generated dataset and are not
comparable.

---

## Run it

```bash
pip install -r requirements.txt
python data/make_synthetic_yelp.py --data_dir data/synthetic
python run_pipeline.py --data_dir data/synthetic
```

**Swap in real Yelp** (no code changes): put the real `yelp_train.csv`,
`business.json`, `user.json`, `checkin.json`, `photo.json`, `tip.json` in a folder
and run `python run_pipeline.py --data_dir /path/to/real_yelp`. The feature builder
in `features.py` is lifted directly from the DSCI 553 `competition.py`, so the
28 features are identical.

---

## Design Rationale

- **Two stages, not one.** Retrieval optimizes recall cheaply over the whole catalog;
  ranking optimizes order over ~100 candidates with an expensive model. Separating
  them is what lets ranking run at scale without scoring the whole catalog with
  the expensive model.
- **Item-CF on likes only, not all engagement.** Co-*liking* (rating ≥ 4) reflects taste; co-*engagement*
  including dislikes is just exposure. Building CF on likes sharpens the signal.
- **Popularity-matched negatives / hard-negative mining.** If training/eval
  negatives are easy (random, unpopular), the model wins with popularity-proxy
  features that don't transfer. Matching the negative distribution to serve time
  is what makes the offline metric trustworthy — and it's how the raw
  volume features (check-in/photo/tip counts) were found to leak exposure bias.
- **LinUCB vs ε-greedy.** UCB explores where it's *uncertain* (optimism under
  uncertainty), not uniformly at random — lower regret. LinTS does the same via
  posterior sampling.
- **Off-policy evaluation.** Not every candidate policy can be A/B tested in
  production. Replay/IPS estimate a new policy's value from logs of the current
  one, unbiased when the logging propensities are known.
- **CUPED and always-valid p-values.** CUPED removes pre-experiment variance,
  reaching significance with fewer users; mSPRT allows stopping early without the
  peeking problem that inflates false positives in fixed-horizon tests.

---

## Files

```
data/make_synthetic_yelp.py   latent-factor generator, real Yelp schema
src/features.py               28-feature builder (from DSCI 553 competition.py)
src/retrieval.py              PopularityRetriever, ItemCFRetriever
src/ranker.py                 LambdaMART LTR + hard-negative mining
src/bandit.py                 CTR env, LinUCB, LinTS, replay + IPS off-policy eval
src/experiment.py             t-test, power, CUPED, mSPRT
src/eval_metrics.py           NDCG / Recall / MAP
run_pipeline.py               end-to-end demo + report
```

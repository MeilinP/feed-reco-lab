"""A/B testing & experimentation harness for evaluating ranking/bandit policies
against each other with proper statistical rigor.

Four things a real experimentation stack needs:
  1. sample_size_for_proportion — how many users are needed before starting (power analysis)
  2. welch_ttest               — is the lift real? (unequal-variance t-test + CI)
  3. cuped                     — variance reduction using a pre-experiment covariate
                                 (CUPED, Deng et al. 2013): same power, fewer users
  4. always_valid_pvalue       — peek any time without inflating false positives (mSPRT)

Everything is plain formulas, reproducible by hand in a notebook — no black box.
"""
from __future__ import annotations
import numpy as np
from scipy import stats


def sample_size_for_proportion(p_baseline: float, mde_abs: float,
                               alpha=0.05, power=0.8) -> int:
    """Per-arm N to detect an absolute lift `mde_abs` on a rate `p_baseline`."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    p1, p2 = p_baseline, p_baseline + mde_abs
    pbar = (p1 + p2) / 2
    n = (z_a * np.sqrt(2 * pbar * (1 - pbar)) + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / mde_abs ** 2
    return int(np.ceil(n))


def welch_ttest(control: np.ndarray, treatment: np.ndarray, alpha=0.05) -> dict:
    """Unequal-variance (Welch) t-test on the mean difference, with a CI."""
    c, t = np.asarray(control, float), np.asarray(treatment, float)
    diff = t.mean() - c.mean()
    se = np.sqrt(c.var(ddof=1) / c.size + t.var(ddof=1) / t.size)
    tstat, p = stats.ttest_ind(t, c, equal_var=False)
    z = stats.norm.ppf(1 - alpha / 2)
    return {"control_mean": c.mean(), "treatment_mean": t.mean(),
            "abs_lift": diff, "rel_lift": diff / c.mean() if c.mean() else float("nan"),
            "t_stat": float(tstat), "p_value": float(p),
            "ci95": (diff - z * se, diff + z * se), "se": se}


def cuped(y_control, y_treatment, x_control, x_treatment) -> dict:
    """CUPED: adjust the metric by a pre-period covariate x to shrink variance.

    y_adj = y - theta*(x - E[x]),  theta = cov(x,y)/var(x)  (pooled).
    Returns the adjusted Welch test and the % variance reduction on the metric.
    """
    yc, yt = np.asarray(y_control, float), np.asarray(y_treatment, float)
    xc, xt = np.asarray(x_control, float), np.asarray(x_treatment, float)
    x_all = np.concatenate([xc, xt]); y_all = np.concatenate([yc, yt])
    theta = np.cov(x_all, y_all, ddof=1)[0, 1] / np.var(x_all, ddof=1)
    xbar = x_all.mean()
    yc_adj = yc - theta * (xc - xbar)
    yt_adj = yt - theta * (yt * 0 + xt - xbar)
    var_reduction = 1 - np.var(np.concatenate([yc_adj, yt_adj]), ddof=1) / np.var(y_all, ddof=1)
    return {"theta": float(theta), "variance_reduction": float(var_reduction),
            "adjusted_test": welch_ttest(yc_adj, yt_adj)}


def always_valid_pvalue(control, treatment, tau2=1.0) -> float:
    """mSPRT always-valid p-value (Gaussian mixture) — safe under continuous peeking.

    Lets the test stop the moment it crosses significance, without the
    fixed-horizon p-value's inflated false-positive rate under continuous peeking.
    """
    c, t = np.asarray(control, float), np.asarray(treatment, float)
    n = min(c.size, t.size)
    diff = t[:n].mean() - c[:n].mean()
    s2 = (c[:n].var(ddof=1) + t[:n].var(ddof=1)) / 2
    if s2 == 0:
        return 1.0
    v = 2 * s2 / n
    like = np.sqrt(v / (v + tau2)) * np.exp((tau2 * (diff ** 2)) / (2 * v * (v + tau2)))
    return float(min(1.0, 1.0 / like))


def per_user_reward(policy_choices: dict) -> np.ndarray:
    """Helper: dict {user: [rewards]} -> per-user mean reward array."""
    return np.array([np.mean(v) for v in policy_choices.values() if len(v)])

"""Sampling designs and estimators for a fixed labeling budget.

Three designs spend the SAME budget of human labels:

1. SRS -- simple random sampling; plain sample mean, Wilson CI.
2. Stratified -- proportional allocation across segments (strata);
   stratified mean with a finite-population-corrected SE, Wilson CIs
   per segment.
3. Risk-weighted -- systematic probability-proportional-to-risk
   sampling with exact inclusion probabilities pi_i, estimated with
   the Horvitz-Thompson estimator. A named floor, ``pi_floor``, mixes
   a uniform component into every inclusion probability:

       pi_i = pi_floor + (budget - N * pi_floor) * risk_i / sum(risk)

   so no case's inclusion probability falls below ``pi_floor`` and no
   Horvitz-Thompson weight exceeds ``1 / pi_floor``. The floor is a
   real mechanism in the sampler: it caps the variance blow-up when
   the risk score is miscalibrated, at a measurable efficiency cost
   when the score is good.

Flag rule (pre-declared, used identically for every design): a segment
is FLAGGED when its 95% CI lower bound sits above the design's own
overall error-rate estimate AND the CI half-width is at most
``FLAG_MAX_HALF_WIDTH`` -- separation plus enough precision to act on.
"""
from __future__ import annotations

import math
import random

Z95 = 1.96
FLAG_MAX_HALF_WIDTH = 0.12
DEFAULT_BUDGET = 300
DEFAULT_PI_FLOOR = 0.01

DESIGNS = ("srs", "stratified", "risk_weighted")


# ---------------------------------------------------------------- CIs

def wilson_ci(k, n, z=Z95):
    """Wilson score interval for k successes in n trials."""
    if n <= 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (max(0.0, (centre - margin) / denom),
            min(1.0, (centre + margin) / denom))


def _normal_ci(est, se, z=Z95):
    return (max(0.0, est - z * se), min(1.0, est + z * se))


def flag_segment(seg_est, overall_est):
    """Apply the pre-declared flag rule to one segment estimate dict."""
    lo, hi = seg_est["ci95"]
    return bool(lo > overall_est and (hi - lo) / 2 <= FLAG_MAX_HALF_WIDTH)


# ------------------------------------------------------------- samplers

def _check_budget(cases, budget):
    if budget < 1:
        raise ValueError("budget must be >= 1")
    if budget > len(cases):
        raise ValueError("budget exceeds queue size")


def srs_sample(cases, budget, seed):
    """Simple random sample without replacement: exactly `budget` ids."""
    _check_budget(cases, budget)
    rng = random.Random(seed)
    return rng.sample([c["id"] for c in cases], budget)


def stratified_sample(cases, budget, seed):
    """Proportional allocation across segments, SRS within each.

    Largest-remainder rounding keeps the total exactly `budget`.
    """
    _check_budget(cases, budget)
    rng = random.Random(seed)
    by_segment = {}
    for c in cases:
        by_segment.setdefault(c["segment"], []).append(c["id"])
    n_total = len(cases)
    raw = {s: budget * len(ids) / n_total for s, ids in by_segment.items()}
    alloc = {s: int(v) for s, v in raw.items()}
    remainder = budget - sum(alloc.values())
    by_frac = sorted(raw, key=lambda s: raw[s] - alloc[s], reverse=True)
    for s in by_frac[:remainder]:
        alloc[s] += 1
    sample = []
    for segment in sorted(by_segment):
        sample.extend(rng.sample(by_segment[segment], alloc[segment]))
    return sample


def inclusion_probabilities(cases, budget, pi_floor=DEFAULT_PI_FLOOR):
    """Exact inclusion probabilities for risk-weighted sampling.

    pi_i = pi_floor + (budget - N*pi_floor) * risk_i / sum(risk).
    Sums exactly to `budget` (up to float error); every pi_i >=
    pi_floor; requires N * pi_floor < budget.
    """
    _check_budget(cases, budget)
    if pi_floor < 0:
        raise ValueError("pi_floor must be >= 0")
    n = len(cases)
    if n * pi_floor >= budget:
        raise ValueError("pi_floor too large: N * pi_floor must be < budget")
    total_risk = sum(c["risk"] for c in cases)
    if total_risk <= 0:
        raise ValueError("risk scores sum to zero")
    scale = (budget - n * pi_floor) / total_risk
    pis = [pi_floor + scale * c["risk"] for c in cases]
    if max(pis) >= 1.0:
        raise ValueError("an inclusion probability reached 1; "
                         "lower the budget or the risk skew")
    return pis


def risk_weighted_sample(cases, budget, seed, pi_floor=DEFAULT_PI_FLOOR):
    """Systematic PPS sample with the exact inclusion probabilities.

    A random start in [0, 1) and unit steps through the cumulative-pi
    axis select exactly `budget` distinct cases, each included with
    exactly its pi_i (valid because every pi_i < 1). Returns
    (sample_ids, pis) where pis is aligned with `cases` order (case
    ids are 0..N-1 in generator order).
    """
    pis = inclusion_probabilities(cases, budget, pi_floor=pi_floor)
    rng = random.Random(seed)
    start = rng.random()
    sample = []
    cum = 0.0
    j = 0
    for case, pi in zip(cases, pis):
        cum += pi
        while j < budget and start + j < cum:
            sample.append(case["id"])
            j += 1
    if len(sample) != budget:  # float guard; should not trigger
        chosen = set(sample)
        rest = sorted((c for c in cases if c["id"] not in chosen),
                      key=lambda c: -pis[c["id"]])
        while len(sample) < budget:
            sample.append(rest.pop(0)["id"])
        sample = sample[:budget]
    return sample, pis


# ------------------------------------------------------------ estimators

def _by_id(cases):
    return {c["id"]: c for c in cases}


def estimate_srs(cases, sample_ids):
    """Plain mean + Wilson CI overall; Wilson CI per segment."""
    by_id = _by_id(cases)
    picked = [by_id[i] for i in sample_ids]
    n = len(picked)
    k = sum(c["is_error"] for c in picked)
    est = k / n
    lo, hi = wilson_ci(k, n)
    segments = _plain_segments(picked)
    se = math.sqrt(est * (1 - est) / n) if n else 0.0
    return _result(est, lo, hi, segments, n, se=se)


def estimate_stratified(cases, sample_ids):
    """Stratified mean (proportional weights) + normal CI from the
    stratified SE with finite-population correction; Wilson per segment."""
    by_id = _by_id(cases)
    picked = [by_id[i] for i in sample_ids]
    sizes = {}
    for c in cases:
        sizes[c["segment"]] = sizes.get(c["segment"], 0) + 1
    n_total = len(cases)
    est = 0.0
    var = 0.0
    for segment, n_seg in sizes.items():
        sel = [c["is_error"] for c in picked if c["segment"] == segment]
        if not sel:
            continue
        w = n_seg / n_total
        p = sum(sel) / len(sel)
        est += w * p
        fpc = 1 - len(sel) / n_seg
        var += w * w * fpc * p * (1 - p) / len(sel)
    se = math.sqrt(var)
    lo, hi = _normal_ci(est, se)
    return _result(est, lo, hi, _plain_segments(picked), len(picked), se=se)


def ht_estimate(cases, sample_ids, pis):
    """Horvitz-Thompson estimate of the overall error rate.

    est = (1/N) * sum_{sampled} y_i / pi_i -- unbiased for the true
    rate under the stated inclusion probabilities. Variance uses the
    standard with-replacement / Poisson-style HT variance estimator
    sum (1 - pi_i) * (y_i / pi_i)^2 / N^2 over the sample.
    """
    by_id = _by_id(cases)
    n_total = len(cases)
    total = 0.0
    var_sum = 0.0
    for i in sample_ids:
        pi = pis[i]
        y = by_id[i]["is_error"]
        total += y / pi
        var_sum += (1 - pi) * (y / pi) ** 2
    est = total / n_total
    se = math.sqrt(var_sum) / n_total
    return est, se


def estimate_risk_weighted(cases, sample_ids, pis):
    """HT overall estimate; per-segment HT estimates with HT CIs."""
    est, se = ht_estimate(cases, sample_ids, pis)
    lo, hi = _normal_ci(est, se)
    by_id = _by_id(cases)
    sizes = {}
    for c in cases:
        sizes[c["segment"]] = sizes.get(c["segment"], 0) + 1
    segments = {}
    for segment, n_seg in sizes.items():
        sel = [i for i in sample_ids if by_id[i]["segment"] == segment]
        total = sum(by_id[i]["is_error"] / pis[i] for i in sel)
        var_sum = sum((1 - pis[i]) * (by_id[i]["is_error"] / pis[i]) ** 2
                      for i in sel)
        s_est = total / n_seg
        s_se = math.sqrt(var_sum) / n_seg
        s_lo, s_hi = _normal_ci(s_est, s_se)
        segments[segment] = {"estimate": round(s_est, 6),
                             "ci95": [round(s_lo, 6), round(s_hi, 6)],
                             "se": round(s_se, 6), "n_sampled": len(sel)}
    out = _result(est, lo, hi, segments, len(sample_ids), se=se)
    return out


def _plain_segments(picked):
    segments = {}
    by_segment = {}
    for c in picked:
        by_segment.setdefault(c["segment"], []).append(c["is_error"])
    for segment, ys in sorted(by_segment.items()):
        k, n = sum(ys), len(ys)
        lo, hi = wilson_ci(k, n)
        segments[segment] = {"estimate": round(k / n, 6),
                             "ci95": [round(lo, 6), round(hi, 6)],
                             "n_sampled": n}
    return segments


def _result(est, lo, hi, segments, n_sampled, se=None):
    out = {"estimate": round(est, 6),
           "ci95": [round(lo, 6), round(hi, 6)],
           "n_sampled": n_sampled,
           "segments": segments}
    if se is not None:
        out["se"] = round(se, 6)
    for seg in out["segments"].values():
        seg["flagged"] = flag_segment(seg, out["estimate"])
    return out


def run_design(cases, design, budget, seed, pi_floor=DEFAULT_PI_FLOOR):
    """Sample with one design at `budget` and return its estimate."""
    if design == "srs":
        return estimate_srs(cases, srs_sample(cases, budget, seed))
    if design == "stratified":
        return estimate_stratified(cases, stratified_sample(cases, budget, seed))
    if design == "risk_weighted":
        sample, pis = risk_weighted_sample(cases, budget, seed,
                                           pi_floor=pi_floor)
        return estimate_risk_weighted(cases, sample, pis)
    raise ValueError(f"unknown design: {design}")

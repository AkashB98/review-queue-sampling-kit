"""Seeded synthetic review queue (no real data, no network).

SYNTHETIC DATA ONLY. A frozen classifier has flagged a queue of cases;
each case belongs to a segment, carries a per-case risk score from the
classifier, and has a ground-truth error label (1 = the classifier's
decision was wrong) that only a human reviewer can reveal -- at the cost
of one label from a fixed budget.

Planted structure (all known, all checkable):

* ~6,000 cases across four segments.
* Exactly ONE planted bad segment (``new_cohort``): its true error rate
  is ~40%, versus ~3-5% in the other three segments.
* In the main scenario the risk score genuinely correlates with errors:
  errors draw risk from [0.45, 1.0], non-errors from a right-skewed
  distribution on [0, 0.35] with mass near zero.
* In the miscalibrated scenario the SAME risk values are shuffled across
  cases, so the score carries no information about the labels. The
  multiset of scores and the labels are unchanged; only the pairing is
  destroyed. This is the failure case for risk-weighted sampling.
"""
from __future__ import annotations

import random

# (segment name, n cases, planted error rate)
SEGMENTS = (
    ("standard_a", 2000, 0.04),
    ("standard_b", 1800, 0.05),
    ("standard_c", 1500, 0.03),
    ("new_cohort", 700, 0.40),
)
BAD_SEGMENT = "new_cohort"
QUEUE_SIZE = sum(n for _name, n, _p in SEGMENTS)

DEFAULT_SEED = 42


def generate_queue(seed=DEFAULT_SEED, miscalibrated=False):
    """Return the queue as a list of dicts, deterministic in `seed`.

    Each case: {"id", "segment", "is_error", "risk"}. With
    miscalibrated=True the risk scores are shuffled across cases with
    the same seeded RNG stream, so scores no longer correlate with
    labels.
    """
    rng = random.Random(seed)
    cases = []
    cid = 0
    for name, n, p_err in SEGMENTS:
        for _ in range(n):
            is_error = 1 if rng.random() < p_err else 0
            u = rng.random()
            if is_error:
                risk = 0.45 + 0.55 * u
            else:
                risk = 0.35 * u * u
            cases.append({"id": cid, "segment": name,
                          "is_error": is_error, "risk": risk})
            cid += 1
    if miscalibrated:
        risks = [c["risk"] for c in cases]
        rng.shuffle(risks)
        for case, risk in zip(cases, risks):
            case["risk"] = risk
    return cases


def true_overall_rate(cases):
    if not cases:
        raise ValueError("empty queue")
    return sum(c["is_error"] for c in cases) / len(cases)


def true_segment_rates(cases):
    totals, errors = {}, {}
    for c in cases:
        totals[c["segment"]] = totals.get(c["segment"], 0) + 1
        errors[c["segment"]] = errors.get(c["segment"], 0) + c["is_error"]
    return {s: errors[s] / totals[s] for s in totals}


def segment_sizes(cases):
    sizes = {}
    for c in cases:
        sizes[c["segment"]] = sizes.get(c["segment"], 0) + 1
    return sizes

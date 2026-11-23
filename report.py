"""Full report: headline comparison, budget sweep, multi-seed study.

Everything is deterministic: the queue, the samples, and the study all
use pinned seeds, so build_report() returns byte-identical JSON across
calls.
"""
from __future__ import annotations

import math

from generator import (BAD_SEGMENT, DEFAULT_SEED, generate_queue,
                       segment_sizes, true_overall_rate,
                       true_segment_rates)
from sampling import (DEFAULT_BUDGET, DEFAULT_PI_FLOOR, DESIGNS,
                      ht_estimate, risk_weighted_sample, run_design,
                      srs_sample, estimate_srs)

SAMPLE_SEED = 1001
BUDGETS = (150, 300, 600)
STUDY_REPS = 24
STUDY_DATASET_SEED0 = 1000
STUDY_SAMPLE_SEED0 = 5000


def _emp_se(values):
    if len(values) < 2:
        raise ValueError("need >= 2 values")
    m = sum(values) / len(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return m, math.sqrt(var)


def multiseed_study(reps=STUDY_REPS):
    """Empirical SEs across fresh queues + fresh samples, per scenario.

    For each rep a NEW queue is generated (dataset seed varies) and each
    design samples it (sample seed varies), so the empirical SE mixes
    queue-realisation and sampling variation exactly as a repeated
    deployment would. Designs: SRS, risk-weighted with no floor, and
    risk-weighted with the default floor.
    """
    study = {}
    for label, miscal in (("good_risk", False), ("miscalibrated_risk", True)):
        ests = {"srs": [], "risk_weighted_no_floor": [],
                "risk_weighted_floored": []}
        truths = []
        for r in range(reps):
            cases = generate_queue(seed=STUDY_DATASET_SEED0 + r,
                                   miscalibrated=miscal)
            truths.append(true_overall_rate(cases))
            seed = STUDY_SAMPLE_SEED0 + r
            ests["srs"].append(estimate_srs(
                cases, srs_sample(cases, DEFAULT_BUDGET, seed))["estimate"])
            for key, floor in (("risk_weighted_no_floor", 0.0),
                               ("risk_weighted_floored", DEFAULT_PI_FLOOR)):
                sample, pis = risk_weighted_sample(
                    cases, DEFAULT_BUDGET, seed, pi_floor=floor)
                ests[key].append(ht_estimate(cases, sample, pis)[0])
        block = {"reps": reps, "mean_truth": round(sum(truths) / reps, 6)}
        for key, values in ests.items():
            mean, se = _emp_se(values)
            block[key] = {"mean_estimate": round(mean, 6),
                          "empirical_se": round(se, 6),
                          "bias_vs_mean_truth": round(
                              mean - sum(truths) / reps, 6)}
        study[label] = block
    return study


def build_report(budget=DEFAULT_BUDGET, pi_floor=DEFAULT_PI_FLOOR):
    if budget < 1:
        raise ValueError("budget must be >= 1")
    cases = generate_queue(seed=DEFAULT_SEED)
    truth = true_overall_rate(cases)
    seg_truth = true_segment_rates(cases)

    designs = {d: run_design(cases, d, budget, SAMPLE_SEED,
                             pi_floor=pi_floor) for d in DESIGNS}

    sweep = {}
    for b in BUDGETS:
        sweep[str(b)] = {d: run_design(cases, d, b, SAMPLE_SEED,
                                       pi_floor=pi_floor)["estimate"]
                         for d in DESIGNS}
        sweep[str(b)]["risk_weighted_se"] = run_design(
            cases, "risk_weighted", b, SAMPLE_SEED,
            pi_floor=pi_floor)["se"]
        sweep[str(b)]["srs_se"] = run_design(
            cases, "srs", b, SAMPLE_SEED)["se"]

    study = multiseed_study()
    good, miscal = study["good_risk"], study["miscalibrated_risk"]
    claims = {
        "risk_weighted_flags_bad_segment": designs["risk_weighted"]
            ["segments"][BAD_SEGMENT]["flagged"],
        "srs_flags_bad_segment": designs["srs"]
            ["segments"][BAD_SEGMENT]["flagged"],
        "stratified_flags_bad_segment": designs["stratified"]
            ["segments"][BAD_SEGMENT]["flagged"],
        "ht_close_to_truth_at_headline_budget": abs(
            designs["risk_weighted"]["estimate"] - truth) <= 0.02,
        "ht_unbiased_across_seeds": abs(
            good["risk_weighted_floored"]["bias_vs_mean_truth"]) <= 0.01,
        "risk_weighted_se_below_srs_at_headline": (
            designs["risk_weighted"]["se"] < designs["srs"]["se"]),
        "miscalibrated_no_floor_se_blows_up": (
            miscal["risk_weighted_no_floor"]["empirical_se"]
            > 2 * miscal["srs"]["empirical_se"]),
        "floor_caps_miscalibrated_blowup": (
            miscal["risk_weighted_floored"]["empirical_se"]
            < 0.5 * miscal["risk_weighted_no_floor"]["empirical_se"]),
        "floor_cost_in_good_scenario_is_small": (
            good["risk_weighted_floored"]["empirical_se"]
            < 1.5 * good["risk_weighted_no_floor"]["empirical_se"]),
    }
    _flag_keys = ("srs_flags_bad_segment", "stratified_flags_bad_segment")
    _others = [v for k, v in claims.items() if k not in _flag_keys]
    claims["all_headline_claims_hold"] = bool(
        all(_others) and not claims["srs_flags_bad_segment"]
        and not claims["stratified_flags_bad_segment"])

    return {
        "queue": {"size": len(cases), "seed": DEFAULT_SEED,
                  "segment_sizes": segment_sizes(cases),
                  "true_overall_error_rate": round(truth, 6),
                  "true_segment_error_rates": {
                      k: round(v, 6) for k, v in sorted(seg_truth.items())},
                  "bad_segment": BAD_SEGMENT},
        "budget": budget,
        "pi_floor": pi_floor,
        "sample_seed": SAMPLE_SEED,
        "designs": designs,
        "budget_sweep": sweep,
        "multiseed_study": study,
        "claims": claims,
    }

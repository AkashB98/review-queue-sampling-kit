"""Golden evals for review-queue-sampling-kit.

Deterministic: eval_report.json is byte-identical across runs
(no timestamps, no temp paths). Run: python3 evals/run_evals.py
"""
from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli import main as cli_main  # noqa: E402
from generator import BAD_SEGMENT, generate_queue, true_segment_rates  # noqa: E402
from report import build_report  # noqa: E402
from sampling import (inclusion_probabilities, risk_weighted_sample,  # noqa: E402
                      srs_sample, stratified_sample)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = []


def check(name, passed, detail=""):
    RESULTS.append({"name": name, "passed": bool(passed), "detail": detail})
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}"
          + (f" - {detail}" if detail else ""))
    return passed


def _run_cli(argv):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return cli_main(argv)


def main():
    print("== golden evals: review-queue-sampling-kit ==")
    report = build_report()
    q = report["queue"]
    designs = report["designs"]
    study = report["multiseed_study"]
    good, miscal = study["good_risk"], study["miscalibrated_risk"]
    cases = generate_queue()

    rates = true_segment_rates(cases)
    others = [r for s, r in rates.items() if s != BAD_SEGMENT]
    check("G1 planted structure: 6000 cases, exactly one bad segment "
          "(~0.41 vs ~0.02-0.05 elsewhere)",
          len(cases) == 6000 and 0.35 <= rates[BAD_SEGMENT] <= 0.45
          and all(0.01 <= r <= 0.07 for r in others),
          f"bad={round(rates[BAD_SEGMENT], 6)} others="
          f"{[round(r, 6) for r in others]} overall={q['true_overall_error_rate']}")

    exact = True
    for b in (150, 300, 600):
        for sample in (srs_sample(cases, b, 7),
                       stratified_sample(cases, b, 7),
                       risk_weighted_sample(cases, b, 7)[0]):
            exact = exact and len(sample) == b and len(set(sample)) == b
    check("G2 every sampler spends exactly the budget (150/300/600, unique)",
          exact, "all designs x all budgets exact and duplicate-free")

    rw_seg = designs["risk_weighted"]["segments"][BAD_SEGMENT]
    srs_seg = designs["srs"]["segments"][BAD_SEGMENT]
    check("G3 at budget 300 risk-weighted HT flags the bad segment; SRS does not",
          rw_seg["flagged"] is True and srs_seg["flagged"] is False,
          f"rw est={rw_seg['estimate']} ci={rw_seg['ci95']} n={rw_seg['n_sampled']} "
          f"| srs est={srs_seg['estimate']} ci={srs_seg['ci95']} n={srs_seg['n_sampled']}")

    bias = good["risk_weighted_floored"]["bias_vs_mean_truth"]
    check("G4 HT is approximately unbiased across 24 fresh queues "
          "(|mean estimate - mean truth| <= 0.01)",
          abs(bias) <= 0.01 and study["good_risk"]["reps"] >= 20,
          f"mean_est={good['risk_weighted_floored']['mean_estimate']} "
          f"mean_truth={good['mean_truth']} bias={bias}")

    check("G5 risk-weighted beats SRS precision in the good scenario "
          "(headline SE and multi-seed empirical SE)",
          designs["risk_weighted"]["se"] < designs["srs"]["se"]
          and good["risk_weighted_floored"]["empirical_se"]
          < good["srs"]["empirical_se"],
          f"headline se rw={designs['risk_weighted']['se']} "
          f"srs={designs['srs']['se']} | empirical rw="
          f"{good['risk_weighted_floored']['empirical_se']} "
          f"srs={good['srs']['empirical_se']}")

    check("G6 miscalibrated risk: unfloored HT empirical SE blows up "
          "(> 2x SRS)",
          miscal["risk_weighted_no_floor"]["empirical_se"]
          > 2 * miscal["srs"]["empirical_se"],
          f"ht_no_floor={miscal['risk_weighted_no_floor']['empirical_se']} "
          f"srs={miscal['srs']['empirical_se']}")

    check("G7 the pi_floor caps the blow-up (< 0.5x unfloored) at a small "
          "good-scenario cost (< 1.5x unfloored)",
          miscal["risk_weighted_floored"]["empirical_se"]
          < 0.5 * miscal["risk_weighted_no_floor"]["empirical_se"]
          and good["risk_weighted_floored"]["empirical_se"]
          < 1.5 * good["risk_weighted_no_floor"]["empirical_se"],
          f"miscal floored={miscal['risk_weighted_floored']['empirical_se']} "
          f"vs no_floor={miscal['risk_weighted_no_floor']['empirical_se']} | "
          f"good floored={good['risk_weighted_floored']['empirical_se']} "
          f"vs no_floor={good['risk_weighted_no_floor']['empirical_se']}")

    pis0 = inclusion_probabilities(cases, 300, pi_floor=0.0)
    pis1 = inclusion_probabilities(cases, 300, pi_floor=0.01)
    check("G8 floor mechanism is real: min pi >= pi_floor and pis sum to budget",
          min(pis1) >= 0.01 - 1e-12 and min(pis0) < 0.001
          and abs(sum(pis1) - 300) < 1e-6,
          f"min pi floor=0: {round(min(pis0), 6)}; floor=0.01: "
          f"{round(min(pis1), 6)}; sum={round(sum(pis1), 6)}")

    codes = (_run_cli(["check"]), _run_cli(["check", "--budget", "0"]),
             _run_cli(["check", "--input", "/tmp/no-such-rqsk-xyz.json"]))
    check("G9 CLI check exit codes 0 / 2 / 2 (pass / bad budget / missing file)",
          codes == (0, 2, 2), f"codes={codes}")

    sweep = report["budget_sweep"]
    check("G10 report deterministic and RW SE falls as budget grows "
          "(150 > 300 > 600)",
          json.dumps(report, sort_keys=True)
          == json.dumps(build_report(), sort_keys=True)
          and sweep["150"]["risk_weighted_se"]
          > sweep["300"]["risk_weighted_se"]
          > sweep["600"]["risk_weighted_se"],
          f"rw_se 150={sweep['150']['risk_weighted_se']} "
          f"300={sweep['300']['risk_weighted_se']} "
          f"600={sweep['600']['risk_weighted_se']}")

    passed = sum(1 for r in RESULTS if r["passed"])
    payload = {"total": len(RESULTS), "passed": passed,
               "failed": len(RESULTS) - passed, "results": RESULTS,
               "headline": {
                   "true_overall_error_rate": q["true_overall_error_rate"],
                   "budget": report["budget"],
                   "pi_floor": report["pi_floor"],
                   "designs": {d: {"estimate": designs[d]["estimate"],
                                   "se": designs[d].get("se"),
                                   "ci95": designs[d]["ci95"],
                                   "bad_segment": designs[d]["segments"]
                                   [BAD_SEGMENT]}
                               for d in sorted(designs)},
                   "multiseed_study": study,
                   "claims": report["claims"],
               }}
    with open(os.path.join(HERE, "eval_report.json"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\n{passed}/{len(RESULTS)} golden evals passed")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())

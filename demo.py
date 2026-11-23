"""End-to-end demo: queue -> three designs at one budget -> flag ->
multi-seed study (good risk vs miscalibrated risk, floor on/off).

SYNTHETIC DATA ONLY. Runs fully offline; every stage asserts.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli import EXIT_OK, main as cli_main  # noqa: E402
from generator import BAD_SEGMENT  # noqa: E402
from report import build_report  # noqa: E402


def run():
    print("== review-queue-sampling-kit demo ==")
    report = build_report()
    q = report["queue"]
    print(f"\n== 1. synthetic queue ({q['size']} cases, seed {q['seed']}) ==")
    print(f"   true overall error rate: {q['true_overall_error_rate']}")
    for seg, rate in sorted(q["true_segment_error_rates"].items()):
        marker = "  <- planted bad segment" if seg == q["bad_segment"] else ""
        print(f"   {seg}: n={q['segment_sizes'][seg]} true rate={rate}{marker}")

    print(f"\n== 2. three designs, same labeling budget ({report['budget']}) ==")
    print(f"   {'design':<14} {'estimate':>9} {'se':>8}  {'95% CI':<19} "
          f"bad-segment flagged?")
    for name, d in report["designs"].items():
        seg = d["segments"][BAD_SEGMENT]
        print(f"   {name:<14} {d['estimate']:>9} {d.get('se', '')!s:>8}  "
              f"{d['ci95']}  {seg['flagged']}")
    rw = report["designs"]["risk_weighted"]
    srs = report["designs"]["srs"]
    assert rw["segments"][BAD_SEGMENT]["flagged"] is True
    assert srs["segments"][BAD_SEGMENT]["flagged"] is False
    assert abs(rw["estimate"] - q["true_overall_error_rate"]) <= 0.02
    assert rw["se"] < srs["se"]

    print("\n== 3. budget sweep (estimate at each budget) ==")
    for b, row in report["budget_sweep"].items():
        print(f"   budget {b}: srs={row['srs']} stratified={row['stratified']} "
              f"risk_weighted={row['risk_weighted']} "
              f"(se srs={row['srs_se']} rw={row['risk_weighted_se']})")

    study = report["multiseed_study"]
    print(f"\n== 4. multi-seed study ({study['good_risk']['reps']} fresh "
          "queues per scenario) ==")
    for label in ("good_risk", "miscalibrated_risk"):
        block = study[label]
        print(f"   {label}: mean truth={block['mean_truth']}")
        for design in ("srs", "risk_weighted_no_floor",
                       "risk_weighted_floored"):
            d = block[design]
            print(f"     {design:<24} mean est={d['mean_estimate']} "
                  f"empirical SE={d['empirical_se']} bias={d['bias_vs_mean_truth']}")
    good = study["good_risk"]
    miscal = study["miscalibrated_risk"]
    assert abs(good["risk_weighted_floored"]["bias_vs_mean_truth"]) <= 0.01
    assert miscal["risk_weighted_no_floor"]["empirical_se"] > \
        2 * miscal["srs"]["empirical_se"]
    assert miscal["risk_weighted_floored"]["empirical_se"] < \
        0.5 * miscal["risk_weighted_no_floor"]["empirical_se"]
    assert good["risk_weighted_floored"]["empirical_se"] < \
        1.5 * good["risk_weighted_no_floor"]["empirical_se"]

    print("\n== 5. CLI check gate ==")
    code = cli_main(["check"])
    print(f"   check exit={code}")
    assert code == EXIT_OK
    assert report["claims"]["all_headline_claims_hold"] is True
    print("\nDemo OK: flag contrast, HT accuracy, miscalibrated blow-up "
          "and floor cap verified.")
    return 0


if __name__ == "__main__":
    sys.exit(run())

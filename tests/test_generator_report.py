import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator import (BAD_SEGMENT, generate_queue, true_overall_rate,  # noqa: E402
                       true_segment_rates)
from report import build_report, multiseed_study  # noqa: E402
from sampling import estimate_srs, ht_estimate, risk_weighted_sample, srs_sample  # noqa: E402


def _corr(cases):
    n = len(cases)
    mx = sum(c["risk"] for c in cases) / n
    my = sum(c["is_error"] for c in cases) / n
    cov = sum((c["risk"] - mx) * (c["is_error"] - my) for c in cases)
    vx = sum((c["risk"] - mx) ** 2 for c in cases)
    vy = sum((c["is_error"] - my) ** 2 for c in cases)
    return cov / (vx * vy) ** 0.5


class TestGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = generate_queue(seed=42)

    def test_queue_shape_and_planted_segment(self):
        self.assertEqual(len(self.cases), 6000)
        rates = true_segment_rates(self.cases)
        self.assertTrue(0.35 <= rates[BAD_SEGMENT] <= 0.45)
        for seg, rate in rates.items():
            if seg != BAD_SEGMENT:
                self.assertTrue(0.01 <= rate <= 0.07, seg)

    def test_exactly_one_bad_segment(self):
        rates = true_segment_rates(self.cases)
        bad = [s for s, r in rates.items() if r > 0.15]
        self.assertEqual(bad, [BAD_SEGMENT])

    def test_deterministic(self):
        again = generate_queue(seed=42)
        self.assertEqual(self.cases, again)

    def test_risk_correlates_with_errors_in_main_scenario(self):
        self.assertGreater(_corr(self.cases), 0.3)

    def test_miscalibration_destroys_correlation_only(self):
        miscal = generate_queue(seed=42, miscalibrated=True)
        self.assertLess(abs(_corr(miscal)), 0.05)
        # same labels, same multiset of risk scores -- only pairing changed
        self.assertEqual([c["is_error"] for c in miscal],
                         [c["is_error"] for c in self.cases])
        self.assertEqual(sorted(c["risk"] for c in miscal),
                         sorted(c["risk"] for c in self.cases))
        self.assertEqual(true_overall_rate(miscal),
                         true_overall_rate(self.cases))


class TestMiscalibratedVarianceOrdering(unittest.TestCase):
    def test_no_floor_ht_se_exceeds_srs_when_miscalibrated(self):
        # Small multi-seed check (kept short; the full 24-rep study is
        # in report/evals): shuffled risk makes unfloored HT unstable.
        rw, srs = [], []
        for r in range(10):
            cases = generate_queue(seed=2000 + r, miscalibrated=True)
            sample, pis = risk_weighted_sample(cases, 300, 7000 + r,
                                               pi_floor=0.0)
            rw.append(ht_estimate(cases, sample, pis)[0])
            srs.append(estimate_srs(
                cases, srs_sample(cases, 300, 7000 + r))["estimate"])

        def sd(v):
            m = sum(v) / len(v)
            return (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5

        self.assertGreater(sd(rw), sd(srs))


class TestReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_report()

    def test_headline_flag_contrast(self):
        designs = self.report["designs"]
        self.assertTrue(
            designs["risk_weighted"]["segments"][BAD_SEGMENT]["flagged"])
        self.assertFalse(designs["srs"]["segments"][BAD_SEGMENT]["flagged"])

    def test_all_claims_hold(self):
        self.assertTrue(self.report["claims"]["all_headline_claims_hold"])

    def test_report_deterministic(self):
        self.assertEqual(build_report(), self.report)

    def test_study_shape(self):
        study = multiseed_study(reps=3)
        self.assertEqual(study["good_risk"]["reps"], 3)
        self.assertIn("risk_weighted_floored", study["miscalibrated_risk"])


if __name__ == "__main__":
    unittest.main()

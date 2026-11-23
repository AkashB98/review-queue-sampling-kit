import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator import generate_queue  # noqa: E402
from sampling import (ht_estimate, inclusion_probabilities,  # noqa: E402
                      risk_weighted_sample, srs_sample, stratified_sample,
                      wilson_ci)


def _tiny_cases():
    # ids 0..3, errors on 0 and 2
    return [{"id": i, "segment": "s", "is_error": y, "risk": 0.5}
            for i, y in enumerate((1, 0, 1, 0))]


class TestHorvitzThompson(unittest.TestCase):
    def test_hand_computable(self):
        # pi all 0.5, sample = cases 0 (error) and 1 (no error):
        # est = (1/4)(1/0.5 + 0/0.5) = 0.5; var = (1/16)[(0.5)(2^2) + 0]
        # = 0.125, se = sqrt(0.125).
        cases = _tiny_cases()
        est, se = ht_estimate(cases, [0, 1], [0.5, 0.5, 0.5, 0.5])
        self.assertAlmostEqual(est, 0.5)
        self.assertAlmostEqual(se, 0.125 ** 0.5)

    def test_full_sample_equal_pis_recovers_truth(self):
        cases = _tiny_cases()
        est, _se = ht_estimate(cases, [0, 1, 2, 3], [1.0, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(est, 0.5)

    def test_low_pi_error_counts_more(self):
        cases = _tiny_cases()
        hi, _ = ht_estimate(cases, [0], [0.5, 0.5, 0.5, 0.5])
        lo, _ = ht_estimate(cases, [0], [0.25, 0.5, 0.5, 0.5])
        self.assertGreater(lo, hi)


class TestFloorMechanism(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = generate_queue(seed=42)

    def test_min_pi_at_least_floor(self):
        for floor in (0.005, 0.01, 0.02):
            pis = inclusion_probabilities(self.cases, 300, pi_floor=floor)
            self.assertGreaterEqual(min(pis), floor - 1e-12)

    def test_pis_sum_to_budget(self):
        pis = inclusion_probabilities(self.cases, 300, pi_floor=0.01)
        self.assertAlmostEqual(sum(pis), 300, places=6)

    def test_floor_raises_min_pi(self):
        plain = inclusion_probabilities(self.cases, 300, pi_floor=0.0)
        floored = inclusion_probabilities(self.cases, 300, pi_floor=0.01)
        self.assertGreater(min(floored), min(plain))
        self.assertLess(min(plain), 0.001)

    def test_floor_too_large_rejected(self):
        with self.assertRaises(ValueError):
            inclusion_probabilities(self.cases, 300, pi_floor=0.06)


class TestSamplersRespectBudget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = generate_queue(seed=42)

    def test_exact_budget_and_unique(self):
        for budget in (150, 300, 600):
            s = srs_sample(self.cases, budget, seed=7)
            self.assertEqual(len(s), budget)
            self.assertEqual(len(set(s)), budget)
            t = stratified_sample(self.cases, budget, seed=7)
            self.assertEqual(len(t), budget)
            self.assertEqual(len(set(t)), budget)
            w, _pis = risk_weighted_sample(self.cases, budget, seed=7)
            self.assertEqual(len(w), budget)
            self.assertEqual(len(set(w)), budget)

    def test_stratified_is_proportional(self):
        sample = stratified_sample(self.cases, 300, seed=7)
        by_id = {c["id"]: c for c in self.cases}
        counts = {}
        for i in sample:
            counts[by_id[i]["segment"]] = counts.get(by_id[i]["segment"], 0) + 1
        # 700/6000 of 300 = 35 for the bad segment, 100 for standard_a
        self.assertEqual(counts["new_cohort"], 35)
        self.assertEqual(counts["standard_a"], 100)

    def test_determinism_same_seed_same_sample(self):
        a = srs_sample(self.cases, 300, seed=99)
        b = srs_sample(self.cases, 300, seed=99)
        self.assertEqual(a, b)
        wa, pa = risk_weighted_sample(self.cases, 300, seed=99)
        wb, pb = risk_weighted_sample(self.cases, 300, seed=99)
        self.assertEqual(wa, wb)
        self.assertEqual(pa, pb)

    def test_budget_validation(self):
        with self.assertRaises(ValueError):
            srs_sample(self.cases, 0, seed=1)
        with self.assertRaises(ValueError):
            srs_sample(self.cases, 6001, seed=1)


class TestWilsonCI(unittest.TestCase):
    def test_contains_point_estimate(self):
        lo, hi = wilson_ci(40, 100)
        self.assertLessEqual(lo, 0.4)
        self.assertGreaterEqual(hi, 0.4)

    def test_edges(self):
        self.assertEqual(wilson_ci(0, 50)[0], 0.0)
        self.assertEqual(wilson_ci(50, 50)[1], 1.0)

    def test_wider_when_smaller_n(self):
        w_small = wilson_ci(4, 10)
        w_large = wilson_ci(40, 100)
        self.assertGreater(w_small[1] - w_small[0], w_large[1] - w_large[0])


if __name__ == "__main__":
    unittest.main()

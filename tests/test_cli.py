import io
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli import main as cli_main  # noqa: E402


def _run(argv):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return cli_main(argv)


class TestCLI(unittest.TestCase):
    def test_check_passes_on_seeded_data(self):
        self.assertEqual(_run(["check"]), 0)

    def test_check_bad_budget_is_error(self):
        self.assertEqual(_run(["check", "--budget", "0"]), 2)

    def test_check_missing_input_is_error(self):
        self.assertEqual(
            _run(["check", "--input", "/tmp/no-such-rqsk-xyz.json"]), 2)

    def test_generate_and_estimate_run(self):
        self.assertEqual(_run(["generate"]), 0)
        self.assertEqual(_run(["estimate", "--budget", "150"]), 0)
        self.assertEqual(_run(["sample", "--design", "risk_weighted"]), 0)

    def test_estimate_bad_budget_is_error(self):
        self.assertEqual(_run(["estimate", "--budget", "99999"]), 2)


if __name__ == "__main__":
    unittest.main()

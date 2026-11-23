import os
import socket
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestNoNetwork(unittest.TestCase):
    def test_report_builds_with_sockets_blocked(self):
        real = socket.socket

        def blocked(*a, **k):
            raise AssertionError("network use is forbidden")

        socket.socket = blocked
        try:
            from report import build_report
            r = build_report()
            self.assertIn("designs", r)
        finally:
            socket.socket = real


if __name__ == "__main__":
    unittest.main()

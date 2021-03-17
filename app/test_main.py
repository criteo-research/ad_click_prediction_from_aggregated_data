import pytest
import unittest

from app import main


class TestGreet(unittest.TestCase):
    """Test the app."""

    def test_greet(self):
        """Test the greet function."""
        self.assertEqual(main.greet(), "Hello Anonymous")
        self.assertEqual(main.greet("C3PO"), "Hello C3PO")


@pytest.mark.benchmark()
def test_greet(benchmark):
    """Benchmark the function."""
    benchmark.pedantic(
        main.greet, args=(b'[a B] foo',),
        iterations=10000, rounds=100)


if __name__ == '__main__':
    unittest.main()

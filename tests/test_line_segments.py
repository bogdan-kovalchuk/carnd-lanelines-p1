"""Characterization tests for lane line-segment classification."""

import math
import unittest

from src.line_segments import split_line_segments


class SplitLineSegmentsTests(unittest.TestCase):
    def test_classifies_notebook_style_lane_candidates(self):
        segments = [
            (0, 0, 10, 10),
            (0, 10, 10, 0),
            (0, 5, 10, 5),
            (5, 0, 5, 10),
            (0, 0, 10, 1),
        ]

        right, left = split_line_segments(segments)

        self.assertEqual(right, ((0, 0, 10, 10),))
        self.assertEqual(left, ((0, 10, 10, 0),))

    def test_preserves_candidate_order_and_coordinates(self):
        segments = [(1.5, 2, 4.5, 5), (10, 9, 7, 6)]

        right, left = split_line_segments(segments)

        self.assertEqual(right, tuple(segments))
        self.assertEqual(left, ())

    def test_empty_input_produces_empty_groups(self):
        self.assertEqual(split_line_segments([]), ((), ()))

    def test_strictly_excludes_threshold_angles(self):
        lower_dy = math.tan(math.radians(20.0)) * 100.0
        upper_dy = math.tan(math.radians(80.0)) * 100.0

        right, left = split_line_segments(
            [(0, 0, 100, lower_dy), (0, 0, 100, upper_dy)]
        )

        self.assertEqual(right, ())
        self.assertEqual(left, ())


if __name__ == "__main__":
    unittest.main()

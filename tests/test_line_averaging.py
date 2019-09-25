"""Characterization tests for lane line averaging."""

import unittest

from src.line_averaging import average_extrapolated_segments


class AverageExtrapolatedSegmentsTests(unittest.TestCase):
    def test_extrapolates_a_single_segment_to_lane_boundaries(self):
        line = average_extrapolated_segments([(2, 2, 4, 4)], 6, 0)

        self.assertEqual(line, (6, 6, 0, 0))

    def test_averages_individually_extrapolated_segments(self):
        segments = [(0, 0, 10, 10), (4, 0, 14, 10)]

        line = average_extrapolated_segments(segments, 10, 0)

        self.assertEqual(line, (12, 10, 2, 0))

    def test_empty_input_has_no_average_line(self):
        self.assertIsNone(average_extrapolated_segments([], 10, 0))


if __name__ == "__main__":
    unittest.main()

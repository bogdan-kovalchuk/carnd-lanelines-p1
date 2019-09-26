"""Characterization tests for lane line averaging."""

import unittest

from src.line_averaging import average_extrapolated_segments, fit_line_to_endpoints


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

    def test_endpoint_order_does_not_change_the_result(self):
        forward = average_extrapolated_segments([(2, 2, 4, 4)], 6, 0)
        reversed_line = average_extrapolated_segments([(4, 4, 2, 2)], 6, 0)

        self.assertEqual(reversed_line, forward)

    def test_ignores_horizontal_segments(self):
        segments = [(0, 3, 5, 3), (2, 2, 4, 4)]

        self.assertEqual(
            average_extrapolated_segments(segments, 6, 0),
            (6, 6, 0, 0),
        )
        self.assertIsNone(average_extrapolated_segments(segments[:1], 6, 0))

    def test_truncates_averaged_coordinates_like_the_notebook(self):
        segments = [(0, 0, 5, 10), (2, 0, 7, 10)]

        line = average_extrapolated_segments(segments, 5, 0)

        self.assertEqual(line, (3, 5, 1, 0))

    def test_does_not_mutate_input_segments(self):
        segments = [[0, 0, 10, 10], [4, 0, 14, 10]]
        original = [segment[:] for segment in segments]

        average_extrapolated_segments(segments, 10, 0)

        self.assertEqual(segments, original)


class FitLineToEndpointsTests(unittest.TestCase):
    def test_matches_segment_average_for_parallel_equal_span_lines(self):
        segments = [(0, 0, 10, 10), (2, 0, 12, 10)]

        self.assertEqual(
            fit_line_to_endpoints(segments, 10, 0),
            average_extrapolated_segments(segments, 10, 0),
        )

    def test_weights_segments_by_their_vertical_span(self):
        segments = [(0, 0, 10, 10), (100, 4, 104, 6)]

        fitted = fit_line_to_endpoints(segments, 10, 0)
        averaged = average_extrapolated_segments(segments, 10, 0)

        self.assertEqual(fitted, (58, 10, 48, 0))
        self.assertEqual(averaged, (61, 10, 46, 0))

    def test_returns_none_without_vertical_variation(self):
        self.assertIsNone(fit_line_to_endpoints([], 10, 0))
        self.assertIsNone(fit_line_to_endpoints([(0, 3, 5, 3)], 10, 0))


if __name__ == "__main__":
    unittest.main()

"""Characterization tests for lane line-segment classification."""

import math
import unittest

from src.line_segments import split_line_segments, split_line_segments_by_ratio


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


class RatioSplitLineSegmentsTests(unittest.TestCase):
    def test_matches_angle_implementation_on_mixed_segments(self):
        segments = [
            (0, 0, 10, 10),
            (0, 10, 10, 0),
            (10, 10, 0, 0),
            (10, 0, 0, 10),
            (0, 0, 10, 2),
            (4, 0, 4, 12),
        ]

        self.assertEqual(
            split_line_segments_by_ratio(segments),
            split_line_segments(segments),
        )

    def test_uses_vector_sign_independently_of_endpoint_order(self):
        forward = [(0, 0, 4, 3), (0, 3, 4, 0)]
        reversed_segments = [(4, 3, 0, 0), (4, 0, 0, 3)]

        self.assertEqual(
            split_line_segments_by_ratio(forward),
            (((0, 0, 4, 3),), ((0, 3, 4, 0),)),
        )
        self.assertEqual(
            split_line_segments_by_ratio(reversed_segments),
            (((4, 3, 0, 0),), ((4, 0, 0, 3),)),
        )


class LineSegmentValidationTests(unittest.TestCase):
    def test_rejects_invalid_segment_shapes_and_values(self):
        invalid_inputs = [
            [(1, 2, 3)],
            [(1, 2, 3, 4, 5)],
            [(1, 2, "3", 4)],
            [(1, 2, math.inf, 4)],
            [(1, 2, math.nan, 4)],
            [None],
        ]

        for segments in invalid_inputs:
            with self.subTest(segments=segments):
                with self.assertRaises(ValueError):
                    split_line_segments(segments)
                with self.assertRaises(ValueError):
                    split_line_segments_by_ratio(segments)

    def test_rejects_invalid_angle_ranges(self):
        invalid_ranges = [(-1, 80), (20, 20), (80, 20), (20, 91), (math.nan, 80)]

        for bounds in invalid_ranges:
            with self.subTest(bounds=bounds):
                with self.assertRaises(ValueError):
                    split_line_segments([], *bounds)
                with self.assertRaises(ValueError):
                    split_line_segments_by_ratio([], *bounds)

    def test_ignores_zero_length_and_vertical_segments(self):
        segments = [(2, 2, 2, 2), (5, -10, 5, 10)]

        self.assertEqual(split_line_segments(segments), ((), ()))
        self.assertEqual(split_line_segments_by_ratio(segments), ((), ()))

    def test_accepts_single_pass_iterables(self):
        segments = ((0, 0, 10, y) for y in (10, -10))

        self.assertEqual(
            split_line_segments_by_ratio(segments),
            (((0.0, 0.0, 10.0, 10.0),), ((0.0, 0.0, 10.0, -10.0),)),
        )


if __name__ == "__main__":
    unittest.main()

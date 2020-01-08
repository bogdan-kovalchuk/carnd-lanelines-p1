"""Synthetic integration fixtures for the extracted lane pipeline helpers."""

import unittest

from src.line_averaging import average_extrapolated_segments, fit_line_to_endpoints
from src.line_segments import split_line_segments, split_line_segments_by_ratio
from src.pipeline_config import PipelineConfig
from src.region_mask import mask_polygon, mask_polygon_scanline


class SyntheticLanePipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = PipelineConfig()
        self.lane_segments = [
            (510, 330, 800, 540),
            (450, 330, 160, 540),
            (100, 400, 300, 400),
            (480, 300, 480, 500),
        ]

    def test_extracts_and_extrapolates_expected_lane_boundaries(self):
        right, left = split_line_segments(
            self.lane_segments,
            self.config.min_segment_angle,
            self.config.max_segment_angle,
        )
        y_bottom, y_top = self.config.y_extents()

        self.assertEqual(
            average_extrapolated_segments(right, y_bottom, y_top),
            (800, 540.0, 510, 330.0),
        )
        self.assertEqual(
            average_extrapolated_segments(left, y_bottom, y_top),
            (160, 540.0, 450, 330.0),
        )

    def test_alternative_algorithms_agree_on_ideal_lane_fixtures(self):
        angle_groups = split_line_segments(
            self.lane_segments,
            self.config.min_segment_angle,
            self.config.max_segment_angle,
        )
        ratio_groups = split_line_segments_by_ratio(
            self.lane_segments,
            self.config.min_segment_angle,
            self.config.max_segment_angle,
        )
        self.assertEqual(ratio_groups, angle_groups)

        y_bottom, y_top = self.config.y_extents()
        for segments in angle_groups:
            with self.subTest(segments=segments):
                fitted = fit_line_to_endpoints(segments, y_bottom, y_top)
                averaged = average_extrapolated_segments(segments, y_bottom, y_top)
                self.assertEqual((fitted[1], fitted[3]), (averaged[1], averaged[3]))
                self.assertLessEqual(abs(fitted[0] - averaged[0]), 1)
                self.assertLessEqual(abs(fitted[2] - averaged[2]), 1)

    def test_non_lane_segments_produce_no_boundaries(self):
        non_lane_segments = [(0, 10, 20, 10), (5, 0, 5, 20), (1, 1, 1, 1)]

        right, left = split_line_segments(non_lane_segments)

        self.assertEqual((right, left), ((), ()))
        self.assertIsNone(average_extrapolated_segments(right, 20, 10))
        self.assertIsNone(average_extrapolated_segments(left, 20, 10))

    def test_configured_region_masks_a_synthetic_edge_map(self):
        config = PipelineConfig(
            image_width=8,
            image_height=6,
            top_shift=3,
            bottom_shift=0,
            horizontal_shift=1,
        )
        edge_map = [[1] * config.image_width for _ in range(config.image_height)]

        point_mask = mask_polygon(edge_map, config.roi_vertices())
        scanline_mask = mask_polygon_scanline(edge_map, config.roi_vertices())

        self.assertEqual(scanline_mask, point_mask)
        self.assertEqual(point_mask[:3], ((0,) * 8,) * 3)
        self.assertEqual(tuple(sum(row) for row in point_mask[3:]), (3, 5, 7))


if __name__ == "__main__":
    unittest.main()

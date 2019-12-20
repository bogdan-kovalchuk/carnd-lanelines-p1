"""Characterization tests for lane pipeline configuration."""

from dataclasses import FrozenInstanceError
import math
import unittest

from src.pipeline_config import PipelineConfig


class PipelineConfigTests(unittest.TestCase):
    def test_defaults_match_the_notebook_constants(self):
        config = PipelineConfig()

        self.assertEqual(config.image_width, 960)
        self.assertEqual(config.image_height, 540)
        self.assertEqual(config.kernel_size, 5)
        self.assertEqual((config.canny_low_threshold, config.canny_high_threshold), (50, 150))
        self.assertEqual((config.top_shift, config.bottom_shift), (210, 0))
        self.assertEqual(config.horizontal_shift, 30)
        self.assertEqual(config.hough_rho, 2.0)
        self.assertAlmostEqual(config.hough_theta, math.pi / 180.0)
        self.assertEqual((config.hough_threshold, config.min_line_length), (15, 40))
        self.assertEqual(config.max_line_gap, 20)

    def test_builds_notebook_region_vertices_and_extents(self):
        config = PipelineConfig()

        self.assertEqual(
            config.roi_vertices(),
            ((0.0, 540.0), (450.0, 330.0), (510.0, 330.0), (960.0, 540.0)),
        )
        self.assertEqual(config.y_extents(), (540.0, 330.0))

    def test_custom_image_geometry_updates_region_vertices(self):
        config = PipelineConfig(
            image_width=100,
            image_height=60,
            top_shift=20,
            bottom_shift=5,
            horizontal_shift=10,
        )

        self.assertEqual(
            config.roi_vertices(),
            ((0.0, 55.0), (40.0, 40.0), (60.0, 40.0), (100.0, 55.0)),
        )

    def test_configuration_is_immutable(self):
        config = PipelineConfig()

        with self.assertRaises(FrozenInstanceError):
            config.kernel_size = 7


if __name__ == "__main__":
    unittest.main()

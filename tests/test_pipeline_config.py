"""Characterization tests for lane pipeline configuration."""

from dataclasses import FrozenInstanceError
import math
import unittest

from src.pipeline_config import (
    PipelineConfig,
    parse_pipeline_config,
    pipeline_config_from_environment,
)


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


class MappingPipelineConfigTests(unittest.TestCase):
    def test_empty_mapping_matches_default_constructor(self):
        self.assertEqual(PipelineConfig.from_mapping({}), PipelineConfig())

    def test_coerces_numeric_strings_by_configuration_schema(self):
        config = PipelineConfig.from_mapping(
            {
                "image_width": "100",
                "image_height": "60",
                "top_shift": "20",
                "bottom_shift": "5",
                "horizontal_shift": "10",
                "hough_rho": "3.5",
            }
        )

        self.assertEqual(config.image_width, 100)
        self.assertEqual(config.hough_rho, 3.5)
        self.assertEqual(
            config.roi_vertices(),
            ((0.0, 55.0), (40.0, 40.0), (60.0, 40.0), (100.0, 55.0)),
        )

    def test_matches_direct_construction_for_typed_values(self):
        values = {"kernel_size": 7, "canny_low_threshold": 40}

        self.assertEqual(
            PipelineConfig.from_mapping(values),
            PipelineConfig(**values),
        )


class TextPipelineConfigTests(unittest.TestCase):
    def test_parses_key_value_lines_and_comments(self):
        text = """
        # Compact synthetic frame
        image_width = 100
        image_height=60
        top_shift = 20
        bottom_shift=5
        horizontal_shift=10
        hough_rho=3.5
        """

        config = parse_pipeline_config(text)

        self.assertEqual(
            config,
            PipelineConfig.from_mapping(
                {
                    "image_width": 100,
                    "image_height": 60,
                    "top_shift": 20,
                    "bottom_shift": 5,
                    "horizontal_shift": 10,
                    "hough_rho": 3.5,
                }
            ),
        )

    def test_empty_text_uses_all_defaults(self):
        self.assertEqual(parse_pipeline_config("\n# defaults\n"), PipelineConfig())


class EnvironmentPipelineConfigTests(unittest.TestCase):
    def test_reads_prefixed_variables_and_ignores_unrelated_entries(self):
        environment = {
            "LANE_IMAGE_WIDTH": "100",
            "LANE_IMAGE_HEIGHT": "60",
            "LANE_TOP_SHIFT": "20",
            "LANE_BOTTOM_SHIFT": "5",
            "LANE_HORIZONTAL_SHIFT": "10",
            "PATH": "ignored",
        }

        config = pipeline_config_from_environment(environment)

        self.assertEqual(
            config,
            PipelineConfig.from_mapping(
                {
                    "image_width": 100,
                    "image_height": 60,
                    "top_shift": 20,
                    "bottom_shift": 5,
                    "horizontal_shift": 10,
                }
            ),
        )

    def test_supports_a_custom_prefix(self):
        environment = {"PIPE_KERNEL_SIZE": "7", "LANE_KERNEL_SIZE": "9"}

        self.assertEqual(
            pipeline_config_from_environment(environment, prefix="PIPE_"),
            PipelineConfig(kernel_size=7),
        )


class PipelineConfigValidationTests(unittest.TestCase):
    def test_rejects_invalid_integer_and_threshold_settings(self):
        invalid_overrides = [
            {"image_width": 0},
            {"image_height": -1},
            {"kernel_size": 0},
            {"kernel_size": 4},
            {"kernel_size": True},
            {"canny_low_threshold": -1},
            {"canny_low_threshold": 150},
            {"canny_high_threshold": 256},
            {"hough_threshold": 0},
            {"min_line_length": -1},
            {"max_line_gap": -1},
        ]

        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    PipelineConfig(**overrides)

    def test_rejects_invalid_region_geometry(self):
        invalid_overrides = [
            {"bottom_shift": -1},
            {"bottom_shift": 210},
            {"top_shift": 0},
            {"top_shift": 541},
            {"horizontal_shift": -1},
            {"horizontal_shift": 481},
        ]

        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    PipelineConfig(**overrides)

    def test_rejects_nonfinite_and_invalid_float_settings(self):
        invalid_overrides = [
            {"hough_rho": 0},
            {"hough_theta": math.inf},
            {"min_segment_angle": math.nan},
            {"min_segment_angle": 80, "max_segment_angle": 20},
            {"max_segment_angle": 91},
            {"hough_rho": True},
        ]

        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    PipelineConfig(**overrides)


if __name__ == "__main__":
    unittest.main()

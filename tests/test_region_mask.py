"""Characterization tests for polygonal region masking."""

import math
import unittest

from src.region_mask import mask_polygon, mask_polygon_scanline


class MaskPolygonTests(unittest.TestCase):
    def test_keeps_grayscale_pixels_inside_a_rectangular_region(self):
        image = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ]
        vertices = [(1, 0), (2, 0), (2, 2), (1, 2)]

        masked = mask_polygon(image, vertices)

        self.assertEqual(
            masked,
            ((0, 2, 3, 0), (0, 6, 7, 0), (0, 10, 11, 0)),
        )

    def test_preserves_multichannel_pixels_and_channel_count(self):
        image = [[(1, 2, 3), (4, 5, 6)], [(7, 8, 9), (10, 11, 12)]]
        vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]

        self.assertEqual(mask_polygon(image, vertices), tuple(map(tuple, image)))

    def test_blacks_out_pixels_when_polygon_is_outside_image(self):
        image = [[(1, 2, 3), (4, 5, 6)]]
        vertices = [(10, 10), (12, 10), (12, 12), (10, 12)]

        self.assertEqual(mask_polygon(image, vertices), (((0, 0, 0), (0, 0, 0)),))

    def test_includes_pixels_on_polygon_edges(self):
        image = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        triangle = [(0, 2), (1, 0), (2, 2)]

        self.assertEqual(
            mask_polygon(image, triangle),
            ((0, 1, 0), (0, 1, 0), (1, 1, 1)),
        )

    def test_does_not_mutate_the_source_image(self):
        image = [[1, 2], [3, 4]]
        original = [row[:] for row in image]

        mask_polygon(image, [(0, 0), (1, 0), (0, 1)])

        self.assertEqual(image, original)


class ScanlineMaskPolygonTests(unittest.TestCase):
    def test_matches_point_testing_for_rectangular_and_triangular_regions(self):
        image = [[1] * 6 for _ in range(5)]
        polygons = [
            [(1, 0), (4, 0), (4, 4), (1, 4)],
            [(0, 4), (3, 0), (5, 4)],
        ]

        for polygon in polygons:
            with self.subTest(polygon=polygon):
                self.assertEqual(
                    mask_polygon_scanline(image, polygon),
                    mask_polygon(image, polygon),
                )

    def test_matches_point_testing_for_a_concave_region(self):
        image = [[1] * 7 for _ in range(6)]
        concave_polygon = [(0, 0), (6, 0), (6, 5), (3, 2), (0, 5)]

        self.assertEqual(
            mask_polygon_scanline(image, concave_polygon),
            mask_polygon(image, concave_polygon),
        )

    def test_preserves_color_pixels_and_blacks_the_exterior(self):
        image = [[(x, y, 1) for x in range(4)] for y in range(3)]
        polygon = [(1, 0), (2, 0), (2, 2), (1, 2)]

        self.assertEqual(
            mask_polygon_scanline(image, polygon),
            mask_polygon(image, polygon),
        )


class RegionMaskValidationTests(unittest.TestCase):
    FUNCTIONS = (mask_polygon, mask_polygon_scanline)
    VALID_IMAGE = [[1, 2], [3, 4]]
    VALID_POLYGON = [(0, 0), (1, 0), (0, 1)]

    def test_rejects_empty_ragged_and_non_matrix_images(self):
        invalid_images = [None, [], [[]], [[1], [2, 3]], [1, 2], "image"]

        for function in self.FUNCTIONS:
            for image in invalid_images:
                with self.subTest(function=function.__name__, image=image):
                    with self.assertRaises(ValueError):
                        function(image, self.VALID_POLYGON)

    def test_rejects_malformed_polygons(self):
        invalid_polygons = [
            None,
            [],
            [(0, 0), (1, 1)],
            [(0, 0), (1, 0), (0,)],
            [(0, 0), (1, 0), (0, "1")],
            [(0, 0), (1, 0), (0, math.nan)],
            [(0, 0), (1, 0), (0, math.inf)],
        ]

        for function in self.FUNCTIONS:
            for polygon in invalid_polygons:
                with self.subTest(function=function.__name__, polygon=polygon):
                    with self.assertRaises(ValueError):
                        function(self.VALID_IMAGE, polygon)

    def test_accepts_single_pass_image_rows(self):
        for function in self.FUNCTIONS:
            image = (row for row in self.VALID_IMAGE)
            with self.subTest(function=function.__name__):
                self.assertEqual(
                    function(image, self.VALID_POLYGON),
                    ((1, 2), (3, 0)),
                )


if __name__ == "__main__":
    unittest.main()

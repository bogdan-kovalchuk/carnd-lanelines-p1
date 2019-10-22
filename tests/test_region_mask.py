"""Characterization tests for polygonal region masking."""

import unittest

from src.region_mask import mask_polygon


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


if __name__ == "__main__":
    unittest.main()

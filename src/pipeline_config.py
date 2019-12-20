"""Typed configuration values for the lane-detection notebook pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple


Point = Tuple[float, float]


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable parameters corresponding to the notebook constants."""

    image_width: int = 960
    image_height: int = 540
    kernel_size: int = 5
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150
    top_shift: int = 210
    bottom_shift: int = 0
    horizontal_shift: int = 30
    hough_rho: float = 2.0
    hough_theta: float = math.pi / 180.0
    hough_threshold: int = 15
    min_line_length: int = 40
    max_line_gap: int = 20
    min_segment_angle: float = 20.0
    max_segment_angle: float = 80.0

    def roi_vertices(self) -> Tuple[Point, Point, Point, Point]:
        """Return the notebook's trapezoidal region of interest."""

        bottom_y = float(self.image_height - self.bottom_shift)
        top_y = float(self.image_height - self.top_shift)
        center_x = self.image_width / 2.0
        return (
            (0.0, bottom_y),
            (center_x - self.horizontal_shift, top_y),
            (center_x + self.horizontal_shift, top_y),
            (float(self.image_width), bottom_y),
        )

    def y_extents(self) -> Tuple[float, float]:
        """Return bottom and top y coordinates used for line extrapolation."""

        vertices = self.roi_vertices()
        return vertices[0][1], vertices[1][1]

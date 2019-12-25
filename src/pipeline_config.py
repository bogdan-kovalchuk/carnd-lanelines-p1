"""Typed configuration values for the lane-detection notebook pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Tuple


Point = Tuple[float, float]

_INTEGER_FIELDS = {
    "image_width",
    "image_height",
    "kernel_size",
    "canny_low_threshold",
    "canny_high_threshold",
    "top_shift",
    "bottom_shift",
    "horizontal_shift",
    "hough_threshold",
    "min_line_length",
    "max_line_gap",
}
_FLOAT_FIELDS = {
    "hough_rho",
    "hough_theta",
    "min_segment_angle",
    "max_segment_angle",
}


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

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "PipelineConfig":
        """Build a configuration from typed values or numeric strings."""

        converted = {}
        for key, value in values.items():
            if key in _INTEGER_FIELDS:
                converted[key] = int(value)
            elif key in _FLOAT_FIELDS:
                converted[key] = float(value)
            else:
                converted[key] = value
        return cls(**converted)

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


def parse_pipeline_config(text: str) -> PipelineConfig:
    """Parse newline-delimited ``name=value`` configuration text."""

    values = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator:
            raise ValueError("configuration lines must use name=value syntax")
        values[key.strip()] = value.strip()
    return PipelineConfig.from_mapping(values)

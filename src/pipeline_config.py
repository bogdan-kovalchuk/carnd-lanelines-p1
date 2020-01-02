"""Typed configuration values for the lane-detection notebook pipeline.

Direct ``PipelineConfig`` construction is constant time because the schema has
a fixed number of fields. ``from_mapping`` is O(k) for k overrides and is the
best adapter for already structured data. ``parse_pipeline_config`` is O(c)
for c input characters and suits small human-edited files. Environment loading
is O(e) for e variables and keeps deployment overrides outside source files.
All approaches use O(k) temporary space and converge on the same validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
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
_CONFIG_FIELDS = _INTEGER_FIELDS | _FLOAT_FIELDS


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable parameters corresponding to the notebook constants.

    Construction and validation are O(1) because the field set is fixed.
    Prefer direct construction for typed values known at development time.
    """

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

    def __post_init__(self) -> None:
        integer_fields = {
            name: getattr(self, name)
            for name in _INTEGER_FIELDS
        }
        if any(type(value) is not int for value in integer_fields.values()):
            raise ValueError("integer configuration fields must contain integers")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if not 0 <= self.canny_low_threshold < self.canny_high_threshold <= 255:
            raise ValueError("Canny thresholds must satisfy 0 <= low < high <= 255")
        if not 0 <= self.bottom_shift < self.top_shift <= self.image_height:
            raise ValueError("ROI shifts must satisfy 0 <= bottom < top <= image height")
        if not 0 <= self.horizontal_shift <= self.image_width / 2:
            raise ValueError("horizontal_shift must fit within half the image width")
        if self.hough_threshold <= 0:
            raise ValueError("hough_threshold must be positive")
        if self.min_line_length < 0 or self.max_line_gap < 0:
            raise ValueError("line length and gap settings cannot be negative")

        real_fields = {
            "hough_rho": self.hough_rho,
            "hough_theta": self.hough_theta,
            "min_segment_angle": self.min_segment_angle,
            "max_segment_angle": self.max_segment_angle,
        }
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            for value in real_fields.values()
        ):
            raise ValueError("floating-point configuration fields must be finite numbers")
        if self.hough_rho <= 0 or self.hough_theta <= 0:
            raise ValueError("Hough resolutions must be positive")
        if not 0 <= self.min_segment_angle < self.max_segment_angle <= 90:
            raise ValueError("segment angles must satisfy 0 <= min < max <= 90")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "PipelineConfig":
        """Build a configuration from typed values or numeric strings.

        Conversion takes O(k) time and O(k) temporary space for k overrides.
        """

        if not isinstance(values, Mapping):
            raise ValueError("configuration values must be a mapping")
        converted = {}
        for key, value in values.items():
            if not isinstance(key, str) or key not in _CONFIG_FIELDS:
                raise ValueError(f"unknown configuration field: {key!r}")
            if isinstance(value, bool):
                raise ValueError(f"configuration field {key!r} cannot be boolean")
            if key in _INTEGER_FIELDS:
                if (
                    isinstance(value, Real)
                    and not isinstance(value, int)
                    and (
                        not math.isfinite(float(value))
                        or not float(value).is_integer()
                    )
                ):
                    raise ValueError(f"configuration field {key!r} must be an integer")
                converter = int
            else:
                converter = float
            try:
                converted[key] = converter(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"invalid value for configuration field {key!r}") from error
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
    """Parse newline-delimited ``name=value`` configuration text.

    Parsing takes O(c) time for c characters and O(k) override space.
    """

    if not isinstance(text, str):
        raise ValueError("configuration text must be a string")
    values = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.count("=") != 1:
            raise ValueError(f"line {line_number} must use name=value syntax")
        key, value = (part.strip() for part in line.split("=", 1))
        if not key or not value:
            raise ValueError(f"line {line_number} requires a name and value")
        if key in values:
            raise ValueError(f"duplicate configuration field on line {line_number}: {key}")
        values[key] = value
    return PipelineConfig.from_mapping(values)


def pipeline_config_from_environment(
    environment: Mapping[str, str],
    prefix: str = "LANE_",
) -> PipelineConfig:
    """Build a configuration from prefixed environment-style variables.

    Scanning takes O(e) time for e environment entries and O(k) override space.
    """

    if not isinstance(environment, Mapping):
        raise ValueError("environment must be a mapping")
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("environment prefix must be a non-empty string")

    values = {}
    for key, value in environment.items():
        if not isinstance(key, str):
            raise ValueError("environment variable names must be strings")
        if not key.startswith(prefix):
            continue
        field_name = key[len(prefix):].lower()
        if field_name in values:
            raise ValueError(f"duplicate environment field: {field_name}")
        values[field_name] = value
    return PipelineConfig.from_mapping(values)

"""Average and extrapolate lane line segments.

``average_extrapolated_segments`` preserves the notebook algorithm: it gives
every segment equal influence after extrapolation. ``fit_line_to_endpoints``
uses ordinary least squares, so long vertical spans contribute more leverage.
Both approaches run in O(n) time and use O(n) auxiliary space for n segments.
The segment average is easier to reason about; least squares usually makes
better use of segments with unequal lengths but is more sensitive to outliers.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Iterable, Optional, Sequence, Tuple


LineSegment = Tuple[float, float, float, float]


def _validate_extents(y_bottom: float, y_top: float) -> Tuple[float, float]:
    if (
        isinstance(y_bottom, bool)
        or isinstance(y_top, bool)
        or not isinstance(y_bottom, Real)
        or not isinstance(y_top, Real)
        or not math.isfinite(float(y_bottom))
        or not math.isfinite(float(y_top))
        or y_bottom <= y_top
    ):
        raise ValueError("y extents must be finite numbers with bottom greater than top")
    return float(y_bottom), float(y_top)


def _validated_segments(
    segments: Iterable[Sequence[float]],
) -> Iterable[LineSegment]:
    if segments is None:
        raise ValueError("segments must be an iterable")
    try:
        iterator = iter(segments)
    except TypeError as error:
        raise ValueError("segments must be an iterable") from error

    for raw_segment in iterator:
        try:
            coordinates = tuple(raw_segment)
        except TypeError as error:
            raise ValueError("each segment must be an iterable") from error
        if len(coordinates) != 4:
            raise ValueError("each segment must contain exactly four coordinates")
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            for value in coordinates
        ):
            raise ValueError("segment coordinates must be finite real numbers")
        yield (
            float(coordinates[0]),
            float(coordinates[1]),
            float(coordinates[2]),
            float(coordinates[3]),
        )


def _finite_mean(values: Sequence[float]) -> float:
    try:
        result = math.fsum(values) / len(values)
    except OverflowError as error:
        raise ValueError("coordinates are too large to average safely") from error
    if not math.isfinite(result):
        raise ValueError("coordinates are too large to average safely")
    return result


def _line_at_extents(
    bottom_x: float,
    y_bottom: float,
    top_x: float,
    y_top: float,
) -> LineSegment:
    if not math.isfinite(bottom_x) or not math.isfinite(top_x):
        raise ValueError("extrapolated coordinates must be finite")
    return (int(bottom_x), y_bottom, int(top_x), y_top)


def average_extrapolated_segments(
    segments: Iterable[Sequence[float]],
    y_bottom: float,
    y_top: float,
) -> Optional[LineSegment]:
    """Extrapolate each segment and average its x coordinates.

    This is the pure-Python equivalent of the notebook's
    ``extrapolate_and_average_lines`` helper. Horizontal segments cannot be
    expressed as ``x = f(y)`` and are ignored.

    Complexity is O(n) time and O(n) auxiliary space. Use this approach when
    every detected segment should have equal weight regardless of its length.
    """

    y_bottom, y_top = _validate_extents(y_bottom, y_top)
    bottom_x_values = []
    top_x_values = []
    for segment in _validated_segments(segments):
        x1, y1, x2, y2 = segment
        if y1 == y2:
            continue
        x_per_y = (x2 - x1) / (y2 - y1)
        bottom_x_values.append(x1 + (y_bottom - y1) * x_per_y)
        top_x_values.append(x1 + (y_top - y1) * x_per_y)

    if not bottom_x_values:
        return None

    return _line_at_extents(
        _finite_mean(bottom_x_values),
        y_bottom,
        _finite_mean(top_x_values),
        y_top,
    )


def fit_line_to_endpoints(
    segments: Iterable[Sequence[float]],
    y_bottom: float,
    y_top: float,
) -> Optional[LineSegment]:
    """Fit ``x = slope * y + intercept`` to all segment endpoints.

    Unlike segment-wise extrapolation, ordinary least squares gives more
    influence to endpoints that cover a larger vertical range.

    Complexity is O(n) time and O(n) auxiliary space. Use this approach when
    segment length is useful evidence and input outliers are already filtered.
    """

    y_bottom, y_top = _validate_extents(y_bottom, y_top)
    points = []
    for segment in _validated_segments(segments):
        x1, y1, x2, y2 = segment
        points.extend(((x1, y1), (x2, y2)))
    if not points:
        return None

    mean_x = _finite_mean([point[0] for point in points])
    mean_y = _finite_mean([point[1] for point in points])
    denominator = math.fsum((point[1] - mean_y) ** 2 for point in points)
    if denominator == 0:
        return None
    if not math.isfinite(denominator):
        raise ValueError("coordinates are too large to fit safely")

    numerator = math.fsum(
        (y - mean_y) * (x - mean_x)
        for x, y in points
    )
    if not math.isfinite(numerator):
        raise ValueError("coordinates are too large to fit safely")
    slope = numerator / denominator
    intercept = mean_x - slope * mean_y
    return _line_at_extents(
        slope * y_bottom + intercept,
        y_bottom,
        slope * y_top + intercept,
        y_top,
    )

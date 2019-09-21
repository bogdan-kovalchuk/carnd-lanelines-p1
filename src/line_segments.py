"""Classify Hough line segments into left and right lane candidates."""

from __future__ import annotations

import math
from numbers import Real
from typing import Iterable, Sequence, Tuple


LineSegment = Tuple[float, float, float, float]


def _validate_angle_bounds(min_angle: float, max_angle: float) -> None:
    if (
        isinstance(min_angle, bool)
        or isinstance(max_angle, bool)
        or not isinstance(min_angle, Real)
        or not isinstance(max_angle, Real)
        or not math.isfinite(float(min_angle))
        or not math.isfinite(float(max_angle))
        or not 0.0 <= min_angle < max_angle <= 90.0
    ):
        raise ValueError("angles must be finite and satisfy 0 <= min < max <= 90")


def _coerce_segment(raw_segment: Sequence[float]) -> LineSegment:
    try:
        coordinates = tuple(raw_segment)
    except TypeError as error:
        raise ValueError("each segment must be an iterable of four coordinates") from error
    if len(coordinates) != 4:
        raise ValueError("each segment must contain exactly four coordinates")
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        for value in coordinates
    ):
        raise ValueError("segment coordinates must be finite real numbers")
    return (
        float(coordinates[0]),
        float(coordinates[1]),
        float(coordinates[2]),
        float(coordinates[3]),
    )


def _validated_segments(
    segments: Iterable[Sequence[float]],
) -> Iterable[LineSegment]:
    if segments is None:
        raise ValueError("segments must be an iterable")
    try:
        iterator = iter(segments)
    except TypeError as error:
        raise ValueError("segments must be an iterable") from error
    return (_coerce_segment(segment) for segment in iterator)


def split_line_segments(
    segments: Iterable[Sequence[float]],
    min_angle: float = 20.0,
    max_angle: float = 80.0,
) -> Tuple[Tuple[LineSegment, ...], Tuple[LineSegment, ...]]:
    """Return ``(right, left)`` lane candidates in their input order.

    Image coordinates increase downwards, so positive slopes identify the
    right lane and negative slopes identify the left lane. Horizontal,
    vertical, and boundary-angle segments are ignored, matching the strict
    threshold comparisons in the notebook pipeline.
    """

    _validate_angle_bounds(min_angle, max_angle)
    right = []
    left = []
    for segment in _validated_segments(segments):
        x1, y1, x2, y2 = segment
        dx = x2 - x1
        dy = y2 - y1
        angle = 90.0 if dx == 0 else math.degrees(math.atan(dy / dx))

        if min_angle < angle < max_angle:
            right.append(segment)
        elif -max_angle < angle < -min_angle:
            left.append(segment)

    return tuple(right), tuple(left)


def split_line_segments_by_ratio(
    segments: Iterable[Sequence[float]],
    min_angle: float = 20.0,
    max_angle: float = 80.0,
) -> Tuple[Tuple[LineSegment, ...], Tuple[LineSegment, ...]]:
    """Classify segments using vector ratios instead of inverse trigonometry.

    The sign of ``dx * dy`` determines the lane side. Squared vector
    components are compared with squared tangent bounds, avoiding an angle
    calculation for each segment.
    """

    _validate_angle_bounds(min_angle, max_angle)
    minimum_ratio = math.tan(math.radians(min_angle))
    maximum_ratio = math.tan(math.radians(max_angle))
    right = []
    left = []

    for segment in _validated_segments(segments):
        x1, y1, x2, y2 = segment
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue

        ratio = abs(dy / dx)
        if not minimum_ratio < ratio < maximum_ratio:
            continue

        if dx * dy > 0:
            right.append(segment)
        elif dx * dy < 0:
            left.append(segment)

    return tuple(right), tuple(left)

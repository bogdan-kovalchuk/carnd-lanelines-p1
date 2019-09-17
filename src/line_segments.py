"""Classify Hough line segments into left and right lane candidates."""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple


LineSegment = Tuple[float, float, float, float]


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

    right = []
    left = []
    for raw_segment in segments:
        x1, y1, x2, y2 = raw_segment
        dx = x2 - x1
        dy = y2 - y1
        angle = 90.0 if dx == 0 else math.degrees(math.atan(dy / dx))
        segment = (x1, y1, x2, y2)

        if min_angle < angle < max_angle:
            right.append(segment)
        elif -max_angle < angle < -min_angle:
            left.append(segment)

    return tuple(right), tuple(left)

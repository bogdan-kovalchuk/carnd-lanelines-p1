"""Average and extrapolate lane line segments."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple


LineSegment = Tuple[float, float, float, float]


def average_extrapolated_segments(
    segments: Iterable[Sequence[float]],
    y_bottom: float,
    y_top: float,
) -> Optional[LineSegment]:
    """Extrapolate each segment and average its x coordinates.

    This is the pure-Python equivalent of the notebook's
    ``extrapolate_and_average_lines`` helper. Horizontal segments cannot be
    expressed as ``x = f(y)`` and are ignored.
    """

    bottom_x_values = []
    top_x_values = []
    for segment in segments:
        x1, y1, x2, y2 = segment
        if y1 == y2:
            continue
        x_per_y = (x2 - x1) / (y2 - y1)
        bottom_x_values.append(x1 + (y_bottom - y1) * x_per_y)
        top_x_values.append(x1 + (y_top - y1) * x_per_y)

    if not bottom_x_values:
        return None

    count = len(bottom_x_values)
    bottom_x = int(sum(bottom_x_values) / count)
    top_x = int(sum(top_x_values) / count)
    return (bottom_x, y_bottom, top_x, y_top)

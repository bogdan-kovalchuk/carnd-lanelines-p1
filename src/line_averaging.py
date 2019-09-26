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


def fit_line_to_endpoints(
    segments: Iterable[Sequence[float]],
    y_bottom: float,
    y_top: float,
) -> Optional[LineSegment]:
    """Fit ``x = slope * y + intercept`` to all segment endpoints.

    Unlike segment-wise extrapolation, ordinary least squares gives more
    influence to endpoints that cover a larger vertical range.
    """

    points = []
    for segment in segments:
        x1, y1, x2, y2 = segment
        points.extend(((x1, y1), (x2, y2)))
    if not points:
        return None

    mean_x = sum(point[0] for point in points) / len(points)
    mean_y = sum(point[1] for point in points) / len(points)
    denominator = sum((point[1] - mean_y) ** 2 for point in points)
    if denominator == 0:
        return None

    numerator = sum(
        (y - mean_y) * (x - mean_x)
        for x, y in points
    )
    slope = numerator / denominator
    intercept = mean_x - slope * mean_y
    return (
        int(slope * y_bottom + intercept),
        y_bottom,
        int(slope * y_top + intercept),
        y_top,
    )

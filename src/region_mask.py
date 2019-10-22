"""Apply polygonal regions of interest to small image-like matrices."""

from __future__ import annotations

from typing import Any, Iterable, Sequence, Tuple


Point = Tuple[float, float]
MaskedImage = Tuple[Tuple[Any, ...], ...]


def _point_on_segment(x: float, y: float, start: Point, end: Point) -> bool:
    x1, y1 = start
    x2, y2 = end
    cross_product = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if cross_product != 0:
        return False
    return min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)


def _point_in_polygon(x: float, y: float, vertices: Sequence[Point]) -> bool:
    inside = False
    previous = vertices[-1]
    for current in vertices:
        if _point_on_segment(x, y, previous, current):
            return True
        x1, y1 = previous
        x2, y2 = current
        if (y1 > y) != (y2 > y):
            intersection_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < intersection_x:
                inside = not inside
        previous = current
    return inside


def _black_pixel(pixel: Any) -> Any:
    if isinstance(pixel, tuple):
        return tuple(0 for _ in pixel)
    if isinstance(pixel, list):
        return [0 for _ in pixel]
    return 0


def mask_polygon(
    image: Iterable[Sequence[Any]],
    vertices: Sequence[Point],
) -> MaskedImage:
    """Keep pixels inside or on ``vertices`` and black out the remainder."""

    rows = tuple(tuple(row) for row in image)
    polygon = tuple(vertices)
    return tuple(
        tuple(
            pixel if _point_in_polygon(x, y, polygon) else _black_pixel(pixel)
            for x, pixel in enumerate(row)
        )
        for y, row in enumerate(rows)
    )

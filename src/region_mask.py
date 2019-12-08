"""Apply polygonal regions of interest to small image-like matrices."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Iterable, Sequence, Tuple


Point = Tuple[float, float]
MaskedImage = Tuple[Tuple[Any, ...], ...]


def _normalize_image(image: Iterable[Sequence[Any]]) -> MaskedImage:
    if image is None or isinstance(image, (str, bytes)):
        raise ValueError("image must be a non-empty rectangular matrix")
    try:
        raw_rows = tuple(image)
    except TypeError as error:
        raise ValueError("image must be a non-empty rectangular matrix") from error
    if not raw_rows:
        raise ValueError("image must contain at least one row")

    rows = []
    for raw_row in raw_rows:
        if isinstance(raw_row, (str, bytes)):
            raise ValueError("image rows must be pixel sequences")
        try:
            row = tuple(raw_row)
        except TypeError as error:
            raise ValueError("image rows must be pixel sequences") from error
        rows.append(row)

    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("image rows must have one consistent non-zero width")
    return tuple(rows)


def _normalize_polygon(vertices: Sequence[Point]) -> Tuple[Point, ...]:
    if vertices is None or isinstance(vertices, (str, bytes)):
        raise ValueError("vertices must describe a polygon")
    try:
        raw_points = tuple(vertices)
    except TypeError as error:
        raise ValueError("vertices must describe a polygon") from error
    if len(raw_points) < 3:
        raise ValueError("a polygon requires at least three vertices")

    polygon = []
    for raw_point in raw_points:
        try:
            point = tuple(raw_point)
        except TypeError as error:
            raise ValueError("each vertex must contain two coordinates") from error
        if len(point) != 2:
            raise ValueError("each vertex must contain two coordinates")
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            for value in point
        ):
            raise ValueError("vertex coordinates must be finite real numbers")
        normalized_point = (float(point[0]), float(point[1]))
        if not polygon or polygon[-1] != normalized_point:
            polygon.append(normalized_point)

    if len(polygon) > 1 and polygon[0] == polygon[-1]:
        polygon.pop()
    if len(polygon) < 3 or len(set(polygon)) < 3:
        raise ValueError("a polygon requires at least three distinct vertices")

    area_terms = (
        start[0] * end[1] - end[0] * start[1]
        for start, end in zip(polygon, polygon[1:] + polygon[:1])
    )
    try:
        doubled_area = math.fsum(area_terms)
    except (OverflowError, ValueError) as error:
        raise ValueError("polygon coordinates are too large to process safely") from error
    if not math.isfinite(doubled_area):
        raise ValueError("polygon coordinates are too large to process safely")
    if doubled_area == 0:
        raise ValueError("polygon vertices must enclose a non-zero area")
    return tuple(polygon)


def _point_on_segment(x: float, y: float, start: Point, end: Point) -> bool:
    x1, y1 = start
    x2, y2 = end
    cross_product = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    scale = max(abs(x - x1), abs(y - y1), abs(x2 - x1), abs(y2 - y1), 1.0)
    if not math.isclose(cross_product, 0.0, abs_tol=1e-12 * scale * scale):
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

    rows = _normalize_image(image)
    polygon = _normalize_polygon(vertices)
    return tuple(
        tuple(
            pixel if _point_in_polygon(x, y, polygon) else _black_pixel(pixel)
            for x, pixel in enumerate(row)
        )
        for y, row in enumerate(rows)
    )


def _scanline_columns(
    width: int,
    y: int,
    vertices: Sequence[Point],
) -> set[int]:
    intersections = []
    boundary_columns = set()
    previous = vertices[-1]

    for current in vertices:
        x1, y1 = previous
        x2, y2 = current
        if y1 == y2 == y:
            start = max(0, math.ceil(min(x1, x2)))
            stop = min(width - 1, math.floor(max(x1, x2)))
            boundary_columns.update(range(start, stop + 1))
        elif y1 != y2:
            boundary_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            nearest_column = round(boundary_x)
            if (
                min(y1, y2) <= y <= max(y1, y2)
                and math.isclose(boundary_x, nearest_column, abs_tol=1e-12)
            ):
                column = int(nearest_column)
                if 0 <= column < width:
                    boundary_columns.add(column)
            if min(y1, y2) <= y < max(y1, y2):
                intersections.append(boundary_x)
        previous = current

    intersections.sort()
    inside_columns = set(boundary_columns)
    for left, right in zip(intersections[::2], intersections[1::2]):
        start = max(0, math.ceil(left))
        stop = min(width - 1, math.floor(right))
        inside_columns.update(range(start, stop + 1))
    return inside_columns


def mask_polygon_scanline(
    image: Iterable[Sequence[Any]],
    vertices: Sequence[Point],
) -> MaskedImage:
    """Mask an image by filling polygon spans one horizontal row at a time."""

    rows = _normalize_image(image)
    polygon = _normalize_polygon(vertices)
    return tuple(
        tuple(
            pixel if x in inside_columns else _black_pixel(pixel)
            for x, pixel in enumerate(row)
        )
        for row, inside_columns in (
            (row, _scanline_columns(len(row), y, polygon))
            for y, row in enumerate(rows)
        )
    )

"""Плоскость: пересечение, позади, параллельный луч."""

import pytest

from minirt.shapes import Plane
from minirt.vec import vec3

P = Plane(vec3(0, -1, 0), vec3(0, 1, 0), vec3(1, 1, 1))


def test_hit() -> None:
    hit = P.intersect(vec3(0, 0, 0), vec3(0, -1, 0))
    assert hit is not None and hit.t == pytest.approx(1.0)


def test_behind() -> None:
    assert P.intersect(vec3(0, 0, 0), vec3(0, 1, 0)) is None


def test_parallel() -> None:
    assert P.intersect(vec3(0, 0, 0), vec3(1, 0, 0)) is None

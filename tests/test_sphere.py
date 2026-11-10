"""Пересечения со сферой: мимо, снаружи, изнутри, позади."""

import pytest

from minirt.shapes import Sphere
from minirt.vec import vec3

ORIGIN = vec3(0, 0, 0)
DIRECTION = vec3(0, 0, 1)
SPHERE = Sphere(vec3(0, 0, 5), 1.0, vec3(1, 0, 0))


def test_hit_outside() -> None:
    hit = SPHERE.intersect(ORIGIN, DIRECTION)
    assert hit is not None
    assert hit.t == pytest.approx(4.0)


def test_miss() -> None:
    assert SPHERE.intersect(vec3(0, 5, 0), DIRECTION) is None


def test_from_inside_picks_far_root() -> None:
    hit = SPHERE.intersect(vec3(0, 0, 5), DIRECTION)  # из центра
    assert hit is not None
    assert hit.t == pytest.approx(1.0)


def test_behind_camera() -> None:
    assert SPHERE.intersect(ORIGIN, vec3(0, 0, -1)) is None

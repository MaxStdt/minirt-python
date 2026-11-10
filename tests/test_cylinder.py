"""Цилиндр: бок, крышка, мимо выше крышки, параллельно оси, нормали."""

import numpy as np
import pytest

from minirt.shapes import Cylinder
from minirt.vec import vec3

# ось Y, центр в (0,0,5), радиус 1, высота 2 → крышки на y=±1
C = Cylinder(vec3(0, 0, 5), vec3(0, 1, 0), 1.0, 2.0, vec3(0, 1, 0))


def test_hit_side() -> None:
    hit = C.intersect(vec3(0, 0, 0), vec3(0, 0, 1))
    assert hit is not None
    assert hit.t == pytest.approx(4.0)


def test_hit_bottom_cap() -> None:
    # луч снизу вверх в центр нижней крышки
    hit = C.intersect(vec3(0, -10, 5), vec3(0, 1, 0))
    assert hit is not None
    assert hit.t == pytest.approx(9.0)
    assert np.allclose(np.abs(hit.normal), [0, 1, 0])


def test_miss_above_cap() -> None:
    # луч идёт мимо: сбоку, но выше крышки
    assert C.intersect(vec3(0, 1.5, 0), vec3(0, 0, 1)) is None


def test_parallel_to_axis_misses_side() -> None:
    # ось X, луч вдоль X на расстоянии 2 > r — промах
    cyl = Cylinder(vec3(0, 0, 5), vec3(1, 0, 0), 1.0, 2.0, vec3(1, 0, 0))
    assert cyl.intersect(vec3(-10, 2, 5), vec3(1, 0, 0)) is None

"""Публичные тесты стартового набора."""

import numpy as np
import pytest

from minirt.shapes import Sphere
from minirt.vec import cross, dot, length, normalize, vec3


def test_vec_helpers() -> None:
    a, b = vec3(1, 0, 0), vec3(0, 1, 0)
    assert dot(a, b) == 0.0
    assert np.allclose(cross(a, b), [0.0, 0.0, 1.0])
    assert length(vec3(3, 4, 0)) == pytest.approx(5.0)
    n = normalize(vec3(0, 3, 4))
    assert length(n) == pytest.approx(1.0)
    assert np.allclose(n, [0.0, 0.6, 0.8])

    # ---------------------------------------------------------------


# Выбор корня (ближайший положительный)
# ---------------------------------------------------------------
def test_sphere_nearest_root() -> None:
    """Извне сферы берётся меньший положительный корень."""
    sphere = Sphere(vec3(0, 0, 5), 1.0, vec3(1, 0, 0))
    hit = sphere.intersect(vec3(0, 0, 0), vec3(0, 0, 1))
    assert hit is not None
    assert hit.t == pytest.approx(4.0)


def test_sphere_inside_far_root() -> None:
    """Изнутри сферы берётся дальний корень (ближний отрицателен)."""
    sphere = Sphere(vec3(0, 0, 5), 1.0, vec3(1, 0, 0))
    hit = sphere.intersect(vec3(0, 0, 5), vec3(0, 0, 1))
    assert hit is not None
    assert hit.t == pytest.approx(1.0)


# ---------------------------------------------------------------
# Нормали
# ---------------------------------------------------------------
def test_sphere_normal_outward() -> None:
    """Снаружи нормаль направлена от центра к точке."""
    sphere = Sphere(vec3(0, 0, 5), 1.0, vec3(1, 0, 0))
    hit = sphere.intersect(vec3(0, 0, 0), vec3(0, 0, 1))
    assert hit is not None
    assert np.allclose(hit.normal, [0, 0, -1], atol=1e-6)

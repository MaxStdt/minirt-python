"""Векторная математика: базовые свойства."""

import numpy as np
import pytest

from minirt.vec import cross, dot, length, normalize, vec3


def test_dot_cross() -> None:
    a, b = vec3(1, 0, 0), vec3(0, 1, 0)
    assert dot(a, b) == 0
    assert np.allclose(cross(a, b), [0, 0, 1])


def test_normalize() -> None:
    v = normalize(vec3(3, 0, 4))
    assert length(v) == pytest.approx(1.0)
    assert np.allclose(v, [0.6, 0.0, 0.8])

"""Публичные тесты стартового набора."""

import numpy as np
import pytest

from minirt.vec import cross, dot, length, normalize, vec3


def test_vec_helpers() -> None:
    a, b = vec3(1, 0, 0), vec3(0, 1, 0)
    assert dot(a, b) == 0.0
    assert np.allclose(cross(a, b), [0.0, 0.0, 1.0])
    assert length(vec3(3, 4, 0)) == pytest.approx(5.0)
    n = normalize(vec3(0, 3, 4))
    assert length(n) == pytest.approx(1.0)
    assert np.allclose(n, [0.0, 0.6, 0.8])

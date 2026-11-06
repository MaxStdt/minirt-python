"""Камера: ортонормированность, центральный луч, крайние лучи, вертикальный взгляд."""

import numpy as np
import pytest

from minirt.camera import camera_basis, primary_rays
from minirt.scene import Camera
from minirt.vec import vec3


def _cam(direction: tuple[float, float, float], fov: float = 70.0) -> Camera:
    return Camera(vec3(0, 0, -5), vec3(*direction), fov)


def test_basis_orthonormal() -> None:
    f, r, u = camera_basis(_cam((0, 0, 1)))
    for a in (f, r, u):
        assert np.linalg.norm(a) == pytest.approx(1.0)
    assert np.isclose(np.dot(f, r), 0.0, atol=1e-9)
    assert np.isclose(np.dot(f, u), 0.0, atol=1e-9)
    assert np.isclose(np.dot(r, u), 0.0, atol=1e-9)


def test_central_ray_matches_direction() -> None:
    width, height = 101, 101
    rays = primary_rays(_cam((0, 0, 1)), width, height)
    center = rays[height // 2, width // 2]
    assert np.allclose(center, [0, 0, 1], atol=1e-6)


def test_extreme_rays_fov() -> None:
    fov = 70.0
    rays = primary_rays(_cam((0, 0, 1), fov), 100, 100)
    # левый край: угол ≈ fov/2
    left = rays[50, 0]
    cos_angle = np.dot(left, [0, 0, 1])
    assert np.rad2deg(np.arccos(cos_angle)) == pytest.approx(fov / 2, rel=1e-2)


def test_looking_down_no_nan() -> None:
    rays = primary_rays(_cam((0, -1, 0)), 10, 10)
    assert np.all(np.isfinite(rays))

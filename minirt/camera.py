"""Камера: ортонормированный базис и массив направлений первичных лучей.

Реализация соответствует разделу 4.1: плоскость на расстоянии 1,
формулы u, v с центром пикселя (+0.5). Все лучи считаются сразу
массивом формы (H, W, 3) — это и есть основной приём производительности.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from minirt.scene import Camera
from minirt.vec import Vec, cross, dot, normalize, vec3


def camera_basis(camera: Camera) -> tuple[Vec, Vec, Vec]:
    """Вернуть (forward, right, up) — ортонормированный базис.

    right = normalize(up_мира × forward), up = forward × right.
    Если forward параллелен (0,1,0) (смотрим строго вверх/вниз),
    вместо up_мира берём ось z.
    """
    forward = normalize(camera.direction)
    world_up = vec3(0.0, 1.0, 0.0)
    if abs(dot(world_up, forward)) > 1.0 - 1e-9:  # почти вертикально
        world_up = vec3(0.0, 0.0, 1.0)
    right = normalize(cross(world_up, forward))
    up = normalize(cross(forward, right))
    return forward, right, up


def primary_rays(camera: Camera, width: int, height: int) -> NDArray[np.float64]:
    """Массив направлений лучей формы (H, W, 3) через центры пикселей."""
    forward, right, up = camera_basis(camera)
    tan_half = float(np.tan(np.deg2rad(camera.fov) / 2.0))
    aspect = height / width

    # u: от -tan(fov/2) слева до +tan(fov/2) справа
    us = (2.0 * (np.arange(width) + 0.5) / width - 1.0) * tan_half
    # v: от +tan(fov/2)*H/W вверху до -... внизу
    vs = (1.0 - 2.0 * (np.arange(height) + 0.5) / height) * tan_half * aspect

    u_grid, v_grid = np.meshgrid(us, vs)  # (H, W)
    dirs = (
        forward[None, None, :]
        + u_grid[..., None] * right[None, None, :]
        + v_grid[..., None] * up[None, None, :]
    )
    # Нормируем каждый луч
    norms = np.sqrt(np.sum(dirs * dirs, axis=-1, keepdims=True))
    return (dirs / norms).astype(np.float64)

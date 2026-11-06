"""Трассировка: ближайший объект (без освещения)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from minirt.camera import primary_rays
from minirt.scene import Scene, SceneObject
from minirt.shapes import EPS, Sphere


def _intersect_batch(
    obj: SceneObject, origins: NDArray[np.float64], dirs: NDArray[np.float64]
) -> NDArray[np.float64]:
    n = origins.shape[0]
    t_out = np.full(n, np.inf, dtype=np.float64)
    if isinstance(obj, Sphere):
        oc = origins - obj.center
        b = np.sum(oc * dirs, axis=1)
        c = np.sum(oc * oc, axis=1) - obj.radius**2
        disc = b * b - c
        ok = disc >= 0
        if np.any(ok):
            sq = np.sqrt(np.maximum(disc, 0.0))
            t1 = -b - sq
            t2 = -b + sq
            t = np.where(t1 > EPS, t1, np.where(t2 > EPS, t2, np.inf))
            t_out = np.where(ok, t, np.inf)
    return t_out


def _compute_normals(obj: SceneObject, points: NDArray[np.float64]) -> NDArray[np.float64]:
    if isinstance(obj, Sphere):
        n = points - obj.center
        norm = np.sqrt(np.sum(n * n, axis=1, keepdims=True))
        return n / np.maximum(norm, 1e-12)
    return np.zeros_like(points)


def render(scene: Scene, width: int, height: int) -> NDArray[np.uint8]:
    camera = scene.camera
    origins_flat = np.broadcast_to(camera.position, (height, width, 3)).reshape(-1, 3)
    dirs_flat = primary_rays(camera, width, height).reshape(-1, 3)
    n = origins_flat.shape[0]

    best_t = np.full(n, np.inf)
    best_idx = np.full(n, -1, dtype=np.int64)
    for idx, obj in enumerate(scene.objects):
        t = _intersect_batch(obj, origins_flat, dirs_flat)
        closer = t < best_t
        best_t = np.where(closer, t, best_t)
        best_idx = np.where(closer, idx, best_idx)

    result = np.zeros((n, 3), dtype=np.float64)
    for idx, obj in enumerate(scene.objects):
        mask = best_idx == idx
        result[mask] = obj.color

    return (result.reshape(height, width, 3) * 255.0).astype(np.uint8)

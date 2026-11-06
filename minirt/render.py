"""Трассировка: ближайший объект, освещение по Ламберту, жёсткие тени.

Приём: первичные лучи обрабатываются массивом (N, 3), а цикл идёт
только по объектам сцены. Для каждого объекта векторно находим t
всех лучей, обновляем ближайшее попадание через np.where.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from minirt.camera import primary_rays
from minirt.scene import Scene, SceneObject
from minirt.shapes import EPS, Cylinder, Plane, Sphere
from minirt.vec import dot


def _intersect_batch(
    obj: SceneObject, origins: NDArray[np.float64], dirs: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Векторное пересечение объекта с пакетом лучей.

    Возвращает массив t формы (N,), где np.inf — промах.
    Реализуем формулы из shapes.py, но без создания Hit на каждый луч.
    """
    n = origins.shape[0]
    t_out = np.full(n, np.inf, dtype=np.float64)

    if isinstance(obj, Sphere):
        oc = origins - obj.center  # (N,3)
        b = np.sum(oc * dirs, axis=1)  # (N,)
        c = np.sum(oc * oc, axis=1) - obj.radius**2
        disc = b * b - c
        ok = disc >= 0
        if np.any(ok):
            sq = np.sqrt(np.maximum(disc, 0.0))
            t1 = -b - sq
            t2 = -b + sq
            t = np.where(t1 > EPS, t1, np.where(t2 > EPS, t2, np.inf))
            t_out = np.where(ok, t, np.inf)

    elif isinstance(obj, Plane):
        denom = dirs @ obj.normal  # (N,)
        num = float(dot(obj.point, obj.normal))
        # t = (P0 - O)·n / (D·n)
        t = (num - origins @ obj.normal) / np.where(np.abs(denom) > 1e-12, denom, np.inf)
        t_out = np.where(t > EPS, t, np.inf)

    elif isinstance(obj, Cylinder):
        a = obj.axis
        v = origins - obj.center  # (N,3)
        da = dirs @ a  # (N,)
        d_perp = dirs - da[:, None] * a[None, :]
        va = v @ a
        v_perp = v - va[:, None] * a[None, :]

        best = np.full(n, np.inf, dtype=np.float64)
        # Бок
        qa = np.sum(d_perp * d_perp, axis=1)
        qb = np.sum(d_perp * v_perp, axis=1)
        qc = np.sum(v_perp * v_perp, axis=1) - obj.radius**2
        disc = qb * qb - qa * qc
        valid = (qa > 1e-12) & (disc >= 0)
        if np.any(valid):
            sq = np.sqrt(np.maximum(disc, 0.0))
            t1 = (-qb - sq) / np.where(qa > 1e-12, qa, 1.0)
            t2 = (-qb + sq) / np.where(qa > 1e-12, qa, 1.0)
            for t in (t1, t2):
                m = va + t * da
                inside = valid & (t > EPS) & (np.abs(m) <= obj.height / 2 + 1e-9)
                best = np.where(inside & (t < best), t, best)
        # Крышки
        ok_caps = np.abs(da) > 1e-12
        if np.any(ok_caps):
            for sign in (+1.0, -1.0):
                m_cap = sign * obj.height / 2
                t = (m_cap - va) / np.where(ok_caps, da, 1.0)
                p = origins + t[:, None] * dirs
                radial = p - obj.center - m_cap * a[None, :]
                inside = (t > EPS) & (np.sum(radial * radial, axis=1) <= obj.radius**2 + 1e-9)
                best = np.where(inside & (t < best), t, best)
        t_out = best

    return t_out


def _compute_normals(obj: SceneObject, points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Нормали к поверхности в точках попадания (форма (N, 3))."""
    if isinstance(obj, Sphere):
        n = points - obj.center
    elif isinstance(obj, Plane):
        return np.broadcast_to(obj.normal, points.shape).copy()
    else:  # Cylinder
        a = obj.axis
        v = points - obj.center
        va = v @ a
        radial = v - va[:, None] * a[None, :]
        radial_len = np.sqrt(np.sum(radial * radial, axis=1))
        # На боковой поверхности |radial| ≈ r; на крышке — меньше и m = ±h/2
        on_cap = np.abs(np.abs(va) - obj.height / 2) < 1e-6
        n = np.where(
            on_cap[:, None],
            np.sign(va)[:, None] * a[None, :],
            radial / np.maximum(radial_len[:, None], 1e-12),
        )
    # Нормируем
    norm = np.sqrt(np.sum(n * n, axis=1, keepdims=True))
    return n / np.maximum(norm, 1e-12)


def render(scene: Scene, width: int, height: int) -> NDArray[np.uint8]:
    """Отрендерить изображение (H, W, 3) uint8."""
    camera = scene.camera
    origins_flat = np.broadcast_to(camera.position, (height, width, 3)).reshape(-1, 3)
    dirs_flat = primary_rays(camera, width, height).reshape(-1, 3)
    n = origins_flat.shape[0]

    # 1) Ближайшее попадание среди всех объектов
    best_t = np.full(n, np.inf)
    best_idx = np.full(n, -1, dtype=np.int64)
    for idx, obj in enumerate(scene.objects):
        t = _intersect_batch(obj, origins_flat, dirs_flat)
        closer = t < best_t
        best_t = np.where(closer, t, best_t)
        best_idx = np.where(closer, idx, best_idx)

    hit_mask = best_idx >= 0
    # 2) Собираем точки и нормали только для попавших лучей
    points = origins_flat[hit_mask] + best_t[hit_mask, None] * dirs_flat[hit_mask]

    normals = np.zeros_like(points)
    colors = np.zeros_like(points)
    for idx, obj in enumerate(scene.objects):
        # индексы оригинальных лучей, попавших в этот объект
        obj_hit = np.zeros(n, dtype=bool)
        obj_hit[hit_mask] = best_idx[hit_mask] == idx
        if not np.any(obj_hit):
            continue
        pts = origins_flat[obj_hit] + best_t[obj_hit, None] * dirs_flat[obj_hit]
        normals[best_idx[hit_mask] == idx] = _compute_normals(obj, pts)
        colors[best_idx[hit_mask] == idx] = obj.color

    # 3) Нормаль к наблюдателю
    d_hit = dirs_flat[hit_mask]
    flip = np.sum(normals * d_hit, axis=1) > 0
    normals[flip] *= -1.0

    # 4) Освещение: фон + Ламберт с тенью
    ambient = scene.ambient
    light = scene.light
    result = np.zeros((n, 3), dtype=np.float64)
    result[:] = ambient.ratio * ambient.color[None, :]

    if np.any(hit_mask):
        to_light = light.position[None, :] - points  # (M,3)
        dist = np.sqrt(np.sum(to_light * to_light, axis=1))
        l_dir = to_light / np.maximum(dist[:, None], 1e-12)

        # Теневые лучи: сдвиг начала вдоль нормали против акне
        shadow_origins = points + normals * 1e-3
        shadow_occluded = np.zeros(points.shape[0], dtype=bool)
        # максимальное расстояние до источника минус сдвиг
        max_t = dist - 1e-3
        for obj in scene.objects:
            t_sh = _intersect_batch(obj, shadow_origins, l_dir)
            shadow_occluded |= t_sh < max_t

        cos_theta = np.maximum(np.sum(normals * l_dir, axis=1), 0.0)
        lambert = light.ratio * cos_theta[:, None] * light.color[None, :] * colors
        lambert[shadow_occluded] = 0.0
        result[hit_mask] += lambert

    result = np.clip(result, 0.0, 1.0)
    img = (result.reshape(height, width, 3) * 255.0).astype(np.uint8)
    return img

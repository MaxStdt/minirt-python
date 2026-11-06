"""Геометрия: пересечения луча с объектами и нормали.

Все пересечения возвращают ближайшее t > EPS или None.
Нормаль в Hit уже развёрнута к наблюдателю (правило 2 раздела 4.2):
если n·D > 0 (видна обратная сторона), берём -n.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from minirt.vec import Vec, dot, normalize

EPS = 1e-4


@dataclass
class Hit:
    """Результат пересечения: расстояние t, точка, нормаль, цвет объекта."""

    t: float
    point: Vec
    normal: Vec
    color: Vec


def _oriented_normal(n: Vec, direction: Vec) -> Vec:
    """Развернуть нормаль к наблюдателю: если n·D > 0, вернуть -n."""
    return -n if dot(n, direction) > 0 else n


class Sphere:
    """Сфера: |P - C|^2 = r^2."""

    def __init__(self, center: Vec, radius: float, color: Vec) -> None:
        self.center = center
        self.radius = float(radius)
        self.color = color

    def intersect(self, origin: Vec, direction: Vec) -> Hit | None:
        # Уравнение t^2 + 2bt + c = 0 при |D|=1: b = (O-C)·D, c = |O-C|^2 - r^2
        oc = origin - self.center
        b = dot(oc, direction)
        c = dot(oc, oc) - self.radius * self.radius
        disc = b * b - c
        if disc < 0:
            return None
        sq = float(np.sqrt(disc))
        t1, t2 = -b - sq, -b + sq
        t = t1 if t1 > EPS else (t2 if t2 > EPS else None)
        if t is None:
            return None
        p = origin + t * direction
        n = normalize(p - self.center)  # для внутренней стороны потом развернём
        return Hit(t, p, _oriented_normal(n, direction), self.color)


class Plane:
    """Плоскость через P0 с нормалью n: (P(t) - P0)·n = 0."""

    def __init__(self, point: Vec, normal: Vec, color: Vec) -> None:
        self.point = point
        self.normal = normalize(normal)
        self.color = color

    def intersect(self, origin: Vec, direction: Vec) -> Hit | None:
        denom = dot(direction, self.normal)
        if abs(denom) < 1e-12:  # луч параллелен плоскости
            return None
        t = dot(self.point - origin, self.normal) / denom
        if t <= EPS:
            return None
        p = origin + t * direction
        return Hit(t, p, _oriented_normal(self.normal, direction), self.color)


class Cylinder:
    """Конечный цилиндр с крышками.

    Центр C — середина оси, ось a единичная, радиус r, высота h.
    Бок: |v_perp + t·D_perp|^2 = r^2, проверка проекции m ∈ [-h/2, h/2].
    Крышки: плоскость m = ±h/2, диск радиуса r.
    """

    def __init__(self, center: Vec, axis: Vec, radius: float, height: float, color: Vec) -> None:
        self.center = center
        self.axis = normalize(axis)
        self.radius = float(radius)
        self.height = float(height)
        self.color = color

    def intersect(self, origin: Vec, direction: Vec) -> Hit | None:
        a = self.axis
        v = origin - self.center
        d_perp = direction - dot(direction, a) * a
        v_perp = v - dot(v, a) * a

        best_t: float | None = None
        best_n: Vec | None = None

        # --- Боковая поверхность ---
        qa = dot(d_perp, d_perp)
        if qa > 1e-12:  # иначе луч параллелен оси — бок не задеть
            qb = dot(d_perp, v_perp)
            qc = dot(v_perp, v_perp) - self.radius * self.radius
            disc = qb * qb - qa * qc
            if disc >= 0:
                sq = float(np.sqrt(disc))
                for t in ((-qb - sq) / qa, (-qb + sq) / qa):
                    if t <= EPS:
                        continue
                    m = dot(v + t * direction, a)
                    if -self.height / 2 - 1e-9 <= m <= self.height / 2 + 1e-9:
                        p = origin + t * direction
                        n = normalize(p - self.center - m * a)
                        if best_t is None or t < best_t:
                            best_t, best_n = t, n
                        break  # ближайший корень бока найден

        # --- Крышки ---
        da = dot(direction, a)
        if abs(da) > 1e-12:
            for sign in (+1.0, -1.0):
                m_cap = sign * self.height / 2
                t = (m_cap - dot(v, a)) / da
                if t <= EPS:
                    continue
                p = origin + t * direction
                if dot(p - self.center - m_cap * a, p - self.center - m_cap * a) <= (
                    self.radius * self.radius + 1e-9
                ):
                    n = sign * a  # нормаль крышки — ±a
                    if best_t is None or t < best_t:
                        best_t, best_n = t, n

        if best_t is None or best_n is None:
            return None
        p = origin + best_t * direction
        return Hit(best_t, p, _oriented_normal(best_n, direction), self.color)

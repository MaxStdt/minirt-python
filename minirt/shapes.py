from __future__ import annotations

from dataclasses import dataclass

from minirt.vec import Vec

EPS = 1e-4


@dataclass
class Hit:
    """Результат пересечения: расстояние, точка, нормаль, цвет."""

    t: float
    point: Vec
    normal: Vec
    color: Vec


class Sphere:
    """Пока заглушка"""

    def __init__(self, center: Vec, radius: float, color: Vec) -> None:
        self.center = center
        self.radius = float(radius)
        self.color = color


class Plane:
    """Пока заглушка"""

    def __init__(self, point: Vec, normal: Vec, color: Vec) -> None:
        self.point = point
        self.normal = normal
        self.color = color


class Cylinder:
    """Пока заглушка"""

    def __init__(self, center: Vec, axis: Vec, radius: float, height: float, color: Vec) -> None:
        self.center = center
        self.axis = axis
        self.radius = float(radius)
        self.height = float(height)
        self.color = color

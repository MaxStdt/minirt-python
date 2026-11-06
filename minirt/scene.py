"""Модель сцены: фоновый свет, камера, источник света, объекты."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Импорт только для проверки типов, чтобы избежать цикла scene -> shapes
    from minirt.shapes import Cylinder, Plane, Sphere

    SceneObject = Sphere | Plane | Cylinder
else:
    # Во время исполнения тип не нужен — dataclass работает с любыми значениями
    SceneObject = object


from minirt.vec import Vec


@dataclass
class Ambient:
    """Фоновый свет A: яркость 0..1 и цвет 0..1 (нормированный)."""

    ratio: float
    color: Vec  # (3,) в диапазоне 0..1


@dataclass
class Camera:
    """Камера C: позиция, направление взгляда, горизонтальный угол обзора в градусах."""

    position: Vec
    direction: Vec  # уже нормирован парсером
    fov: float  # градусы, 0 < fov < 180


@dataclass
class Light:
    """Точечный источник L: позиция, яркость 0..1, цвет 0..1."""

    position: Vec
    ratio: float
    color: Vec


# Тип-алиас для объекта сцены; импорт Sphere/Plane/Cylinder ленивый
# (в shapes.py), чтобы избежать циклического импорта.
Object = "Sphere | Plane | Cylinder"


@dataclass
class Scene:
    """Сцена целиком: A, C, L (список — на будущее для доп. задания), объекты."""

    ambient: Ambient
    camera: Camera
    lights: list[Light]
    objects: list[SceneObject] = field(default_factory=list)

    @property
    def light(self) -> Light:
        """Обязательная часть использует ровно один источник L."""
        return self.lights[0]

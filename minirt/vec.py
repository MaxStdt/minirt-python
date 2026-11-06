"""Векторная математика на NumPy.

Все функции — чистые, принимают и возвращают np.ndarray формы (3,)
или массив формы (N, 3). Никаких вызовов np.linalg.norm там, где
достаточно sqrt(dot) — это влияет на производительность (раздел 6).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Vec = NDArray[np.float64]


def vec3(x: float, y: float, z: float) -> Vec:
    """Создать вектор (3,) из трёх чисел."""
    return np.array([x, y, z], dtype=np.float64)


def dot(a: Vec, b: Vec) -> float:
    """Скалярное произведение."""
    return float(np.dot(a, b))


def cross(a: Vec, b: Vec) -> Vec:
    """Векторное произведение."""
    return np.cross(a, b).astype(np.float64)


def length(a: Vec) -> float:
    """Длина вектора через sqrt(dot) — быстрее np.linalg.norm."""
    return float(np.sqrt(np.dot(a, a)))


def normalize(a: Vec) -> Vec:
    """Нормировать вектор. Для нулевого — вернуть как есть (защита от деления на 0)."""
    n = length(a)
    if n < 1e-12:
        return a.astype(np.float64)
    return (a / n).astype(np.float64)

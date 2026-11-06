"""Проект рендера."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from minirt.scene import Scene


def render(scene: Scene, width: int, height: int) -> NDArray[np.uint8]:
    """Вернуть чёрное изображение нужного размера."""
    return np.zeros((height, width, 3), dtype=np.uint8)

"""Освещение: cos=1 полная яркость, скользящий ноль, тень, фон."""

from __future__ import annotations

import numpy as np

from minirt.render import render
from minirt.scene import Ambient, Camera, Light, Scene
from minirt.shapes import Plane, Sphere
from minirt.vec import vec3

_DEFAULT_LIGHT_POS = vec3(0, 5, 0)


def _scene(objects: list[object], light_pos=None) -> Scene:
    if light_pos is None:
        light_pos = _DEFAULT_LIGHT_POS
    return Scene(
        Ambient(0.1, vec3(1, 1, 1)),
        Camera(vec3(0, 0, -5), vec3(0, -0.5, 1), 70.0),
        [Light(light_pos, 1.0, vec3(1, 1, 1))],
        objects,
    )


def test_lighting_gradient_on_plane() -> None:
    img = render(_scene([Plane(vec3(0, -1, 0), vec3(0, 1, 0), vec3(1, 1, 1))]), 50, 50)
    assert img[-1, :, :].max() > 0
    assert img[-1, :, :].mean() > img[0, :, :].mean()


def test_shadow_on_plane() -> None:
    """Сравниваем один и тот же пиксель пола с тенью и без неё."""
    plane = Plane(vec3(0, -1, 0), vec3(0, 1, 0), vec3(1, 1, 1))

    scene_open = _scene([plane])
    img_open = render(scene_open, 80, 60).astype(np.int32)

    scene_shadow = _scene([plane, Sphere(vec3(0, 0, 0), 1.0, vec3(1, 1, 1))])
    img_shadow = render(scene_shadow, 80, 60).astype(np.int32)

    diff = (img_open - img_shadow).max(axis=2)
    assert diff.max() > 50  # тень есть и заметна
    assert (diff > 50).sum() > 100  # площадь тени разумна

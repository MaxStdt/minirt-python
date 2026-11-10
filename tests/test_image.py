"""Изображение: размер, тип, фон верхней строки, освещённый пол внизу."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from minirt.parser import parse_scene_file
from minirt.render import render

SCENES = Path(__file__).resolve().parent.parent / "scenes"


def test_image_size_and_dtype() -> None:
    scene = parse_scene_file(SCENES / "three.rt")
    img = render(scene, 40, 30)
    assert img.shape == (30, 40, 3)
    assert img.dtype == np.uint8


def test_background_matches_ambient() -> None:
    """Верх кадра — фон, окрашенный ambient: ratio_A * 255."""
    scene = parse_scene_file(SCENES / "three.rt")
    img = render(scene, 40, 30)
    expected = int(round(scene.ambient.ratio * 255))
    assert img[0, :, :].max() == expected


def test_lit_floor_bottom() -> None:
    scene = parse_scene_file(SCENES / "three.rt")
    img = render(scene, 40, 30)
    assert img[-1, :, :].max() > 0

"""Тесты парсера: успешный разбор, нормировка, радиус=диаметр/2, ошибки."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from minirt.errors import SceneFormatError
from minirt.parser import parse_scene_file, parse_scene_text
from minirt.shapes import Sphere

_BAD_SCENES = sorted((Path(__file__).parent.parent / "scenes" / "bad").glob("*.rt"))

THREE = """
# комментарий
A 0.2 255,255,255
C 0,3,-10 0,-0.2,1 70
L -6,8,-4 0.8 255,255,255
pl 0,-1,0 0,1,0 200,200,200
sp -2.5,0.2,4 3 220,50,50
sp 2.5,-0.2,2.5 1.6 60,200,80
cy 5,0,7 0,1,0 2 4 70,110,230
"""


def test_parse_all_elements() -> None:
    scene = parse_scene_text(THREE)
    assert scene.ambient.ratio == pytest.approx(0.2)
    assert len(scene.lights) == 1
    assert len(scene.objects) == 4  # plane + 2 spheres + cylinder


def test_camera_direction_normalized() -> None:
    scene = parse_scene_text(THREE)
    d = scene.camera.direction
    assert np.linalg.norm(d) == pytest.approx(1.0)


def test_sphere_radius_is_half_diameter() -> None:
    scene = parse_scene_text(THREE)
    spheres = [o for o in scene.objects if isinstance(o, Sphere)]
    assert spheres[0].radius == pytest.approx(1.5)  # 3 / 2


def test_comments_and_blank_lines_skipped() -> None:
    scene = parse_scene_text(THREE)
    assert scene is not None


@pytest.mark.parametrize("bad_file", _BAD_SCENES)
def test_bad_scenes_raise_with_line(bad_file: Path) -> None:
    with pytest.raises(SceneFormatError) as exc:
        parse_scene_file(bad_file)
    assert "строка" in str(exc.value)


def test_sphere_missing_field() -> None:
    text = "A 0.5 255,255,255\nC 0,0,-5 0,0,1 70\nL 0,5,0 1 255,255,255\nsp 0,0,0 1"
    with pytest.raises(SceneFormatError) as exc:
        parse_scene_text(text)
    assert "ожидается 3 полей" in str(exc.value)


def test_brightness_range() -> None:
    text = "A 1.5 255,255,255\nC 0,0,-5 0,0,1 70\nL 0,5,0 1 255,255,255"
    with pytest.raises(SceneFormatError) as exc:
        parse_scene_text(text)
    assert "диапазоне 0..1" in str(exc.value)

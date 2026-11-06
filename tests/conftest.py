"""Общие фикстуры pytest: путь к сценам, простые объекты."""

from __future__ import annotations

from pathlib import Path

import pytest

from minirt.scene import Ambient, Camera, Light, Scene
from minirt.vec import vec3

SCENES = Path(__file__).resolve().parent.parent / "scenes"


@pytest.fixture
def three_scene_path() -> Path:
    return SCENES / "three.rt"


@pytest.fixture
def bad_scenes() -> list[Path]:
    return sorted((SCENES / "bad").glob("*.rt"))


@pytest.fixture
def simple_scene() -> Scene:
    return Scene(
        ambient=Ambient(0.2, vec3(1, 1, 1)),
        camera=Camera(vec3(0, 0, -5), vec3(0, 0, 1), 70.0),
        lights=[Light(vec3(0, 5, 0), 1.0, vec3(1, 1, 1))],
        objects=[],
    )

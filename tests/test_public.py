"""Публичные тесты стартового набора.

Фиксируют имена модулей, сигнатуры функций и ключевые соглашения:
базис камеры, выбор корня, нормали, формулы освещения.
Прохождение этих тестов — необходимое условие проверки работы.
"""

import numpy as np
import pytest

# Проверка имен модулей и импортов
from minirt.camera import camera_basis, primary_rays
from minirt.scene import Ambient, Camera, Light, Scene
from minirt.shapes import Plane, Sphere
from minirt.vec import cross, dot, length, normalize, vec3


# ---------------------------------------------------------------
# Соглашения по векторной математике
# ---------------------------------------------------------------
def test_vec_helpers() -> None:
    """Базовые операции возвращают корректные типы и значения."""
    a = vec3(1.0, 0.0, 0.0)
    b = vec3(0.0, 1.0, 0.0)
    assert dot(a, b) == 0.0
    assert np.allclose(cross(a, b), [0.0, 0.0, 1.0])
    assert length(vec3(3.0, 4.0, 0.0)) == pytest.approx(5.0)
    n = normalize(vec3(0.0, 3.0, 4.0))
    assert length(n) == pytest.approx(1.0)
    assert np.allclose(n, [0.0, 0.6, 0.8])


# ---------------------------------------------------------------
# Базис камеры
# ---------------------------------------------------------------
def test_camera_basis_orthonormal() -> None:
    """forward, right, up — ортонормированный правый базис."""
    cam = Camera(vec3(0, 0, 0), vec3(0, 0, 1), 60.0)
    f, r, u = camera_basis(cam)
    assert np.isclose(length(f), 1.0)
    assert np.isclose(length(r), 1.0)
    assert np.isclose(length(u), 1.0)
    assert np.isclose(dot(f, r), 0.0, atol=1e-9)
    assert np.isclose(dot(f, u), 0.0, atol=1e-9)
    assert np.isclose(dot(r, u), 0.0, atol=1e-9)


def test_camera_central_ray() -> None:
    """Центральный луч совпадает с направлением взгляда."""
    cam = Camera(vec3(0, 0, -5), vec3(0, 0, 1), 70.0)
    rays = primary_rays(cam, 101, 101)  # нечётный размер -> есть центр
    center = rays[50, 50]
    assert np.allclose(center, [0, 0, 1], atol=1e-6)


# ---------------------------------------------------------------
# Выбор корня (ближайший положительный)
# ---------------------------------------------------------------
def test_sphere_nearest_root() -> None:
    """Извне сферы берётся меньший положительный корень."""
    sphere = Sphere(vec3(0, 0, 5), 1.0, vec3(1, 0, 0))
    hit = sphere.intersect(vec3(0, 0, 0), vec3(0, 0, 1))
    assert hit is not None
    assert hit.t == pytest.approx(4.0)


def test_sphere_inside_far_root() -> None:
    """Изнутри сферы берётся дальний корень (ближний отрицателен)."""
    sphere = Sphere(vec3(0, 0, 5), 1.0, vec3(1, 0, 0))
    hit = sphere.intersect(vec3(0, 0, 5), vec3(0, 0, 1))
    assert hit is not None
    assert hit.t == pytest.approx(1.0)


# ---------------------------------------------------------------
# Нормали
# ---------------------------------------------------------------
def test_sphere_normal_outward() -> None:
    """Снаружи нормаль направлена от центра к точке."""
    sphere = Sphere(vec3(0, 0, 5), 1.0, vec3(1, 0, 0))
    hit = sphere.intersect(vec3(0, 0, 0), vec3(0, 0, 1))
    assert hit is not None
    assert np.allclose(hit.normal, [0, 0, -1], atol=1e-6)


def test_plane_normal_constant() -> None:
    """Нормаль плоскости постоянна и развёрнута к наблюдателю."""
    plane = Plane(vec3(0, -1, 0), vec3(0, 1, 0), vec3(1, 1, 1))
    hit = plane.intersect(vec3(0, 0, 0), vec3(0, -1, 0))
    assert hit is not None
    assert np.allclose(hit.normal, [0, 1, 0])


def test_normal_flipped_to_viewer() -> None:
    """Если луч смотрит на обратную сторону плоскости, нормаль меняет знак."""
    plane = Plane(vec3(0, 0, 0), vec3(0, 1, 0), vec3(1, 1, 1))
    # луч идёт снизу вверх, нормаль плоскости смотрит вверх -> n·D > 0, нужно -n
    hit = plane.intersect(vec3(0, -5, 0), vec3(0, 1, 0))
    assert hit is not None
    assert np.allclose(hit.normal, [0, -1, 0])


# ---------------------------------------------------------------
# Формулы освещения
# ---------------------------------------------------------------
def test_lighting_lambert_cos_one() -> None:
    """При cos = 1 (свет точно по нормали) достигается полная яркость."""
    scene = Scene(
        ambient=Ambient(0.0, vec3(1, 1, 1)),
        camera=Camera(vec3(0, 1, -5), vec3(0, -0.5, 1), 70.0),
        lights=[Light(vec3(0, 5, 0), 1.0, vec3(1, 1, 1))],
        objects=[Plane(vec3(0, 0, 0), vec3(0, 1, 0), vec3(1, 1, 1))],
    )
    from minirt.render import render

    # маленькое разрешение, чтобы быстро; луч в центр должен попасть на плоскость
    img = render(scene, 11, 11)
    # Нормаль плоскости (0,1,0), свет сверху (0,5,0) -> l = (0,1,0), cos = 1
    # Ожидаем яркий пиксель в области попадания (в зависимости от камеры)
    assert img.max() > 0


def test_lighting_shadow() -> None:
    """Объект между точкой и источником отбрасывает тень (только фоновый свет)."""
    scene = Scene(
        ambient=Ambient(0.1, vec3(1, 1, 1)),
        camera=Camera(vec3(0, 0, -5), vec3(0, 0, 1), 70.0),
        lights=[Light(vec3(0, 5, 0), 1.0, vec3(1, 1, 1))],
        objects=[
            Plane(vec3(0, -1, 0), vec3(0, 1, 0), vec3(1, 1, 1)),
            Sphere(vec3(0, 0, 0), 0.5, vec3(1, 1, 1)),
        ],
    )
    from minirt.render import render

    img = render(scene, 81, 61)
    # Под сферой (в центре кадра) тень -> пиксели темнее, чем на открытом полу
    shadow_mean = img[35:45, 35:45, :].mean()
    open_floor_mean = img[-5:, :, :].mean()
    assert shadow_mean < open_floor_mean

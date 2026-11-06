"""Парсер файла сцены .rt.

Формат (раздел 3): каждая непустая строка — элемент; # — комментарий.
Векторы/цвета — три числа через запятую без пробелов.
При любом нарушении — SceneFormatError с номером строки и причиной.
Отсутствие файла (OSError) парсер не перехватывает — это делает main.
"""

from __future__ import annotations

from pathlib import Path

from minirt.errors import SceneFormatError
from minirt.scene import Ambient, Camera, Light, Scene, SceneObject
from minirt.shapes import Cylinder, Plane, Sphere
from minirt.vec import Vec, length, normalize, vec3


def _split_vec(token: str, line_no: int, what: str) -> Vec:
    """Разобрать "x,y,z" в вектор (3,)."""
    parts = token.split(",")
    if len(parts) != 3:
        raise SceneFormatError(line_no, f"{what} должен состоять из 3 чисел через запятую")
    try:
        vals = [float(p) for p in parts]
    except ValueError as exc:
        raise SceneFormatError(line_no, f"{what}: не число в '{token}'") from exc
    return vec3(vals[0], vals[1], vals[2])


def _parse_color(token: str, line_no: int) -> Vec:
    """Цвет — три целых 0..255, нормируем в 0..1 (покомпонентно)."""
    parts = token.split(",")
    if len(parts) != 3:
        raise SceneFormatError(line_no, "цвет должен состоять из 3 компонент через запятую")
    out = []
    for p in parts:
        try:
            v = int(p)
        except ValueError as exc:
            raise SceneFormatError(line_no, f"цвет: не целое '{p}'") from exc
        if not 0 <= v <= 255:
            raise SceneFormatError(line_no, f"цвет: компонента {v} вне 0..255")
        out.append(v / 255.0)
    return vec3(out[0], out[1], out[2])


def _parse_ratio(token: str, line_no: int, what: str) -> float:
    """Яркость — float 0..1."""
    try:
        v = float(token)
    except ValueError as exc:
        raise SceneFormatError(line_no, f"{what}: не число '{token}'") from exc
    if not 0.0 <= v <= 1.0:
        raise SceneFormatError(line_no, f"{what} должна быть в диапазоне 0..1, получено {token}")
    return v


def parse_scene_text(text: str) -> Scene:
    """Разобрать текст сцены в Scene.

    Проверяет: ровно один A, ровно одна C, хотя бы один L,
    диапазоны полей, положительность диаметров/высот.
    """
    ambient: Ambient | None = None
    camera: Camera | None = None
    lights: list[Light] = []
    objects: list[SceneObject] = []

    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        tag = parts[0]

        if tag == "A":
            if ambient is not None:
                raise SceneFormatError(line_no, "элемент A уже задан")
            if len(parts) != 3:
                raise SceneFormatError(line_no, "у элемента A ожидается 2 поля")
            ambient = Ambient(
                _parse_ratio(parts[1], line_no, "яркость фонового света"),
                _parse_color(parts[2], line_no),
            )

        elif tag == "C":
            if camera is not None:
                raise SceneFormatError(line_no, "элемент C уже задан")
            if len(parts) != 4:
                raise SceneFormatError(line_no, "у элемента C ожидается 3 поля")
            pos = _split_vec(parts[1], line_no, "позиция камеры")
            d = _split_vec(parts[2], line_no, "направление камеры")
            if any(abs(c) > 1.0 for c in d):
                raise SceneFormatError(line_no, "компоненты направления камеры в [-1, 1]")
            if length(d) < 1e-12:
                raise SceneFormatError(line_no, "направление камеры не может быть нулевым")
            try:
                fov = float(parts[3])
            except ValueError as exc:
                raise SceneFormatError(line_no, "угол обзора — не число") from exc
            if not 0.0 < fov < 180.0:
                raise SceneFormatError(line_no, f"угол обзора 0<fov<180, получено {fov}")
            camera = Camera(pos, normalize(d), fov)

        elif tag == "L":
            if len(parts) != 4:
                raise SceneFormatError(line_no, "у элемента L ожидается 3 поля")
            lights.append(
                Light(
                    _split_vec(parts[1], line_no, "позиция источника"),
                    _parse_ratio(parts[2], line_no, "яркость источника"),
                    _parse_color(parts[3], line_no),
                )
            )

        elif tag == "sp":
            if len(parts) != 4:
                raise SceneFormatError(
                    line_no, f"у элемента sp ожидается 3 полей, получено {len(parts) - 1}"
                )
            center = _split_vec(parts[1], line_no, "центр сферы")
            try:
                diam = float(parts[2])
            except ValueError as exc:
                raise SceneFormatError(line_no, "диаметр сферы — не число") from exc
            if diam <= 0:
                raise SceneFormatError(line_no, f"диаметр сферы > 0, получено {diam}")
            objects.append(Sphere(center, diam / 2.0, _parse_color(parts[3], line_no)))

        elif tag == "pl":
            if len(parts) != 4:
                raise SceneFormatError(line_no, "у элемента pl ожидается 3 поля")
            p0 = _split_vec(parts[1], line_no, "точка плоскости")
            n = normalize(_split_vec(parts[2], line_no, "нормаль плоскости"))
            objects.append(Plane(p0, n, _parse_color(parts[3], line_no)))

        elif tag == "cy":
            if len(parts) != 6:
                raise SceneFormatError(line_no, "у элемента cy ожидается 5 полей")
            center = _split_vec(parts[1], line_no, "центр цилиндра")
            axis = normalize(_split_vec(parts[2], line_no, "ось цилиндра"))
            try:
                diam = float(parts[3])
                h = float(parts[4])
            except ValueError as exc:
                raise SceneFormatError(line_no, "диаметр/высота цилиндра — не число") from exc
            if diam <= 0 or h <= 0:
                raise SceneFormatError(line_no, "диаметр и высота цилиндра > 0")
            objects.append(Cylinder(center, axis, diam / 2.0, h, _parse_color(parts[5], line_no)))

        else:
            raise SceneFormatError(line_no, f"неизвестный элемент '{tag}'")

    if ambient is None:
        raise SceneFormatError(0, "отсутствует обязательный элемент A")
    if camera is None:
        raise SceneFormatError(0, "отсутствует обязательный элемент C")
    if not lights:
        raise SceneFormatError(0, "отсутствует обязательный элемент L")

    return Scene(ambient, camera, lights, objects)


def parse_scene_file(path: str | Path) -> Scene:
    """Прочитать файл и разобрать. OSError (нет файла) пробрасывается в main."""
    return parse_scene_text(Path(path).read_text(encoding="utf-8"))

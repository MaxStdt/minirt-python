"""Точка входа: разбор аргументов, рендер, сохранение PNG, окно, тайминг.

Коды возврата:
  0 — успех;
  1 — ошибка формата/чтения файла (в stderr "Error: <причина>");
  2 — неверные аргументы CLI (печатается справка).
"""

from __future__ import annotations

import argparse
import sys
import time

from minirt.errors import RtError
from minirt.parser import parse_scene_file
from minirt.render import render


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="miniRT — трассировщик лучей на Python + NumPy",
    )
    p.add_argument("scene", help="путь к файлу сцены .rt")
    p.add_argument("--width", type=int, default=800, help="ширина изображения (по умолчанию 800)")
    p.add_argument("--height", type=int, default=600, help="высота изображения (по умолчанию 600)")
    p.add_argument("--out", help="сохранить результат в PNG")
    p.add_argument("--no-window", action="store_true", help="не открывать окно")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse уже напечатал справку
        return 2 if exc.code != 0 else 0

    if args.width <= 0 or args.height <= 0:
        parser.print_help(sys.stderr)
        return 2

    try:
        scene = parse_scene_file(args.scene)
    except RtError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    img = render(scene, args.width, args.height)
    elapsed = time.perf_counter() - t0
    print(f"render: {elapsed:.3f} s ({args.width}x{args.height})", file=sys.stderr)

    if args.out:
        try:
            from PIL import Image  # необязательная зависимость только для PNG
        except ImportError:
            print("Error: для --out требуется Pillow", file=sys.stderr)
            return 1
        Image.fromarray(img).save(args.out)

    if not args.no_window:
        from minirt.window import show_image

        show_image(img)

    return 0


if __name__ == "__main__":
    sys.exit(main())

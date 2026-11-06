"""Точка входа: пока только парсинг сцены и печать «OK»."""

from __future__ import annotations

import argparse
import sys

from minirt.errors import RtError
from minirt.parser import parse_scene_file
from minirt.render import render


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="main.py")
    p.add_argument("scene")
    p.add_argument("--width", type=int, default=800)
    p.add_argument("--height", type=int, default=600)
    p.add_argument("--out")
    p.add_argument("--no-window", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 2 if exc.code != 0 else 0
    try:
        scene = parse_scene_file(args.scene)
        img = render(scene, args.width, args.height)
        if args.out:
            from PIL import Image

            Image.fromarray(img).save(args.out)

    except (RtError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"OK: {len(scene.objects)} objects")
    return 0


if __name__ == "__main__":
    sys.exit(main())

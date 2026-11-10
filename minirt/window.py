"""Единственный модуль, импортирующий pygame.

Показывает готовое изображение (H, W, 3) uint8 в окне.
Esc и закрытие окна завершают программу. Тестами не покрывается.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def show_image(img: NDArray[np.uint8]) -> None:
    """Открыть окно и ждать Esc/закрытия. img — (H, W, 3) uint8."""
    import pygame  # локальный импорт: библиотека нужна только здесь

    pygame.init()
    h, w = img.shape[:2]
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("miniRT")
    # pygame ожидает (W, H, 3), поэтому транспонируем оси
    surface = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
    screen.blit(surface, (0, 0))
    pygame.display.flip()

    running = True
    while running:
        for event in pygame.event.get():
            is_quit = event.type == pygame.QUIT
            is_esc = event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            if is_quit or is_esc:
                running = False
    pygame.quit()

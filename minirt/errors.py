"""Иерархия ошибок проекта.

Все собственные ошибки наследуются от RtError, чтобы точка входа
могла ловить их единым except RtError и печатать "Error: <причина>".
"""


class RtError(Exception):
    """Базовый класс всех ошибок трассировщика."""


class SceneFormatError(RtError):
    """Ошибка формата файла .rt с указанием номера строки.

    Пример сообщения: "строка 7: у элемента sp ожидается 3 полей, получено 2".
    """

    def __init__(self, line_no: int, message: str) -> None:
        self.line_no = line_no
        self.message = message
        super().__init__(f"строка {line_no}: {message}")

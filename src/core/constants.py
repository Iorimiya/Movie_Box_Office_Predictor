from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class Constants:
    # save file setting
    DEFAULT_SAVE_FILE_EXTENSION: Final[str] = 'yaml'
    DEFAULT_ENCODING: Final[str] = 'utf-8'

    # status bar setting
    STATUS_BAR_FORMAT: Final[str] = '{desc}: {percentage:3.2f}%|{bar}{r_bar}'

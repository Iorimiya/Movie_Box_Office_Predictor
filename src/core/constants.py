from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class Constants:
    """
    A frozen dataclass that centralizes application-wide constants.

    This class provides a single, immutable source for various static values
    used throughout the application, such as file settings and UI formats.

    :ivar DEFAULT_SAVE_FILE_EXTENSION: The default file extension for saved files (e.g., 'yaml').
    :ivar DEFAULT_ENCODING: The default character encoding for file I/O (e.g., 'utf-8').
    :ivar STATUS_BAR_FORMAT: The format string for progress bars, compatible with libraries like tqdm.
    """
    # save file setting
    DEFAULT_SAVE_FILE_EXTENSION: Final[str] = 'yaml'
    DEFAULT_ENCODING: Final[str] = 'utf-8'

    # status bar setting
    STATUS_BAR_FORMAT: Final[str] = '{desc}: {percentage:3.2f}%|{bar}{r_bar}'

from dataclasses import dataclass
from pathlib import Path
from typing import Final

@dataclass(frozen=True)
class Constants:
    STATUS_BAR_FORMAT: Final[str] = '{desc}: {percentage:3.2f}%|{bar}{r_bar}'
    INDEX_HEADER: Final[tuple[str]] = ('id', 'name')
    INPUT_MOVIE_LIST_HEADER: Final[str] = 'movie_name'
    BOX_OFFICE_DOWNLOAD_PROGRESS_HEADER: Final[tuple[str]] = ('id', 'movie_page_url', 'file_path')
    PROJECT_FOLDER: Final[Path] = Path(__file__).parent.parent
    DATA_FOLDER: Final[Path] = PROJECT_FOLDER.joinpath('data')
    BOX_OFFICE_FOLDER: Final[Path] = DATA_FOLDER.joinpath('web_scraping_data', 'box_office')
    PUBLIC_REVIEW_FOLDER: Final[Path] = DATA_FOLDER.joinpath('web_scraping_data', 'public_review')
    EXPERT_REVIEW_FOLDER: Final[Path] = DATA_FOLDER.joinpath('web_scraping_data', 'expert_review')
    INDEX_PATH: Final[Path] = DATA_FOLDER.joinpath('index.csv')
    DEFAULT_SAVE_FILE_EXTENSION: Final[str] = 'yaml'
    DEFAULT_ENCODING: str = 'utf-8'
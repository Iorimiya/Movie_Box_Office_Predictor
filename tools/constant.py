from dataclasses import dataclass
from pathlib import Path
from typing import Final

@dataclass(frozen=True)
class Constants:
    # Folder Path
    PROJECT_FOLDER: Final[Path] = Path(__file__).parent.parent
    DATA_FOLDER: Final[Path] = PROJECT_FOLDER.joinpath('data')
    BOX_OFFICE_FOLDER: Final[Path] = DATA_FOLDER.joinpath('web_scraping_data', 'box_office')
    PUBLIC_REVIEW_FOLDER: Final[Path] = DATA_FOLDER.joinpath('web_scraping_data', 'public_review')
    EXPERT_REVIEW_FOLDER: Final[Path] = DATA_FOLDER.joinpath('web_scraping_data', 'expert_review')
    EMOTION_ANALYSER_MODEL_FOLDER: Final[Path] = DATA_FOLDER.joinpath('emotion_analysis', 'model')
    EMOTION_ANALYSER_TOKENIZER_FOLDER: Final[Path] = DATA_FOLDER.joinpath('emotion_analysis', 'dataset')

    # File Path
    INDEX_PATH: Final[Path] = DATA_FOLDER.joinpath('index.csv')
    EMOTION_ANALYSER_MODEL_PATH:  Final[Path] = EMOTION_ANALYSER_MODEL_FOLDER.joinpath('emotion_analysis_model_1000.keras')
    EMOTION_ANALYSER_TOKENIZER_PATH: Final[Path] = EMOTION_ANALYSER_TOKENIZER_FOLDER.joinpath('tokenizer.pickle')

    # Header
    INPUT_MOVIE_LIST_HEADER: Final[str] = 'movie_name'
    INDEX_HEADER: Final[tuple[str]] = ('id', 'name')
    BOX_OFFICE_DOWNLOAD_PROGRESS_HEADER: Final[tuple[str]] = ('id', 'movie_page_url', 'file_path')

    # Save
    DEFAULT_SAVE_FILE_EXTENSION: Final[str] = 'yaml'
    DEFAULT_ENCODING: Final[str] = 'utf-8'

    # other
    STATUS_BAR_FORMAT: Final[str] = '{desc}: {percentage:3.2f}%|{bar}{r_bar}'

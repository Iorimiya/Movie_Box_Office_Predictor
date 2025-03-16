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
    REVIEW_SENTIMENT_ANALYSIS_MODEL_FOLDER: Final[Path] = DATA_FOLDER.joinpath('review_sentiment_analysis', 'model')
    REVIEW_SENTIMENT_ANALYSIS_DATASET_FOLDER: Final[Path] = DATA_FOLDER.joinpath('review_sentiment_analysis', 'dataset')
    BOX_OFFICE_PREDICTION_FOLDER: Final[Path] = DATA_FOLDER.joinpath('box_office_prediction')
    BOX_OFFICE_PREDICTION_MODEL_FOLDER: Final[Path] = BOX_OFFICE_PREDICTION_FOLDER.joinpath('model')


    # Defaults Model Name
    REVIEW_SENTIMENT_ANALYSIS_MODEL_NAME:Final[str] = 'review_sentiment_analysis_model'
    BOX_OFFICE_PREDICTION_MODEL_NAME:Final[str] = 'box_office_prediction_model'

    # File Path
    INDEX_PATH: Final[Path] = DATA_FOLDER.joinpath('index.csv')
    REVIEW_SENTIMENT_ANALYSIS_MODEL_PATH: Final[Path] = REVIEW_SENTIMENT_ANALYSIS_MODEL_FOLDER.joinpath(
        f'{REVIEW_SENTIMENT_ANALYSIS_MODEL_NAME}_1000.keras')
    REVIEW_SENTIMENT_ANALYSIS_TOKENIZER_PATH: Final[Path] = REVIEW_SENTIMENT_ANALYSIS_DATASET_FOLDER.joinpath(
        'tokenizer.pickle')
    BOX_OFFICE_PREDICTION_MODEL_PATH: Final[Path] = BOX_OFFICE_PREDICTION_MODEL_FOLDER.joinpath(
        f'{BOX_OFFICE_PREDICTION_MODEL_NAME}_1000.keras')
    BOX_OFFICE_PREDICTION_SETTING_PATH: Final[Path] = BOX_OFFICE_PREDICTION_FOLDER.joinpath('setting.yaml')
    BOX_OFFICE_PREDICTION_SCALER_PATH: Final[Path] = BOX_OFFICE_PREDICTION_FOLDER.joinpath('scaler.gz')

    # Header
    INPUT_MOVIE_LIST_HEADER: Final[str] = 'movie_name'
    INDEX_HEADER: Final[tuple[str]] = ('id', 'name')
    BOX_OFFICE_DOWNLOAD_PROGRESS_HEADER: Final[tuple[str]] = ('id', 'movie_page_url', 'file_path')

    # Save
    DEFAULT_SAVE_FILE_EXTENSION: Final[str] = 'yaml'
    DEFAULT_ENCODING: Final[str] = 'utf-8'

    # other
    STATUS_BAR_FORMAT: Final[str] = '{desc}: {percentage:3.2f}%|{bar}{r_bar}'

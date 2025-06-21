from dataclasses import dataclass
from pathlib import Path
from typing import Final


@dataclass(frozen=True)
class Constants:
    # Defaults Model Name
    REVIEW_SENTIMENT_ANALYSIS_MODEL_NAME: Final[str] = 'review_sentiment_analysis_model'
    BOX_OFFICE_PREDICTION_MODEL_NAME: Final[str] = 'box_office_prediction_model'
    BOX_OFFICE_PREDICTION_MODEL_DEFAULT_EPOCH: Final[int] = 123

    # Folder Path
    PROJECT_FOLDER: Final[Path] = Path(__file__).parent.parent
    DATA_FOLDER: Final[Path] = PROJECT_FOLDER.joinpath('data')
    SCRAPING_DATA_FOLDER: Final[Path] = DATA_FOLDER.joinpath('web_scraping_data')
    PUBLIC_REVIEW_FOLDER: Final[Path] = SCRAPING_DATA_FOLDER.joinpath('public_review')
    EXPERT_REVIEW_FOLDER: Final[Path] = SCRAPING_DATA_FOLDER.joinpath('expert_review')
    REVIEW_SENTIMENT_ANALYSIS_MODEL_FOLDER: Final[Path] = DATA_FOLDER.joinpath('review_sentiment_analysis', 'model')
    REVIEW_SENTIMENT_ANALYSIS_DATASET_FOLDER: Final[Path] = DATA_FOLDER.joinpath('review_sentiment_analysis', 'dataset')
    BOX_OFFICE_PREDICTION_FOLDER: Final[Path] = DATA_FOLDER.joinpath('box_office_prediction')
    BOX_OFFICE_PREDICTION_DEFAULT_MODEL_FOLDER: Final[Path] = BOX_OFFICE_PREDICTION_FOLDER.joinpath(
        f'{BOX_OFFICE_PREDICTION_MODEL_NAME}_{BOX_OFFICE_PREDICTION_MODEL_DEFAULT_EPOCH}')

    # File Path
    INDEX_PATH: Final[Path] = DATA_FOLDER.joinpath('index.csv')
    REVIEW_SENTIMENT_ANALYSIS_MODEL_PATH: Final[Path] = REVIEW_SENTIMENT_ANALYSIS_MODEL_FOLDER.joinpath(
        f'{REVIEW_SENTIMENT_ANALYSIS_MODEL_NAME}_1000.keras')
    REVIEW_SENTIMENT_ANALYSIS_TOKENIZER_PATH: Final[Path] = REVIEW_SENTIMENT_ANALYSIS_DATASET_FOLDER.joinpath(
        'tokenizer.pickle')
    BOX_OFFICE_PREDICTION_MODEL_PATH: Final[Path] = BOX_OFFICE_PREDICTION_DEFAULT_MODEL_FOLDER.joinpath(
        f'{BOX_OFFICE_PREDICTION_MODEL_NAME}_{BOX_OFFICE_PREDICTION_MODEL_DEFAULT_EPOCH}.keras')
    BOX_OFFICE_PREDICTION_SETTING_PATH: Final[Path] = BOX_OFFICE_PREDICTION_DEFAULT_MODEL_FOLDER.joinpath(
        'setting.yaml')
    BOX_OFFICE_PREDICTION_SCALER_PATH: Final[Path] = BOX_OFFICE_PREDICTION_DEFAULT_MODEL_FOLDER.joinpath('scaler.gz')

    # Save
    DEFAULT_SAVE_FILE_EXTENSION: Final[str] = 'yaml'
    DEFAULT_ENCODING: Final[str] = 'utf-8'

    # other
    STATUS_BAR_FORMAT: Final[str] = '{desc}: {percentage:3.2f}%|{bar}{r_bar}'

# TODO:replace all path constants with ProjectConfig

from dataclasses import dataclass
from pathlib import Path
from typing import  Optional, TypedDict

from numpy import int32, int64
from numpy.typing import NDArray
from pandas import DataFrame
from typing_extensions import override

from src.models.base.base_data_processor import BaseDataProcessor


@dataclass(frozen=True)
class SentimentDataSource:
    """
    A data source configuration for the sentiment model.

    :ivar file_name: The name of the source CSV file located in the
                     `inputs/sentiment_analysis_resources` directory.
    """
    file_name: str


@dataclass(frozen=True)
class SentimentTrainingConfig:
    """
    Configuration for processing data for sentiment model training.

    :ivar vocabulary_size: The maximum number of words to keep, based on word frequency.
    :ivar split_ratios: The ratio for splitting data into train, validation, and test sets.
    :ivar random_state: The seed used by the random number generator for data splitting.
    """
    vocabulary_size: int
    split_ratios: tuple[int, int, int]
    random_state: int


class ProcessedSentimentData(TypedDict):
    """
    A TypedDict representing the processed and split dataset for sentiment analysis.

    :ivar x_train: Training data (features).
    :ivar y_train: Training data (labels).
    :ivar x_val: Validation data (features).
    :ivar y_val: Validation data (labels).
    :ivar x_test: Test data (features).
    :ivar y_test: Test data (labels).
    """
    x_train: NDArray[int32]
    y_train: NDArray[int64]
    x_val: NDArray[int32]
    y_val: NDArray[int64]
    x_test: NDArray[int32]
    y_test: NDArray[int64]


class SentimentDataProcessor(
    BaseDataProcessor[
        SentimentDataSource,
        DataFrame,
        ProcessedSentimentData,
        str,
        NDArray[int32],
        SentimentTrainingConfig
    ]
):
    """
    Handles all data-related tasks for the review sentiment analysis model.

    This processor loads raw positive/negative words from a CSV, constructs
    sample sentences, tokenizes them using Jieba and Keras Tokenizer, and
    prepares training/validation/testing splits. It manages the Keras Tokenizer as its
    primary artifact.
    """


    @override
    def __init__(self, model_artifacts_path: Optional[Path] = None):
        """
        Initializes the SentimentDataProcessor.

        :param model_artifacts_path: Path to the directory for model artifacts.
        """

    @override
    def save_artifacts(self) -> None:
        """
        Saves the Tokenizer and max_sequence_length artifacts to a file named 'tokenizer.pickle'.

        :raises ValueError: If `model_artifacts_path` is not set or the tokenizer is not available.
        """

    @override
    def load_artifacts(self) -> None:
        """
        Loads the Tokenizer and max_sequence_length artifacts from 'tokenizer.pickle' if it exists.
        """


    @override
    def load_raw_data(self, source: SentimentDataSource) -> DataFrame:
        """
        Loads training data from a CSV file.

        The path is constructed using `ProjectPaths`.

        :param source: The data source object containing the file name.
        :returns: The loaded training data.
        """


    @override
    def process_for_training(self, raw_data: DataFrame, config: SentimentTrainingConfig) -> ProcessedSentimentData:
        """
        Prepares raw data for training a sentiment analysis model.

        This involves constructing sample sentences, performing word segmentation,
        initializing and fitting a Tokenizer, converting texts to padded sequences,
        and splitting the data into training, validation, and testing sets.

        :param raw_data: Input data with 'is_positive' and 'word' columns.
        :param config: The configuration object with training parameters.
        :returns: A dictionary containing all data splits.
        :raises ValueError: If the sum of `split_ratios` is zero.
        """


    @override
    def process_for_prediction(self, single_input: str) -> NDArray[int32]:
        """
        Processes a single text input for sentiment prediction.

        :param single_input: The text string to predict sentiment for.
        :returns: A processed and padded sequence ready for the model.
        :raises ValueError: If the tokenizer has not been fitted or `max_sequence_length` is not provided.
        """


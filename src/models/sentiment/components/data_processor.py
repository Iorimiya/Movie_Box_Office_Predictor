from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Final, Optional, TypedDict

import jieba
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from numpy import int32, int64
from numpy.typing import NDArray
from pandas import DataFrame, Series
from typing_extensions import override

from src.core.logging_manager import LoggingManager
from src.core.project_config import ProjectPaths
from src.data_handling.file_io import CsvFile, PickleFile
from src.models.base.base_data_processor import BaseDataProcessor
from src.models.base.data_splitter import DatasetSplitter, SplitDataset


@dataclass(frozen=True)
class SentimentDataSource:
    """
    A data source configuration for the sentiment model.

    :ivar file_name: The name of the source CSV file located in the
                     `inputs/sentiment_analysis_resources` directory.
    """
    file_name: str


SentimentTrainingRawData: type = DataFrame


class SentimentTrainingProcessedData(TypedDict):
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


SentimentPredictionRawData: type = str
SentimentPredictionProcessedData: type = NDArray[int32]


@dataclass(frozen=True)
class SentimentDataConfig:
    """
    Configuration for processing data for sentiment model training.

    :ivar vocabulary_size: The maximum number of words to keep, based on word frequency.
    :ivar split_ratios: The ratio for splitting data into train, validation, and test sets.
    :ivar random_state: The seed used by the random number generator for data splitting.
    """
    vocabulary_size: int
    split_ratios: tuple[int, int, int]
    random_state: int


class SentimentDataProcessor(
    BaseDataProcessor[
        SentimentDataSource,
        SentimentTrainingRawData,
        SentimentTrainingProcessedData,
        SentimentPredictionRawData,
        SentimentPredictionProcessedData,
        SentimentDataConfig
    ]
):
    """
    Handles all data-related tasks for the review sentiment analysis model.

    This processor loads raw positive/negative words from a CSV, constructs
    sample sentences, tokenizes them using Jieba and Keras Tokenizer, and
    prepares training/validation/testing splits. It manages the Keras Tokenizer as its
    primary artifact.
    """

    ARTIFACTS_FILE_NAME: Final[str] = "artifacts.pickle"

    @override
    def __init__(self, model_artifacts_path: Optional[Path] = None):
        """
        Initializes the SentimentDataProcessor.

        :param model_artifacts_path: Path to the directory for model artifacts.
        """
        self.tokenizer: Optional[Tokenizer] = None
        self.max_sequence_length: Optional[int] = None
        self.logger: Logger = LoggingManager().get_logger('machine_learning')
        self.splitter: DatasetSplitter[NDArray[int32], NDArray[int64]] = \
            DatasetSplitter[NDArray[int32], NDArray[int64]]()
        super().__init__(model_artifacts_path=model_artifacts_path)

    @override
    def save_artifacts(self) -> None:
        """
        Saves the Tokenizer and max_sequence_length artifacts to a file named 'tokenizer.pickle'.

        :raises ValueError: If `model_artifacts_path` is not set or the tokenizer is not available.
        """
        if not self.model_artifacts_path:
            raise ValueError("model_artifacts_path is not set. Cannot save tokenizer.")
        if not self.tokenizer or self.max_sequence_length is None:
            raise ValueError("Tokenizer or max_sequence_length is not available to be saved.")

        self.model_artifacts_path.mkdir(parents=True, exist_ok=True)
        artifact_path: Path = self.model_artifacts_path / self.ARTIFACTS_FILE_NAME
        self.logger.info(f"Saving Tokenizer and max_sequence_length artifact to: {artifact_path}")
        pickle_file: PickleFile = PickleFile(path=artifact_path)
        pickle_file.save(data={
            'tokenizer': self.tokenizer,
            'max_sequence_length': self.max_sequence_length
        })

    @override
    def load_artifacts(self) -> None:
        """
        Loads the Tokenizer and max_sequence_length artifacts from 'tokenizer.pickle' if it exists.
        """
        if not self.model_artifacts_path:
            return

        tokenizer_path: Path = self.model_artifacts_path / self.ARTIFACTS_FILE_NAME
        if tokenizer_path.exists():
            self.logger.info(f"Loading Tokenizer and max_sequence_length artifact from: {tokenizer_path}")
            pickle_file: PickleFile = PickleFile(path=tokenizer_path)
            try:
                loaded_dict: dict = pickle_file.load()
                self.tokenizer = loaded_dict.get('tokenizer')
                self.max_sequence_length = loaded_dict.get('max_sequence_length')
                self.logger.info(f"Loaded max_sequence_length: {self.max_sequence_length}")
            except (TypeError, ValueError):
                self.logger.warning("Could not unpack two values from pickle.")
                self.tokenizer = None
                self.max_sequence_length = None

    @override
    def load_raw_data(self, source: SentimentDataSource) -> SentimentTrainingRawData:
        """
        Loads training data from a CSV file.

        The path is constructed using `ProjectPaths`.

        :param source: The data source object containing the file name.
        :returns: The loaded training data.
        """
        file_path: Path = ProjectPaths.sentiment_analysis_resources_dir / source.file_name
        self.logger.info(f"Loading raw sentiment data from: {file_path}")
        csv_file: CsvFile = CsvFile(path=file_path)
        return SentimentTrainingRawData(csv_file.load())

    @override
    def process_for_training(self, raw_data: SentimentTrainingRawData,
                             config: SentimentDataConfig) -> SentimentTrainingProcessedData:
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
        # Step 1 & 2: Construct and segment texts
        segmented_texts, labels = self._construct_and_segment_texts(raw_data=raw_data)

        # Step 3 & 4: Tokenize and pad sequences
        x_data, y_data = self._tokenize_and_pad_sequences(
            segmented_texts=segmented_texts,
            vocabulary_size=config.vocabulary_size,
            labels=labels
        )

        # Step 5: Split data
        processed_data: SplitDataset[NDArray[int32], NDArray[int64]] = self.splitter.split(
            x_data=x_data,
            y_data=y_data,
            split_ratios=config.split_ratios,
            random_state=config.random_state,
            shuffle=True
        )

        return SentimentTrainingProcessedData(**processed_data)

    @override
    def process_for_prediction(self, single_input: SentimentPredictionRawData, config: Optional[SentimentDataConfig] = None) -> NDArray[int32]:
        """
        Processes a single text input for sentiment prediction.

        :param single_input: The text string to predict sentiment for.
        :param config: This parameter is ignored by this implementation.
        :returns: A processed and padded sequence ready for the model.
        :raises ValueError: If the tokenizer has not been fitted or `max_sequence_length` is not provided.
        """
        if not self.tokenizer or self.max_sequence_length is None:
            raise ValueError(
                "Processor has not been trained or artifacts are not loaded. "
                "Call train or load_artifacts first."
            )

        segmented_text: str = " ".join(jieba.lcut(single_input))
        sequence: list[list[int]] = self.tokenizer.texts_to_sequences(texts=[segmented_text])
        padded_sequence: NDArray[int32] = pad_sequences(
            sequences=sequence, maxlen=self.max_sequence_length
        )
        return padded_sequence

    def _construct_and_segment_texts(self, raw_data: SentimentTrainingRawData) -> tuple[list[str], list[int]]:
        """
        Constructs sample sentences and performs word segmentation.

        :param raw_data: The raw data containing positive and negative words.
        :returns: A tuple containing the list of segmented texts and their corresponding labels.
        """
        self.logger.info("Constructing sample sentences and performing word segmentation...")
        positive_words: Series = raw_data[raw_data['is_positive']].dropna().loc[:, 'word']
        negative_words: Series = raw_data[~raw_data['is_positive']].dropna().loc[:, 'word']

        positive_samples: list[str] = [f"這是一個非常{word}的電影，值得推薦！" for word in positive_words]
        negative_samples: list[str] = [f"這是一個非常{word}的電影，完全不推薦！" for word in negative_words]

        texts: list[str] = positive_samples + negative_samples
        labels: list[int] = [1] * len(positive_samples) + [0] * len(negative_samples)

        segmented_texts: list[str] = [" ".join(jieba.lcut(text)) for text in texts]
        return segmented_texts, labels

    def _tokenize_and_pad_sequences(
        self, segmented_texts: list[str], labels: list[int], vocabulary_size: int
    ) -> tuple[NDArray[int32], NDArray[int64]]:
        """
        Initializes, fits a tokenizer, and converts texts to padded sequences.

        This method updates `self.tokenizer` and `self.max_sequence_length`.

        :param segmented_texts: A list of space-separated segmented texts.
        :param vocabulary_size: The maximum number of words for the tokenizer.
        :returns: A tuple containing the padded feature data (x) and label data (y).
        """
        self.logger.info("Tokenizing texts and padding sequences...")
        self.tokenizer = Tokenizer(num_words=vocabulary_size)
        self.tokenizer.fit_on_texts(texts=segmented_texts)

        sequences: list[list[int]] = self.tokenizer.texts_to_sequences(texts=segmented_texts)
        self.max_sequence_length = max(len(s) for s in sequences) if sequences else 0
        x_data: NDArray[int32] = pad_sequences(sequences=sequences, maxlen=self.max_sequence_length)
        y_data: NDArray[int64] = np.array(labels)

        return x_data, y_data

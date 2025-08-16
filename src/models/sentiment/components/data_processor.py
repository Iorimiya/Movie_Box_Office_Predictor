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
from pandas import DataFrame
from typing_extensions import override

from src.core.logging_manager import LoggingManager
from src.core.project_config import ProjectPaths
from src.data_handling.file_io import CsvFile, PickleFile
from src.models.base.data_splitter import DatasetSplitter, SplitDataset
from src.models.base.evaluable_data_processor import EvaluableDataProcessor


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
    EvaluableDataProcessor[
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
        csv_raw_data: list[dict[str, any]] = csv_file.load()
        raw_dataframe: DataFrame = SentimentTrainingRawData(csv_raw_data)

        if 'is_positive' in raw_dataframe.columns:
            self.logger.debug("Converting 'is_positive' column to boolean type.")
            # This handles strings like 'True', 'False', 'true', 'false'
            raw_dataframe['is_positive'] = raw_dataframe['is_positive'].astype(bool)
        else:
            # This is a critical data error, so we should raise it.
            raise KeyError("The required column 'is_positive' was not found in the loaded data.")

        return raw_dataframe

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

        x_text: NDArray[object] = np.array(segmented_texts, dtype=object)
        y_data: NDArray[int64] = np.array(labels)

        self.logger.info("Splitting raw text data before tokenization...")
        text_splitter: DatasetSplitter[NDArray[object], NDArray[int64]] = DatasetSplitter()
        text_splits: SplitDataset[NDArray[object], NDArray[int64]] = text_splitter.split(
            x_data=x_text,
            y_data=y_data,
            split_ratios=config.split_ratios,
            random_state=config.random_state,
            shuffle=True
        )

        # Step 3 & 4: Fit tokenizer on the training set and transform all sets.
        x_train_seq, x_val_seq, x_test_seq = self._fit_tokenizer_and_transform_splits(
            x_train_text=text_splits['x_train'],
            x_val_text=text_splits['x_val'],
            x_test_text=text_splits['x_test'],
            vocabulary_size=config.vocabulary_size
        )

        # Step 5: Pad all sequences to the same length.
        self.logger.info("Padding all sequence splits to a uniform length...")
        self.max_sequence_length = max(len(s) for s in x_train_seq + x_val_seq + x_test_seq) if (
                x_train_seq or x_val_seq or x_test_seq) else 0
        self.logger.info(f"Determined max_sequence_length: {self.max_sequence_length}")

        x_train_pad: NDArray[int32] = pad_sequences(sequences=x_train_seq, maxlen=self.max_sequence_length)
        x_val_pad: NDArray[int32] = pad_sequences(sequences=x_val_seq, maxlen=self.max_sequence_length)
        x_test_pad: NDArray[int32] = pad_sequences(sequences=x_test_seq, maxlen=self.max_sequence_length)

        return SentimentTrainingProcessedData(
            x_train=x_train_pad, y_train=text_splits['y_train'],
            x_val=x_val_pad, y_val=text_splits['y_val'],
            x_test=x_test_pad, y_test=text_splits['y_test']
        )

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

    def process_for_evaluation(
        self, raw_data: SentimentTrainingRawData, config: Optional[SentimentDataConfig] = None
    ) -> tuple[NDArray[int32], NDArray[int64]]:
        """
        Processes a full raw dataset for evaluation without splitting it.

        This method is designed for evaluating a model on a new, unseen dataset
        where the entire dataset should be treated as a single test set. It
        performs all processing steps (segmentation, tokenization, padding)
        except for the train/val/test split.

        :param raw_data: The raw DataFrame containing the data.
        :param config: This parameter is ignored by this implementation.
        :returns: A tuple containing the full processed features (x) and labels (y).
        :raises ValueError: If the tokenizer is not loaded.
        """
        self.logger.info("Processing full dataset for evaluation (no splitting).")
        if not self.tokenizer or self.max_sequence_length is None:
            raise ValueError("Tokenizer must be loaded to process data for evaluation.")

        segmented_texts, labels = self._construct_and_segment_texts(raw_data=raw_data)

        # Use the pre-loaded tokenizer, do not re-fit
        sequences: list[list[int]] = self.tokenizer.texts_to_sequences(texts=segmented_texts)
        x_data: NDArray[int32] = pad_sequences(sequences=sequences, maxlen=self.max_sequence_length)
        y_data: NDArray[int64] = np.array(labels)

        return x_data, y_data

    def _construct_and_segment_texts(self, raw_data: SentimentTrainingRawData) -> tuple[list[str], list[int]]:
        """
        Constructs sample sentences and performs word segmentation.

        :param raw_data: The raw data containing positive and negative words.
        :returns: A tuple containing the list of segmented texts and their corresponding labels.
        """
        self.logger.info("Extracting sentences and performing word segmentation...")

        clean_data: DataFrame = raw_data.dropna(subset=['word', 'is_positive'])
        texts: list[str] = clean_data['word'].tolist()
        labels: list[int] = clean_data['is_positive'].astype(int).tolist()
        segmented_texts: list[str] = [" ".join(jieba.lcut(text)) for text in texts]

        self.logger.info(f"Processed {len(segmented_texts)} sentences for training.")
        return segmented_texts, labels

    def _fit_tokenizer_and_transform_splits(
        self,
        x_train_text: NDArray[object],
        x_val_text: NDArray[object],
        x_test_text: NDArray[object],
        vocabulary_size: int
    ) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
        """
        Fits the tokenizer on the training text and then transforms all data splits.

        :param x_train_text: The training text data.
        :param x_val_text: The validation text data.
        :param x_test_text: The test text data.
        :param vocabulary_size: The maximum number of words for the tokenizer.
        :returns: A tuple of tokenized (but not padded) sequences for train, val, and test sets.
        """
        self.logger.info("Fitting tokenizer ONLY on the training data...")
        self.tokenizer = Tokenizer(num_words=vocabulary_size)
        self.tokenizer.fit_on_texts(texts=x_train_text)

        self.logger.info("Transforming all text splits to sequences using the fitted tokenizer...")
        x_train_seq: list[list[int]] = self.tokenizer.texts_to_sequences(texts=x_train_text)
        x_val_seq: list[list[int]] = self.tokenizer.texts_to_sequences(texts=x_val_text)
        x_test_seq: list[list[int]] = self.tokenizer.texts_to_sequences(texts=x_test_text)

        return x_train_seq, x_val_seq, x_test_seq

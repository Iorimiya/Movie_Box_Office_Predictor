from dataclasses import dataclass
from pathlib import Path
from typing import Final, Optional

import jieba
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from numpy import int32, int64
from numpy.typing import NDArray
from pandas import DataFrame
from typing_extensions import override

from src.core.project_config import ProjectPaths
from src.data_handling.file_io import CsvFile, PickleFile
from src.models.base.base_data_processor import BaseDataConfig
from src.models.base.data_splitter import SplitDataset
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
SentimentTrainingProcessedData: type = SplitDataset[NDArray[int32], NDArray[int64]]
SentimentPredictionRawData: type = str
SentimentPredictionProcessedData: type = NDArray[int32]


@dataclass(frozen=True)
class SentimentDataConfig(BaseDataConfig):
    """
    Configuration for processing data for sentiment model training.

    Inherits common splitting parameters from BaseDataConfig.

    :ivar vocabulary_size: The maximum number of words to keep, based on word frequency.
    """
    # The common fields are now inherited. We only need to define the unique ones.
    vocabulary_size: int


class SentimentDataProcessor(
    EvaluableDataProcessor[
        SentimentDataSource,
        SentimentTrainingRawData,
        SentimentTrainingProcessedData,
        SentimentPredictionRawData,
        SentimentPredictionProcessedData,
        SentimentDataConfig,
        NDArray[int32],
        NDArray[int64]
    ]
):
    """
    Handles all data-related tasks for the review sentiment analysis model.

    This processor loads raw text data, tokenizes it using Jieba and Keras Tokenizer,
    and prepares training, validation, and testing splits. It manages the Keras
    Tokenizer and the maximum sequence length as its primary artifacts, which can be
    saved and loaded.

    :ivar tokenizer: The Keras Tokenizer instance, fitted on the training data.
                     It is `None` until the processor is trained or artifacts are loaded.
    :ivar max_sequence_length: The length to which all sequences are padded.
                               It is `None` until the processor is trained or artifacts are loaded.
    """

    ARTIFACTS_FILE_NAME: Final[str] = "artifacts.pickle"

    @override
    def __init__(self, model_artifacts_path: Optional[Path] = None):
        """
        Initializes the SentimentDataProcessor.

        :param model_artifacts_path: Path to the directory for model artifacts.
        """
        super().__init__(model_artifacts_path=model_artifacts_path)
        self.tokenizer: Optional[Tokenizer] = None
        self.max_sequence_length: Optional[int] = None
        self.load_artifacts()

    @override
    def save_artifacts(self) -> None:
        """
        Saves the tokenizer and max_sequence_length to a pickle file.

        The artifacts are saved as a dictionary to a single file within the
        `model_artifacts_path` directory.

        :raises ValueError: If `model_artifacts_path` is not set, or if the
                            tokenizer or `max_sequence_length` has not been set.
        """
        if not self.model_artifacts_path:
            raise ValueError("model_artifacts_path is not set. Cannot save tokenizer.")
        if not self.tokenizer or self.max_sequence_length is None:
            raise ValueError("Tokenizer or max_sequence_length is not available to be saved.")

        self.model_artifacts_path.mkdir(parents=True, exist_ok=True)
        artifact_path: Path = self.model_artifacts_path / self.ARTIFACTS_FILE_NAME
        self.logger.info(f"Saving Tokenizer and max_sequence_length artifact to: {artifact_path}")
        PickleFile(path=artifact_path).save(data={
            'tokenizer': self.tokenizer,
            'max_sequence_length': self.max_sequence_length
        })

    @override
    def load_artifacts(self) -> None:
        """
        Loads the tokenizer and max_sequence_length from the artifacts file.

        If `model_artifacts_path` is set and the artifact file exists, this method
        populates `self.tokenizer` and `self.max_sequence_length`. It handles
        cases where the file is not found or is corrupted.
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
        Loads and pre-processes raw sentiment data from a source CSV file.

        The method reads the specified CSV file, converts it into a pandas DataFrame,
        and ensures the 'is_positive' column exists and is of boolean type.

        :param source: The data source object specifying the file to load.
        :returns: A DataFrame containing the raw sentiment data.
        :raises KeyError: If the 'is_positive' column is not found in the CSV file.
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
    def process_for_prediction(self, single_input: SentimentPredictionRawData,
                               config: Optional[SentimentDataConfig] = None) -> NDArray[int32]:
        """
        Processes a single text input for sentiment prediction.

        This method segments the input text, converts it to a sequence of integers
        using the pre-trained tokenizer, and pads it to the required length.

        :param single_input: The text string to process for prediction.
        :param config: This parameter is ignored by this implementation.
        :returns: A processed and padded sequence ready for the model.
        :raises ValueError: If the tokenizer or `max_sequence_length` has not been
                            loaded or trained.
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

    @override
    def _prepare_for_split(self, raw_data: SentimentTrainingRawData, config: SentimentDataConfig) -> tuple[
        NDArray[object], NDArray[int64]]:
        """
        Prepares raw data for the splitting process.

        This method takes the raw DataFrame, extracts and segments the texts and
        labels, and converts them into NumPy arrays suitable for the data splitter.

        :param raw_data: The input DataFrame containing text and label columns.
        :param config: The data processing configuration (not used in this method).
        :returns: A tuple of NumPy arrays: one for the segmented texts (features)
                  and one for the labels.
        """
        segmented_texts, labels = self._construct_and_segment_texts(raw_data=raw_data)
        x_text: NDArray[object] = np.array(segmented_texts, dtype=object)
        y_data: NDArray[int64] = np.array(labels)
        return x_text, y_data

    @override
    def _post_process_splits(self, split_data: SplitDataset[NDArray[object], NDArray[int64]],
                             config: SentimentDataConfig) -> SentimentTrainingProcessedData:
        """
        Finalizes the data processing after splitting.

        This method performs two main tasks:
        1. Fits a new Keras Tokenizer on the training text data and transforms all
           text splits (train, validation, test) into integer sequences.
        2. Determines the maximum sequence length and pads all sequences to this length.

        This method sets the `self.tokenizer` and `self.max_sequence_length` attributes.

        :param split_data: A `SplitDataset` containing the text-based data splits.
        :param config: The configuration containing parameters like `vocabulary_size`.
        :returns: A `SentimentTrainingProcessedData` object with all splits tokenized
                  and padded.
        """
        x_train_seq, x_val_seq, x_test_seq = self._fit_tokenizer_and_transform_splits(
            x_train_text=split_data['x_train'],
            x_val_text=split_data['x_val'],
            x_test_text=split_data['x_test'],
            vocabulary_size=config.vocabulary_size
        )

        self.logger.info("Padding all sequence splits to a uniform length...")
        self.max_sequence_length = max(len(s) for s in x_train_seq + x_val_seq + x_test_seq) if (
            x_train_seq or x_val_seq or x_test_seq) else 0
        self.logger.info(f"Determined max_sequence_length: {self.max_sequence_length}")

        x_train_pad: NDArray[int32] = pad_sequences(sequences=x_train_seq, maxlen=self.max_sequence_length)
        x_val_pad: NDArray[int32] = pad_sequences(sequences=x_val_seq, maxlen=self.max_sequence_length)
        x_test_pad: NDArray[int32] = pad_sequences(sequences=x_test_seq, maxlen=self.max_sequence_length)

        return SplitDataset(
            x_train=x_train_pad, y_train=split_data['y_train'],
            x_val=x_val_pad, y_val=split_data['y_val'],
            x_test=x_test_pad, y_test=split_data['y_test']
        )

    def _construct_and_segment_texts(self, raw_data: SentimentTrainingRawData) -> tuple[list[str], list[int]]:
        """
        Extracts text and labels from the raw data and performs word segmentation.

        This method cleans the input DataFrame by dropping rows with missing data in
        key columns ('word', 'is_positive'). It then extracts the text, segments it
        into words using Jieba, and pairs it with the corresponding integer label.

        :param raw_data: The raw DataFrame containing 'word' and 'is_positive' columns.
        :returns: A tuple containing a list of space-separated segmented texts and a
                  list of their corresponding integer labels.
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
        Fits a new tokenizer on the training text and transforms all data splits.

        This method initializes a new Keras Tokenizer, fits it exclusively on the
        training text data, and then uses this tokenizer to convert the train,
        validation, and test text sets into sequences of integers. The instance's
        `tokenizer` attribute is set by this method.

        :param x_train_text: The training text data.
        :param x_val_text: The validation text data.
        :param x_test_text: The test text data.
        :param vocabulary_size: The maximum number of words for the tokenizer.
        :returns: A tuple of tokenized (but not padded) sequences for the train,
                  validation, and test sets.
        """
        self.logger.info("Fitting tokenizer ONLY on the training data...")
        self.tokenizer = Tokenizer(num_words=vocabulary_size)
        self.tokenizer.fit_on_texts(texts=x_train_text)

        self.logger.info("Transforming all text splits to sequences using the fitted tokenizer...")
        x_train_seq: list[list[int]] = self.tokenizer.texts_to_sequences(texts=x_train_text)
        x_val_seq: list[list[int]] = self.tokenizer.texts_to_sequences(texts=x_val_text)
        x_test_seq: list[list[int]] = self.tokenizer.texts_to_sequences(texts=x_test_text)

        return x_train_seq, x_val_seq, x_test_seq

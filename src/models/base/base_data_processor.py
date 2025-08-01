from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Optional, TypeVar

RawDataSourceType = TypeVar('RawDataSourceType')
RawDataType = TypeVar('RawDataType')
ProcessedTrainingDataType = TypeVar('ProcessedTrainingDataType')
PredictionDataType = TypeVar('PredictionDataType')
ProcessedPredictionDataType = TypeVar('ProcessedPredictionDataType')
TrainingConfigType = TypeVar('TrainingConfigType')


class BaseDataProcessor(
    Generic[
        RawDataSourceType,
        RawDataType,
        ProcessedTrainingDataType,
        PredictionDataType,
        ProcessedPredictionDataType,
        TrainingConfigType],
    ABC
):
    """
    Abstract base class for data processors.

    Defines a common interface for loading raw data and processing it for training or prediction.
    It is agnostic to the specific preprocessing tools (like scalers or tokenizers) used by subclasses.
    Subclasses are responsible for managing the lifecycle of their own artifacts.
    """

    def __init__(self, model_artifacts_path: Optional[Path] = None) -> None:
        """
        Initializes the BaseDataProcessor.

        :param model_artifacts_path: Path to the directory where model artifacts
                                     (like a scaler or tokenizer) are or will be stored.
        """
        self.model_artifacts_path: Optional[Path] = model_artifacts_path
        self.load_artifacts()

    @abstractmethod
    def save_artifacts(self) -> None:
        """
        Saves all necessary preprocessing artifacts to files.

        Subclasses must implement this method to save their specific tools,
        such as scalers, tokenizers, or vocabulary files, to the location
        specified by `self.model_artifacts_path`.
        """
        pass

    @abstractmethod
    def load_artifacts(self) -> None:
        """
        Loads all necessary preprocessing artifacts from files.

        Subclasses must implement this method to load their specific tools.
        This method is typically called during the initialization of the processor.
        It should handle cases where artifact files do not yet exist (e.g., during a first training run).
        """
        pass

    @abstractmethod
    def load_raw_data(self, source: RawDataSourceType) -> RawDataType:
        """
        Loads raw data from a structured source object.

        Subclasses must implement this to read data from a given source,
        such as a file path or a database connection object.

        :param source: The source from which to load the data.
        :returns: The loaded raw data in its original, unprocessed format.
        """
        pass

    @abstractmethod
    def process_for_training(self, raw_data: RawDataType, config: TrainingConfigType) -> ProcessedTrainingDataType:
        """
        Processes raw data into a format suitable for model training.

        This typically involves fitting and transforming data using tools like
        scalers or tokenizers, and splitting the data into training/validation/test sets.

        :param raw_data: The raw data loaded by `load_raw_data`.
        :param config: A configuration object containing parameters for the training process.
        :returns: The processed data, ready to be fed into a model.
        """
        pass

    @abstractmethod
    def process_for_prediction(self, single_input: PredictionDataType) -> ProcessedPredictionDataType:
        """
        Processes a single input sample for prediction.

        This method should use the already-fitted artifacts (e.g., a loaded tokenizer)
        to transform a single piece of input data into a format the model can understand.
        It relies entirely on the processor's internal state (loaded artifacts) and
        does not require an external configuration object for prediction.

        :param single_input: A single raw input sample (e.g., a string of text or a dictionary of features).
        :returns: The processed sample, ready for the model's predict method.
        """
        pass

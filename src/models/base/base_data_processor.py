from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar


class BaseDataConfig:
    """
    The base class for all data processing configurations.

    Implements a manual 'frozen' mechanism to ensure immutability after initialization.

    :ivar _locked: Internal flag to indicate if the instance is locked.
    """
    _locked: bool = False

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the BaseDataConfig instance.

        Allows subclasses to pass unused keyword arguments up the chain.

        :param kwargs: Arbitrary keyword arguments to be handled by subclasses or ignored.
        """
        pass

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Sets an attribute on the instance, enforcing immutability if locked.

        :param name: The name of the attribute.
        :param value: The value to assign.
        :raises AttributeError: If the instance is locked.
        """
        if self._locked:
            raise AttributeError(f"Cannot assign to attribute '{name}'. Instance is immutable.")

        super().__setattr__(name, value)

    def _lock(self) -> None:
        """
        Locks the instance, making it immutable.
        """
        object.__setattr__(self, '_locked', True)


RawDataSourceType = TypeVar('RawDataSourceType')
TrainingRawDataType = TypeVar('TrainingRawDataType')
ProcessedTrainingDataType = TypeVar('ProcessedTrainingDataType')
PredictionRawDataType = TypeVar('PredictionRawDataType')
ProcessedPredictionDataType = TypeVar('ProcessedPredictionDataType')
DataConfigType = TypeVar('DataConfigType', bound=BaseDataConfig)


class BaseDataProcessor(
    Generic[
        RawDataSourceType,
        TrainingRawDataType,
        ProcessedTrainingDataType,
        PredictionRawDataType,
        ProcessedPredictionDataType,
        DataConfigType],
    ABC
):
    """
    Abstract base class for data processors.

    Defines a common interface for loading raw data and processing it for training or prediction.
    It is agnostic to the specific preprocessing tools (like scalers or tokenizers) used by subclasses.
    Subclasses are responsible for managing the lifecycle of their own artifacts.

    :ivar _model_artifacts_path: Path to the directory where model artifacts are stored.
    """
    _model_artifacts_path: Optional[Path]

    def __init__(self, model_artifacts_path: Optional[Path] = None) -> None:
        """
        Initializes the BaseDataProcessor.

        :param model_artifacts_path: Path to the directory where model artifacts
                                     (like a scaler or tokenizer) are or will be stored.
        """
        self._model_artifacts_path: Optional[Path] = model_artifacts_path

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
    def load_raw_data(self, source: RawDataSourceType) -> TrainingRawDataType:
        """
        Loads raw data from a structured source object.

        Subclasses must implement this to read data from a given source,
        such as a file path or a database connection object.

        :param source: The source from which to load the data.
        :return: The loaded raw data in its original, unprocessed format.
        """
        pass

    @abstractmethod
    def process_for_training(self, raw_data: TrainingRawDataType, config: DataConfigType) -> ProcessedTrainingDataType:
        """
        Processes raw data into a format suitable for model training.

        This typically involves fitting and transforming data using tools like
        scalers or tokenizers, and splitting the data into training/validation/test sets.

        :param raw_data: The raw data loaded by `load_raw_data`.
        :param config: A configuration object containing parameters for the training process.
        :return: The processed data, ready to be fed into a model.
        """
        pass

    @abstractmethod
    def process_for_prediction(
        self, single_input: PredictionRawDataType, config: Optional[DataConfigType]
    ) -> ProcessedPredictionDataType:
        """
        Processes a single input sample for prediction.

        This method should use the already-fitted artifacts (e.g., a loaded tokenizer)
        to transform a single piece of input data into a format the model can understand.
        It relies entirely on the processor's internal state (loaded artifacts) and
        may require an external configuration object for prediction.

        :param single_input: A single raw input sample (e.g., a string of text or a dictionary of features).
        :param config: An optional configuration object containing necessary parameters.
        :return: The processed sample, ready for the model's predict method.
        :raises NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

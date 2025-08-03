from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import TypedDict, Optional, Final

from numpy import float32, float64
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler
from typing_extensions import override

from src.core.logging_manager import LoggingManager
from src.data_handling.file_io import PickleFile
from src.data_handling.movie_collections import MovieData
from src.models.base.base_data_processor import BaseDataProcessor


@dataclass(frozen=True)
class PredictionDataSource:
    """
    A data source configuration for the prediction model.

    :ivar dataset_name: The name of the structured dataset to load movie data from.
    """
    dataset_name: str


PredictionRawData: type = list[MovieData]


class ProcessedPredictionData(TypedDict):
    """
    A TypedDict representing the processed and split dataset for box office prediction.

    :ivar x_train: Training data (features).
    :ivar y_train: Training data (labels).
    :ivar x_val: Validation data (features).
    :ivar y_val: Validation data (labels).
    :ivar x_test: Test data (features).
    :ivar y_test: Test data (labels).
    :ivar lengths_train: Original sequence lengths for the training set.
    :ivar lengths_val: Original sequence lengths for the validation set.
    :ivar lengths_test: Original sequence lengths for the test set.
    """
    x_train: NDArray[float32]
    y_train: NDArray[float64]
    x_val: NDArray[float32]
    y_val: NDArray[float64]
    x_test: NDArray[float32]
    y_test: NDArray[float64]
    lengths_train: list[int]
    lengths_val: list[int]
    lengths_test: list[int]


@dataclass(frozen=True)
class PredictionTrainingConfig:
    """
    Configuration for processing data for prediction model training.

    :ivar training_week_len: The number of past weeks to use as input for prediction.
    :ivar split_ratios: The ratio for splitting data into train, validation, and test sets.
    """
    training_week_len: int
    split_ratios: tuple[int, int, int]
    random_state: int


class PredictionDataProcessor(
    BaseDataProcessor[
        PredictionDataSource,
        PredictionRawData,
        ProcessedPredictionData,
        MovieData,
        NDArray[float32],
        PredictionTrainingConfig
    ]
):
    """
    Handles all data-related tasks for the box office prediction model.

    This processor loads movie data, transforms it into weekly features,
    creates sequences, scales the data using a MinMaxScaler, and splits it
    into training, validation, and test sets. It manages the MinMaxScaler
    and key training parameters as its primary artifacts.
    """

    SCALER_FILE_NAME: Final[str] = "scaler_and_settings.pickle"


    @override
    def __init__(self, model_artifacts_path: Optional[Path] = None):
        """
        Initializes the PredictionDataProcessor.

        :param model_artifacts_path: Path to the directory for model artifacts.
        """
        self.scaler: Optional[MinMaxScaler] = None
        self.training_week_len: Optional[int] = None
        self.logger: Logger = LoggingManager().get_logger('machine_learning')
        super().__init__(model_artifacts_path=model_artifacts_path)

    @override
    def save_artifacts(self) -> None:
        """
        Saves the MinMaxScaler and training parameters to a single pickle file.

        :raises ValueError: If `model_artifacts_path` is not set or artifacts are not available.
        """
        if not self.model_artifacts_path:
            raise ValueError("model_artifacts_path is not set. Cannot save artifacts.")
        if not self.scaler or self.training_week_len is None:
            raise ValueError("Scaler or training parameters are not available to be saved.")

        self.model_artifacts_path.mkdir(parents=True, exist_ok=True)
        artifact_path: Path = self.model_artifacts_path / self.SCALER_FILE_NAME
        self.logger.info(f"Saving scaler and settings artifact to: {artifact_path}")

        PickleFile(path=artifact_path).save(data={'scaler': self.scaler, 'training_week_len': self.training_week_len})


    @override
    def load_artifacts(self) -> None:
        """
        Loads the MinMaxScaler and training parameters from the artifact file.
        """
        if not self.model_artifacts_path:
            return

        artifact_path: Path = self.model_artifacts_path / self.SCALER_FILE_NAME
        if artifact_path.exists():
            self.logger.info(f"Loading scaler and settings artifact from: {artifact_path}")
            try:
                load_data: dict = PickleFile(path=artifact_path).load()
                self.scaler: MinMaxScaler = load_data.get('scaler')
                self.training_week_len = load_data.get('training_week_len')
                self.logger.info(
                    f"Loaded artifacts: training_week_limit={self.training_week_len}."
                )
            except (TypeError, ValueError) as e:
                self.logger.error(f"Failed to load artifacts from {artifact_path}: {e}", exc_info=True)
                # Reset to ensure a clean state
                self.scaler = None
                self.training_week_len = None

    @override
    def load_raw_data(self, source: PredictionDataSource) -> PredictionRawData:
        """
        Loads raw movie data from a structured dataset.

        This method purely loads the data without any transformation, adhering
        to the semantic meaning of its name.

        :param source: The data source object containing the dataset name.
        :returns: A list of MovieData objects.
        """
        pass
        # TODO

    @override
    def process_for_training(
        self, raw_data: PredictionRawData, config: PredictionTrainingConfig
    ) -> ProcessedPredictionData:
        """
        Processes raw movie data into a format suitable for model training.

        This method orchestrates the full transformation pipeline:
        1. Converts raw `MovieData` objects into weekly `WeekData`.
        2. Transforms weekly data into a numerical format.
        3. Creates time-series sequences for the LSTM model.
        4. Splits the data into train, validation, and test sets.
        5. Applies scaling to the split data.

        :param raw_data: The raw list of `MovieData` objects.
        :param config: A configuration object for the training process.
        :returns: The processed and split data, ready to be fed into a model.
        """
        pass
        # TODO

    @override
    def process_for_prediction(self, single_input: MovieData) -> NDArray[float32]:
        """
        Processes a single movie's data for prediction.

        :param single_input: A `MovieData` object for a single movie.
        :returns: A processed and padded sequence ready for the model.
        :raises ValueError: If artifacts (scaler, etc.) are not loaded or input is invalid.
        """
        pass
        # TODO

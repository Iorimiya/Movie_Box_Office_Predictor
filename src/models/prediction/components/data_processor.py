from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Final, Optional, TypedDict

import numpy as np
from numpy import float32, float64
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler
from typing_extensions import override

from src.core.logging_manager import LoggingManager
from src.data_handling.box_office import BoxOffice
from src.data_handling.file_io import PickleFile
from src.data_handling.movie_collections import MovieData, WeekData, MovieSessionData
from src.models.base.base_data_processor import BaseDataProcessor
from src.models.base.data_splitter import DatasetSplitter, SplitDataset


@dataclass(frozen=True)
class PredictionDataSource:
    """
    A data source configuration for the prediction model.

    :ivar dataset_name: The name of the structured dataset to load movie data from.
    """
    dataset_name: str


PredictionTrainingRawData: type = list[MovieData]


class PredictionTrainingProcessedData(TypedDict):
    """
    A TypedDict representing the processed and split dataset for box office prediction.

    :ivar x_train: Training data (features).
    :ivar y_train: Training data (labels).
    :ivar x_val: Validation data (features).
    :ivar y_val: Validation data (labels).
    :ivar x_test: Test data (features).
    :ivar y_test: Test data (labels).
    """
    x_train: NDArray[float32]
    y_train: NDArray[float64]
    x_val: NDArray[float32]
    y_val: NDArray[float64]
    x_test: NDArray[float32]
    y_test: NDArray[float64]


PredictionPredictionRawData: type = MovieData
PredictionPredictionProcessedData: type = NDArray[float32]

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


@dataclass(frozen=True)
class PredictionFeature:
    """
    A structured container for the features of a single week used in the prediction model.
    """
    box_office: int | float
    avg_sentiment: float
    reply_count: int

    def as_numerical_list(self) -> list[int | float]:
        """
        Converts the structured features into a numerical list for model input.

        :returns: A list of numerical features in a specific order.
        """
        return [self.box_office, self.avg_sentiment, self.reply_count]

class PredictionDataProcessor(
    BaseDataProcessor[
        PredictionDataSource,
        PredictionTrainingRawData,
        PredictionTrainingProcessedData,
        PredictionPredictionRawData,
        PredictionPredictionProcessedData,
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
        self.splitter: DatasetSplitter[NDArray[float32], NDArray[float64]] = \
            DatasetSplitter[NDArray[float32], NDArray[float64]]()
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
    def load_raw_data(self, source: PredictionDataSource) -> PredictionTrainingRawData:
        """
        Loads raw movie data from a structured dataset.

        This method purely loads the data without any transformation, adhering
        to the semantic meaning of its name.

        :param source: The data source object containing the dataset name.
        :returns: A list of MovieData objects.
        """
        self.logger.info(f"Loading raw prediction data from dataset: '{source.dataset_name}'")
        movie_data_list: list[MovieData] = MovieData.load_multiple_movie_data_from_dataset(
            dataset_name=source.dataset_name, mode='ALL')

        if not movie_data_list:
            self.logger.warning(f"No movie data loaded from dataset: {source.dataset_name}")

        return movie_data_list

    @override
    def process_for_training(
        self, raw_data: PredictionTrainingRawData, config: PredictionTrainingConfig
    ) -> PredictionTrainingProcessedData:
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
        self.training_week_len = config.training_week_len

        # transform_raw_to_weekly_data
        sessions: list[MovieSessionData] = MovieSessionData.create_sessions_from_movie_data_list(
            movie_data_list=raw_data, number_of_weeks=self.training_week_len+1)

        if not sessions:
            raise ValueError("No sessions data available.")

        x, y = PredictionDataProcessor._create_xy_from_sessions(sessions=sessions, week_limit=self.training_week_len)

        split_data: SplitDataset[NDArray[float32], NDArray[float64]] = self.splitter.split(
            x_data=x,
            y_data=y,
            split_ratios=config.split_ratios,
            random_state=config.random_state,
            shuffle=True
        )
        scaled_data: PredictionTrainingProcessedData = self._scale_data(unscaled_data=split_data)

        return scaled_data

    @override
    def process_for_prediction(self, single_input: PredictionPredictionRawData) -> PredictionPredictionProcessedData:
        """
        Processes a single movie's data for prediction.

        :param single_input: A `MovieData` object for a single movie.
        :returns: A processed and padded sequence ready for the model.
        :raises ValueError: If artifacts (scaler, etc.) are not loaded or input is invalid.
        """
        if not self.scaler or self.training_week_len is None:
            raise ValueError("Scaler or training_week_len has not been set. Please train first or load an artifact.")

        box_office_history: list[BoxOffice] = single_input.box_office
        if len(box_office_history) < self.training_week_len:
            raise ValueError(
                f"Input movie '{single_input.name}' has only {len(box_office_history)} weeks of data, "
                f"but the model requires {self.training_week_len} weeks."
            )
        latest_box_office_weeks: list[BoxOffice] = box_office_history[-self.training_week_len:]

        # Convert this slice into WeekData objects to get reviews
        latest_weeks_data: list[WeekData] = WeekData.create_multiple_week_data(
            weeks_data_source=latest_box_office_weeks,
            public_reviews_master_source=single_input.public_reviews,
            expert_reviews_master_source=single_input.expert_reviews
        )
        numerical_sequence: list[list[int | float]] = \
            PredictionDataProcessor._convert_weeks_to_numerical_sequence(weeks=latest_weeks_data)

        if len(numerical_sequence) != self.training_week_len:
            raise ValueError("Failed to create a numerical sequence of the required length.")

        # Scale the features
        processed_array: NDArray[float32] = np.array(numerical_sequence, dtype=float32)
        # Note: We only scale the box office feature (index 0)
        processed_array[:, 0] = self.scaler.transform(processed_array[:, 0].reshape(-1, 1)).flatten()

        # Reshape for model input (1 sample, N timesteps, M features)
        return np.expand_dims(processed_array, axis=0)

    @staticmethod
    def _create_xy_from_sessions(sessions: list[MovieSessionData], week_limit: int) \
        -> tuple[NDArray[float32], NDArray[float64]]:
        """
        Creates input sequences (x) and target values (y) from a list of MovieSessionData.

        Each session is converted into a single (x, y) pair. The first `week_limit`
        weeks form the input sequence (x), and the box office of the final week
        becomes the target (y).

        :param sessions: A list of `MovieSessionData` objects.
        :param week_limit: The number of past weeks to use as input features.
        :returns: A tuple of (x, y) as NumPy arrays.
        """
        x_list: list[list[list[int | float]]] = []
        y_list: list[float] = []

        for session in sessions:
            numerical_movie: list[list[int | float]] = (
                PredictionDataProcessor._convert_weeks_to_numerical_sequence(weeks=session.weeks_data))

            # Each session should have exactly `week_limit + 1` weeks.
            if len(numerical_movie) == week_limit + 1:
                seq_x: list[list[int | float]] = numerical_movie[:week_limit]
                seq_y: float = numerical_movie[week_limit][0]  # Target is the box office of the last week
                x_list.append(seq_x)
                y_list.append(seq_y)

        return np.array(x_list, dtype=float32), np.array(y_list, dtype=float64)

    @staticmethod
    def _extract_features_from_week(week: WeekData) -> PredictionFeature:
        """
        Extracts raw features from a WeekData object and populates a PredictionFeature container.

        :param week: The WeekData object to extract features from.
        :returns: A PredictionFeature object containing the extracted features.
        """
        return PredictionFeature(
            box_office=week.box_office_data.box_office,
            avg_sentiment=week.average_sentiment_score or 0.0,
            reply_count=week.total_reply_count
        )
    @staticmethod
    def _convert_weeks_to_numerical_sequence( weeks: list[WeekData]) -> list[list[int | float]]:
        """
        Converts a list of WeekData objects into a numerical sequence.

        Each WeekData object is transformed into a list of features:
        [box_office, average_sentiment_score, total_reply_count].

        :param weeks: A list of WeekData objects to be converted.
        :returns: A list of lists, where each inner list represents the numerical features for a week.
        """

        return list(map(lambda week: PredictionDataProcessor._extract_features_from_week(week=week).as_numerical_list(), weeks))

    def _scale_feature_in_sequences(self, sequences: NDArray[float32]) -> NDArray[float32]:
        """
        Applies the fitted scaler to the box office feature within sequences.

        This version assumes sequences are dense (no padding) and uses vectorized
        operations for efficiency.

        :param sequences: A 3D array of sequences (samples, timesteps, features).
        :returns: The sequences with the first feature scaled.
        """
        if not self.scaler:
            raise ValueError("Scaler is not fitted.")
        if sequences.size == 0:
            return sequences

        scaled_sequences: NDArray[float32] = sequences.copy()
        box_office_data: NDArray[float32] = scaled_sequences[:, :, 0].reshape(-1, 1)
        scaled_box_office: NDArray[float32] = self.scaler.transform(box_office_data)
        scaled_sequences[:, :, 0] = scaled_box_office.reshape(sequences.shape[0], sequences.shape[1])

        return scaled_sequences

    def _scale_data(self, unscaled_data: SplitDataset[NDArray[float32], NDArray[float64]]) \
        -> PredictionTrainingProcessedData:
        """
        Fits a scaler on the training data and applies it to all data splits.

        :param unscaled_data: A TypedDict containing the unscaled train, validation, and test sets.
        :returns: A TypedDict containing the scaled data splits.
        """
        self.logger.info("Scaling data splits.")
        x_train, y_train = unscaled_data['x_train'], unscaled_data['y_train']
        x_val, y_val = unscaled_data['x_val'], unscaled_data['y_val']
        x_test, y_test = unscaled_data['x_test'], unscaled_data['y_test']

        # Initialize and fit the scaler ONLY on the training target data
        self.scaler = MinMaxScaler()
        y_train_scaled = self.scaler.fit_transform(y_train.reshape(-1, 1)) if len(y_train) > 0 else np.array([])
        y_val_scaled = self.scaler.transform(y_val.reshape(-1, 1)) if len(y_val) > 0 else np.array([])
        y_test_scaled = self.scaler.transform(y_test.reshape(-1, 1)) if len(y_test) > 0 else np.array([])

        # Scale the box office feature (index 0) in x sets
        x_train_scaled = self._scale_feature_in_sequences(sequences=x_train)
        x_val_scaled = self._scale_feature_in_sequences(sequences=x_val)
        x_test_scaled = self._scale_feature_in_sequences(sequences=x_test)

        return PredictionTrainingProcessedData(
            x_train=x_train_scaled, y_train=y_train_scaled.flatten(),
            x_val=x_val_scaled, y_val=y_val_scaled.flatten(),
            x_test=x_test_scaled, y_test=y_test_scaled.flatten()
        )

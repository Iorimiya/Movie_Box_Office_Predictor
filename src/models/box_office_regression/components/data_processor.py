from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Optional, TypeAlias, TypedDict

from numpy import array, float32, float64
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler
from typing_extensions import override

from src.data_handling.file_io import PickleFile
from src.data_handling.movie_collections import MovieSessionData, WeekData
from src.models.base.data_splitter import SplitDataset
from src.models.box_office_common.box_office_data_processor import (
    BoxOfficeDataConfig, BoxOfficeFeature, BoxOfficeSessionDataProcessor, BoxOfficeTrainingRawData
)

BoxOfficeRegressionTrainingProcessedData: TypeAlias = SplitDataset[NDArray[float32], NDArray[float64]]
BoxOfficeRegressionPredictionProcessedData: TypeAlias = NDArray[float32]


class BoxOfficeRegressionConfigDict(TypedDict, total=False):
    """
    Type definition for the Box Office Regression Model's configuration dictionary (loaded from YAML).

    This ensures type safety when handling the raw configuration dictionary
    before it is converted into specific dataclasses.

    :ivar model_id: The unique identifier for the model series.
    :ivar dataset_name: The name of the source structured dataset.
    :ivar training_week_len: The number of past weeks used as input.
    :ivar lstm_units: The number of units in the LSTM layer.
    :ivar dropout_rate: The dropout rate to apply after the LSTM layer.
    :ivar epochs: The number of epochs for training.
    :ivar batch_size: The batch size for training.
    :ivar checkpoint_interval: The interval at which to save model checkpoints.
    :ivar early_stopping_patience: Number of epochs to wait for improvement.
    :ivar early_stopping_monitor: Metric to monitor for early stopping.
    :ivar early_stopping_min_delta: Minimum change to qualify as improvement.
    :ivar box_office_ranges: Upper boundaries for classification ranges.
    :ivar f1_average_method: Averaging method for F1 score calculation.
    :ivar split_ratios: Ratios for data splitting (train, val, test).
    :ivar random_state: Seed for the random number generator.
    """
    model_id: str
    dataset_name: str
    training_week_len: int
    lstm_units: int
    dropout_rate: float
    epochs: int
    batch_size: int
    checkpoint_interval: int
    early_stopping_patience: int
    early_stopping_monitor: str
    early_stopping_min_delta: float
    box_office_ranges: list[int]
    f1_average_method: str
    split_ratios: Optional[tuple[int, int, int]]
    random_state: Optional[int]


@dataclass(frozen=True)
class BoxOfficeRegressionFeature(BoxOfficeFeature):
    """
    A structured container for the features of a single week used in the Box Office Regression Model.

    :ivar box_office: The box office revenue for the week.
    :ivar avg_sentiment: The average sentiment score of reviews for the week.
    :ivar reply_count: The total number of replies to reviews for the week.
    :ivar total_positive_reply_count: The total number of positive reactions for the week.
    :ivar total_negative_reply_count: The total number of negative reactions for the week.
    """
    box_office: int | float
    avg_sentiment: float
    reply_count: int
    total_positive_reply_count: int
    total_negative_reply_count: int

    @override
    def as_numerical_list(self) -> list[int | float]:
        """
        Converts the structured features into a numerical list for model input.

        :return: A list of numerical features in a specific order.
        """
        return [
            self.box_office,
            self.avg_sentiment,
            self.reply_count,
            self.total_positive_reply_count,
            self.total_negative_reply_count
        ]

    @classmethod
    @override
    def from_week_data(cls, week: WeekData) -> 'BoxOfficeRegressionFeature':
        """
        Extracts raw features from a WeekData object.

        :param week: The WeekData object to extract features from.
        :return: A BoxOfficeRegressionFeature object containing the extracted features.
        """
        return cls(
            box_office=week.box_office,
            avg_sentiment=week.average_sentiment_score or 0.0,
            reply_count=week.total_reply_count,
            total_positive_reply_count=week.total_positive_reply_count,
            total_negative_reply_count=week.total_negative_reply_count
        )


class BoxOfficeRegressionDataProcessor(
    BoxOfficeSessionDataProcessor[
        BoxOfficeRegressionTrainingProcessedData,
        BoxOfficeRegressionPredictionProcessedData,
        NDArray[float32],
        NDArray[float64],
        BoxOfficeRegressionFeature,
        MinMaxScaler
    ]
):
    """
    Handles all data-related tasks for the Box Office Regression Model.

    This processor loads movie data, transforms it into weekly features,
    creates sequences, scales the data using a MinMaxScaler, and splits it
    into training, validation, and test sets. It manages the MinMaxScaler
    as its primary artifact.

    :cvar SCALER_FILE_NAME: The constant filename for the saved scaler artifact.
    :ivar _scaler: The ``MinMaxScaler`` instance used for scaling data.
    """

    SCALER_FILE_NAME: Final[str] = "scaler.pickle"

    @override
    def __init__(self, model_artifacts_path: Optional[Path] = None) -> None:
        """
        Initializes the BoxOfficeRegressionDataProcessor.

        :param model_artifacts_path: Path to the directory for model artifacts.
        """
        super().__init__(model_artifacts_path=model_artifacts_path, feature_class=BoxOfficeRegressionFeature)

    @override
    def save_artifacts(self) -> None:
        """
        Saves the MinMaxScaler to a single pickle file.

        :raises ValueError: If `model_artifacts_path` is not set or artifacts are not available.
        """
        if not self._model_artifacts_path:
            raise ValueError("model_artifacts_path is not set. Cannot save artifacts.")
        if self._scaler is None:
            raise ValueError("Scaler is not available to be saved.")

        self._model_artifacts_path.mkdir(parents=True, exist_ok=True)
        artifact_path: Path = self._model_artifacts_path / self.SCALER_FILE_NAME
        self._logger.debug(f"Saving scaler and settings artifact to: {artifact_path}")
        PickleFile(path=artifact_path).save(data=self._scaler)

    @override
    def load_artifacts(self) -> None:
        """
        Loads the MinMaxScaler from the artifact file.
        """
        if not self._model_artifacts_path:
            return

        artifact_path: Path = self._model_artifacts_path / self.SCALER_FILE_NAME
        if artifact_path.exists():
            self._logger.debug(f"Loading scaler artifact from: {artifact_path}")
            try:
                self._scaler: Optional[MinMaxScaler] = PickleFile(path=artifact_path).load()
                self._logger.debug("Scaler artifact loaded successfully.")
            except (TypeError, ValueError) as e:
                self._logger.error(f"Failed to load scaler artifact from {artifact_path}: {e}", exc_info=True)
                self._scaler: Optional[MinMaxScaler] = None

    @override
    def process_for_evaluation(
        self, raw_data: BoxOfficeTrainingRawData, config: BoxOfficeDataConfig
    ) -> tuple[NDArray[float32], NDArray[float64]]:
        """
        Processes a full raw dataset for evaluation without splitting it.

        :param raw_data: The raw list of `MovieSessionData` objects.
        :param config: A configuration object, primarily used for `training_week_len`.
        :return: A tuple containing the full processed features (x) and labels (y).
        :raises ValueError: If the scaler is not loaded or no data sessions can be created.
        """
        if config is None:
            raise ValueError(
                "BoxOfficeDataConfig is required for processing box office regression data."
            )

        self._logger.debug("Processing full dataset for evaluation (no splitting).")
        sessions: list[MovieSessionData] = raw_data

        if not sessions:
            raise ValueError("No sessions data available for evaluation.")

        x: NDArray[float32]
        y: NDArray[float64]
        x, y = self._create_xy_from_sessions(sessions=sessions, week_limit=config.training_week_len)

        # Scale the entire dataset using the preloaded scaler
        if not self._scaler:
            raise ValueError("Scaler must be loaded to process data for evaluation.")

        x_scaled: NDArray[float32] = self._scale_feature_in_sequences(sequences=x)
        y_scaled: NDArray[float64] = self._scaler.transform(X=y.reshape(-1, 1)).flatten()

        return x_scaled, y_scaled

    @override
    def _prepare_for_split(
        self, raw_data: BoxOfficeTrainingRawData, config: BoxOfficeDataConfig
    ) -> tuple[NDArray[float32], NDArray[float64]]:
        """
        Creates time-series sequences (x and y) from raw movie data.

        :param raw_data: The raw list of ``MovieSessionData`` objects.
        :param config: The data processing configuration.
        :return: A tuple containing the feature sequences (x) and target values (y).
        :raises ValueError: If no session data can be created from the raw data.
        """
        sessions: list[MovieSessionData] = raw_data

        if not sessions:
            raise ValueError("No sessions data available.")

        x: NDArray[float32]
        y: NDArray[float64]
        x, y = self._create_xy_from_sessions(sessions=sessions, week_limit=config.training_week_len)
        return x, y

    @override
    def _post_process_splits(
        self, split_data: SplitDataset[NDArray[float32], NDArray[float64]], config: BoxOfficeDataConfig
    ) -> BoxOfficeRegressionTrainingProcessedData:
        """
        Fits the scaler on the training data and applies it to all data splits.

        :param split_data: The TypedDict containing the train, validation, and test splits.
        :param config: The data processing configuration.
        :return: The final, fully processed and scaled data ready for model training.
        """
        return self._scale_data(unscaled_data=split_data)

    def _create_xy_from_sessions(
        self, sessions: list[MovieSessionData], week_limit: int
    ) -> tuple[NDArray[float32], NDArray[float64]]:
        """
        Creates input sequences (x) and target values (y) from a list of MovieSessionData.

        :param sessions: A list of `MovieSessionData` objects.
        :param week_limit: The number of past weeks to use as input features.
        :return: A tuple of (x, y) as NumPy arrays.
        """
        x_list: list[list[list[int | float]]] = []
        y_list: list[float] = []

        for session in sessions:
            numerical_movie: list[list[int | float]] = self._convert_weeks_to_numerical_sequence(
                weeks=session.weeks_data
            )

            # Each session should have exactly `week_limit + 1` weeks.
            if len(numerical_movie) == week_limit + 1:
                seq_x: list[list[int | float]] = numerical_movie[:week_limit]
                seq_y: float = numerical_movie[week_limit][0]
                x_list.append(seq_x)
                y_list.append(seq_y)

        return array(x_list, dtype=float32), array(y_list, dtype=float64)

    @override
    def _scale_feature_in_sequences(self, sequences: NDArray[float32]) -> NDArray[float32]:
        """
        Applies the fitted scaler to the box office feature within sequences.

        :param sequences: A 3D array of sequences (samples, timesteps, features).
        :return: The sequences with the first feature scaled.
        :raises ValueError: If the scaler has not been fitted.
        """
        if not self._scaler:
            raise ValueError("Scaler is not fitted.")
        if sequences.size == 0:
            return sequences

        scaled_sequences: NDArray[float32] = sequences.copy()
        # Scale only the first feature (box office) at index 0
        box_office_data: NDArray[float32] = scaled_sequences[:, :, 0].reshape(-1, 1)
        scaled_box_office: NDArray[float32] = self._scaler.transform(X=box_office_data)
        scaled_sequences[:, :, 0] = scaled_box_office.reshape(sequences.shape[0], sequences.shape[1])

        return scaled_sequences

    def _scale_data(
        self, unscaled_data: SplitDataset[NDArray[float32], NDArray[float64]]
    ) -> BoxOfficeRegressionTrainingProcessedData:
        """
        Fits a scaler on the training data and applies it to all data splits.

        :param unscaled_data: A TypedDict containing the unscaled data splits.
        :return: A TypedDict containing the scaled data splits.
        """
        self._logger.debug("Scaling data splits.")
        x_train: NDArray[float32] = unscaled_data['x_train']
        y_train: NDArray[float64] = unscaled_data['y_train']
        x_val: NDArray[float32] = unscaled_data['x_val']
        y_val: NDArray[float64] = unscaled_data['y_val']
        x_test: NDArray[float32] = unscaled_data['x_test']
        y_test: NDArray[float64] = unscaled_data['y_test']

        # Initialize and fit the scaler ONLY on the training target data
        self._scaler: Optional[MinMaxScaler] = MinMaxScaler()
        y_train_scaled: NDArray[Any] = self._scaler.fit_transform(X=y_train.reshape(-1, 1)) \
            if len(y_train) > 0 else array([])
        y_val_scaled: NDArray[Any] = self._scaler.transform(X=y_val.reshape(-1, 1)) \
            if len(y_val) > 0 else array([])
        y_test_scaled: NDArray[Any] = self._scaler.transform(X=y_test.reshape(-1, 1)) \
            if len(y_test) > 0 else array([])

        # Scale the box office feature (index 0) in x sets
        x_train_scaled: NDArray[float32] = self._scale_feature_in_sequences(sequences=x_train)
        x_val_scaled: NDArray[float32] = self._scale_feature_in_sequences(sequences=x_val)
        x_test_scaled: NDArray[float32] = self._scale_feature_in_sequences(sequences=x_test)

        return BoxOfficeRegressionTrainingProcessedData(
            x_train=x_train_scaled, y_train=y_train_scaled.flatten(),
            x_val=x_val_scaled, y_val=y_val_scaled.flatten(),
            x_test=x_test_scaled, y_test=y_test_scaled.flatten()
        )

    @staticmethod
    def get_range_index(value: float, ranges: tuple[int, ...]) -> int:
        """
        Determines the index of the range a given value falls into.

        :param value: The box office value to classify.
        :param ranges: A tuple of upper boundaries defining the ranges.
        :return: The integer index of the corresponding range.
        """
        # Sort ranges to ensure correct interval checking
        sorted_ranges: list[int] = sorted(list(ranges))

        # Find the first range boundary that the value is less than
        for i, boundary in enumerate(sorted_ranges):
            if value < boundary:
                return i

        # If the value is greater than or equal to all boundaries, it belongs to the last range
        return len(sorted_ranges)

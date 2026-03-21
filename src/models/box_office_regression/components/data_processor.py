from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Optional, TypeAlias, TypedDict

from numpy import array, expand_dims, float32, float64
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler
from typing_extensions import override

from src.data_handling.box_office import BoxOffice
from src.data_handling.dataset import BaseDataset, DatabaseDataset
from src.data_handling.file_io import PickleFile
from src.data_handling.movie_collections import MovieData, MovieSessionData, WeekData
from src.models.base.data_splitter import SplitDataset
from src.models.base.gradient_data_processor import GradientDataConfig, GradientDataProcessor


class BoxOfficeRegressionConfigDict(TypedDict, total=False):
    """
    Type definition for the Box Office Regression Model's configuration dictionary (loaded from YAML).

    This ensures type safety when handling the raw configuration dictionary before it is converted into specific dataclasses. It includes Optional fields to allow for partial configurations or fields that are not required in all modes (e.g., inference).
    """
    model_id: str
    dataset_name: str
    training_week_len: int
    lstm_units: int
    dropout_rate: float
    epochs: int
    batch_size: int
    verbose: int
    checkpoint_interval: int
    early_stopping_patience: int
    early_stopping_monitor: str
    early_stopping_min_delta: float
    box_office_ranges: list[int]
    f1_average_method: str
    # Made Optional to align with BoxOfficeRegressionDataConfig's flexibility for inference mode
    split_ratios: Optional[tuple[int, int, int]]
    # Made Optional as it might not be present in partial configs or inference
    random_state: Optional[int]


@dataclass(frozen=True)
class BoxOfficeRegressionDataSource:
    """
    A data source configuration for the Box Office Regression Model.

    :ivar dataset_name: The name of the structured dataset to load movie data from.
    """
    dataset_name: str


BoxOfficeRegressionTrainingRawData: TypeAlias = list[MovieSessionData]
BoxOfficeRegressionTrainingProcessedData: TypeAlias = SplitDataset[NDArray[float32], NDArray[float64]]
BoxOfficeRegressionPredictionRawData: TypeAlias = MovieData
BoxOfficeRegressionPredictionProcessedData: TypeAlias = NDArray[float32]


class BoxOfficeRegressionDataConfig(GradientDataConfig):
    """
    Configuration for the Box Office Regression Model's data processing.

    Inherits splitting capabilities from GradientDataConfig and adds specific parameters of Box Office Regression Model.

    :ivar _training_week_len: The length of the training week window.
    """
    _training_week_len: int

    def __init__(self, *,
                 training_week_len: int,
                 split_ratios: Optional[tuple[int, int, int]] = None,
                 random_state: Optional[int] = None,
                 **kwargs: Any):
        """
        Initializes the BoxOfficeRegressionDataConfig.

        :param training_week_len: The number of weeks of data to use for training.
        :param split_ratios: The ratio for splitting data (train, val, test).
        :param random_state: The seed for the random number generator.
        :param kwargs: Additional keyword arguments passed to the base class.
        """
        super().__init__(split_ratios=split_ratios, random_state=random_state, **kwargs)

        self._training_week_len = training_week_len

    @property
    def training_week_len(self) -> int:
        return self._training_week_len


@dataclass(frozen=True)
class BoxOfficeRegressionFeature:
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


class BoxOfficeRegressionDataProcessor(
    GradientDataProcessor[
        BoxOfficeRegressionDataSource,
        BoxOfficeRegressionTrainingRawData,
        BoxOfficeRegressionTrainingProcessedData,
        BoxOfficeRegressionPredictionRawData,
        BoxOfficeRegressionPredictionProcessedData,
        BoxOfficeRegressionDataConfig,
        NDArray[float32],
        NDArray[float64]
    ]
):
    """
    Handles all data-related tasks for the Box Office Regression Model.

    This processor loads movie data, transforms it into weekly features,
    creates sequences, scales the data using a MinMaxScaler, and splits it
    into training, validation, and test sets. It manages the MinMaxScaler
    as its primary artifact.

    :cvar SCALER_FILE_NAME: The constant filename for the saved scaler artifact.
    :ivar scaler: The ``MinMaxScaler`` instance used for scaling and inverse-scaling data.
                  It is ``None`` until it is fitted during training or loaded from an artifact.
    """

    SCALER_FILE_NAME: Final[str] = "scaler.pickle"

    @override
    def __init__(self, model_artifacts_path: Optional[Path] = None):
        """
        Initializes the BoxOfficeRegressionDataProcessor.

        :param model_artifacts_path: Path to the directory for model artifacts.
        """
        super().__init__(model_artifacts_path=model_artifacts_path)
        self.scaler: Optional[MinMaxScaler] = None
        self.load_artifacts()

    @override
    def save_artifacts(self) -> None:
        """
        Saves the MinMaxScaler and training parameters to a single pickle file.

        :raises ValueError: If `model_artifacts_path` is not set or artifacts are not available.
        """
        if not self.model_artifacts_path:
            raise ValueError("model_artifacts_path is not set. Cannot save artifacts.")
        if self.scaler is None:
            raise ValueError("Scaler is not available to be saved.")

        self.model_artifacts_path.mkdir(parents=True, exist_ok=True)
        artifact_path: Path = self.model_artifacts_path / self.SCALER_FILE_NAME
        self.logger.debug(f"Saving scaler and settings artifact to: {artifact_path}")
        PickleFile(path=artifact_path).save(data=self.scaler)

    @override
    def load_artifacts(self) -> None:
        """
        Loads the MinMaxScaler from the artifact file.
        """
        if not self.model_artifacts_path:
            return

        artifact_path: Path = self.model_artifacts_path / self.SCALER_FILE_NAME
        if artifact_path.exists():
            self.logger.debug(f"Loading scaler artifact from: {artifact_path}")
            try:
                self.scaler = PickleFile(path=artifact_path).load()
                self.logger.debug("Scaler artifact loaded successfully.")
            except (TypeError, ValueError) as e:
                self.logger.error(f"Failed to load scaler artifact from {artifact_path}: {e}", exc_info=True)
                self.scaler = None

    @override
    def load_raw_data(
        self, source: BoxOfficeRegressionDataSource, config: Optional[BoxOfficeRegressionDataConfig] = None
    ) -> BoxOfficeRegressionTrainingRawData:
        """
        Loads and processes raw movie data into fixed-length sessions.

        This method leverages the dataset's `load_movie_sessions` method, which
        efficiently loads and processes data based on the underlying storage.

        :param source: The data source object containing the dataset name.
        :param config: The data configuration, used to determine the session length.
        :returns: A list of MovieSessionData objects.
        """
        if config is None:
            raise ValueError("BoxOfficeRegressionDataConfig is required to determine session length.")

        self.logger.debug(f"Loading movie sessions from dataset: '{source.dataset_name}'")
        dataset: BaseDataset = DatabaseDataset(name=source.dataset_name)

        # The number of weeks needed is the training length + 1 for the target week
        number_of_weeks = config.training_week_len + 1
        sessions: list[MovieSessionData] = dataset.get_movie_sessions(number_of_weeks=number_of_weeks)

        if not sessions:
            self.logger.warning(f"No movie sessions loaded from dataset: {source.dataset_name}")

        return sessions

    @override
    def process_for_prediction(
        self, single_input: BoxOfficeRegressionPredictionRawData, config: BoxOfficeRegressionDataConfig
    ) -> BoxOfficeRegressionPredictionProcessedData:
        """
        Processes a single movie's data for box_office_regression.

        :param single_input: A `MovieData` object for a single movie.
        :param config: A configuration object containing necessary parameters like `training_week_len`.
        :returns: A processed and padded sequence ready for the model.
        :raises ValueError: If artifacts (scaler, etc.) are not loaded or input is invalid.
        """

        if config is None:
            raise ValueError(
                "BoxOfficeRegressionDataConfig is required for processing box office regression data."
            )

        if not self.scaler:
            raise ValueError("Scaler has not been set. Please train first or load an artifact.")

        box_office_history: list[BoxOffice] = single_input.box_office
        training_week_len: int = config.training_week_len
        if len(box_office_history) < training_week_len:
            raise ValueError(
                f"Input movie '{single_input.name}' has only {len(box_office_history)} weeks of data, "
                f"but the model requires {training_week_len} weeks."
            )
        latest_box_office_weeks: list[BoxOffice] = box_office_history[-training_week_len:]

        # Convert this slice into WeekData objects to get reviews
        latest_weeks_data: list[WeekData] = WeekData.create_multiple_from_source_variable(
            weeks_data_source=latest_box_office_weeks,
            public_reviews_master_source=single_input.public_reviews,
            movie_id=single_input.id
        )
        numerical_sequence: list[list[int | float]] = \
            BoxOfficeRegressionDataProcessor._convert_weeks_to_numerical_sequence(weeks=latest_weeks_data)

        if len(numerical_sequence) != training_week_len:
            raise ValueError("Failed to create a numerical sequence of the required length.")

        unscaled_array: NDArray[float32] = expand_dims(
            array(numerical_sequence, dtype=float32), axis=0
        )
        scaled_array: NDArray[float32] = self._scale_feature_in_sequences(sequences=unscaled_array)
        return scaled_array

    def process_for_evaluation(
        self, raw_data: BoxOfficeRegressionTrainingRawData, config: BoxOfficeRegressionDataConfig
    ) -> tuple[NDArray[float32], NDArray[float64]]:
        """
        Processes a full raw dataset for evaluation without splitting it.

        This method is designed for evaluating a model on a new, unseen dataset
        where the entire dataset should be treated as a single test set. It
        performs all processing steps (sequencing, feature extraction, scaling)
        except for the train/val/test split.

        :param raw_data: The raw list of `MovieSessionData` objects.
        :param config: A configuration object, primarily used for `training_week_len`.
        :returns: A tuple containing the full processed features (x) and labels (y).
        :raises ValueError: If the scaler is not loaded or no data sessions can be created.
        """

        if config is None:
            raise ValueError(
                "BoxOfficeRegressionDataConfig is required for processing box office regression data."
            )

        self.logger.debug("Processing full dataset for evaluation (no splitting).")
        sessions: list[MovieSessionData] = raw_data

        if not sessions:
            raise ValueError("No sessions data available for evaluation.")

        x, y = self._create_xy_from_sessions(sessions=sessions, week_limit=config.training_week_len)

        # Scale the entire dataset using the preloaded scaler
        if not self.scaler:
            raise ValueError("Scaler must be loaded to process data for evaluation.")

        x_scaled: NDArray[float32] = self._scale_feature_in_sequences(sequences=x)
        y_scaled: NDArray[float64] = self.scaler.transform(y.reshape(-1, 1)).flatten()

        return x_scaled, y_scaled

    @override
    def _prepare_for_split(
        self, raw_data: BoxOfficeRegressionTrainingRawData, config: BoxOfficeRegressionDataConfig
    ) -> tuple[NDArray[float32], NDArray[float64]]:
        """
        Creates time-series sequences (x and y) from raw movie data.

        :param raw_data: The raw list of ``MovieSessionData`` objects.
        :param config: The data processing configuration.
        :returns: A tuple containing the feature sequences (x) and target values (y).
        :raises ValueError: If no session data can be created from the raw data.
        """
        sessions: list[MovieSessionData] = raw_data

        if not sessions:
            raise ValueError("No sessions data available.")

        x, y = BoxOfficeRegressionDataProcessor._create_xy_from_sessions(sessions=sessions,
                                                                         week_limit=config.training_week_len)
        return x, y

    @override
    def _post_process_splits(
        self, split_data: SplitDataset[NDArray[float32], NDArray[float64]], config: BoxOfficeRegressionDataConfig
    ) -> BoxOfficeRegressionTrainingProcessedData:
        """
        Fits the scaler on the training data and applies it to all data splits.

        :param split_data: The TypedDict containing the train, validation, and test splits.
        :param config: The data processing configuration.
        :returns: The final, fully processed and scaled data ready for model training.
        """
        return self._scale_data(unscaled_data=split_data)

    @staticmethod
    def _create_xy_from_sessions(
        sessions: list[MovieSessionData], week_limit: int
    ) -> tuple[NDArray[float32], NDArray[float64]]:
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
                BoxOfficeRegressionDataProcessor._convert_weeks_to_numerical_sequence(weeks=session.weeks_data))

            # Each session should have exactly `week_limit + 1` weeks.
            if len(numerical_movie) == week_limit + 1:
                seq_x: list[list[int | float]] = numerical_movie[:week_limit]
                seq_y: float = numerical_movie[week_limit][0]  # Target is the box office of the last week
                x_list.append(seq_x)
                y_list.append(seq_y)

        return array(x_list, dtype=float32), array(y_list, dtype=float64)

    @staticmethod
    def _extract_features_from_week(week: WeekData) -> BoxOfficeRegressionFeature:
        """
        Extracts raw features from a WeekData object and populates a BoxOfficeRegressionFeature container.

        :param week: The WeekData object to extract features from.
        :returns: A BoxOfficeRegressionFeature object containing the extracted features.
        """
        return BoxOfficeRegressionFeature(
            box_office=week.box_office,
            avg_sentiment=week.average_sentiment_score or 0.0,
            reply_count=week.total_reply_count,
            total_positive_reply_count=week.total_positive_reply_count,
            total_negative_reply_count=week.total_negative_reply_count
        )

    @staticmethod
    def _convert_weeks_to_numerical_sequence(weeks: list[WeekData]) -> list[list[int | float]]:
        """
        Converts a list of WeekData objects into a numerical sequence.

        Each WeekData object is transformed into a list of features:
        [box_office, average_sentiment_score, total_reply_count].

        :param weeks: A list of WeekData objects to be converted.
        :returns: A list of lists, where each inner list represents the numerical features for a week.
        """

        return list(
            map(lambda week: BoxOfficeRegressionDataProcessor._extract_features_from_week(
                week=week).as_numerical_list(), weeks)
        )

    def _scale_feature_in_sequences(self, sequences: NDArray[float32]) -> NDArray[float32]:
        """
        Applies the fitted scaler to the box office feature within sequences.

        This version assumes sequences are dense (no padding) and uses vectorized
        operations for efficiency.

        :param sequences: A 3D array of sequences (samples, timesteps, features).
        :returns: The sequences with the first feature scaled.
        :raises ValueError: If the scaler has not been fitted.
        """
        if not self.scaler:
            raise ValueError("Scaler is not fitted.")
        if sequences.size == 0:
            return sequences

        scaled_sequences: NDArray[float32] = sequences.copy()
        box_office_data: NDArray[float32] = scaled_sequences[:, :, 0].reshape(-1, 1)
        scaled_box_office: NDArray[float32] = self.scaler.transform(box_office_data)
        scaled_sequences[:, :, 0] = scaled_box_office.reshape(sequences.shape[0], sequences.shape[1])

        box_office_data = scaled_sequences[:, :, 0].reshape(-1, 1)
        scaled_box_office = self.scaler.transform(box_office_data)
        scaled_sequences[:, :, 0] = scaled_box_office.reshape(sequences.shape[0], sequences.shape[1])

        return scaled_sequences

    def _scale_data(
        self, unscaled_data: SplitDataset[NDArray[float32], NDArray[float64]]
    ) -> BoxOfficeRegressionTrainingProcessedData:
        """
        Fits a scaler on the training data and applies it to all data splits.

        :param unscaled_data: A TypedDict containing the unscaled train, validation, and test sets.
        :returns: A TypedDict containing the scaled data splits.
        """
        self.logger.debug("Scaling data splits.")
        x_train, y_train = unscaled_data['x_train'], unscaled_data['y_train']
        x_val, y_val = unscaled_data['x_val'], unscaled_data['y_val']
        x_test, y_test = unscaled_data['x_test'], unscaled_data['y_test']

        # Initialize and fit the scaler ONLY on the training target data
        self.scaler = MinMaxScaler()
        y_train_scaled = self.scaler.fit_transform(y_train.reshape(-1, 1)) if len(y_train) > 0 else array([])
        y_val_scaled = self.scaler.transform(y_val.reshape(-1, 1)) if len(y_val) > 0 else array([])
        y_test_scaled = self.scaler.transform(y_test.reshape(-1, 1)) if len(y_test) > 0 else array([])

        # Scale the box office feature (index 0) in x sets
        x_train_scaled = self._scale_feature_in_sequences(sequences=x_train)
        x_val_scaled = self._scale_feature_in_sequences(sequences=x_val)
        x_test_scaled = self._scale_feature_in_sequences(sequences=x_test)

        return BoxOfficeRegressionTrainingProcessedData(
            x_train=x_train_scaled, y_train=y_train_scaled.flatten(),
            x_val=x_val_scaled, y_val=y_val_scaled.flatten(),
            x_test=x_test_scaled, y_test=y_test_scaled.flatten()
        )

    @staticmethod
    def get_range_index(value: float, ranges: tuple[int, ...]) -> int:
        """
        Determines the index of the range a given value falls into.

        This method is centralized here as it represents a form of data transformation
        and is part of the public utility API of this class.

        :param value: The box office value to classify.
        :param ranges: A tuple of upper boundaries defining the ranges (e.g., (1M, 10M, 90M)).
        :returns: The integer index of the corresponding range.
        """
        # Sort ranges to ensure correct interval checking
        sorted_ranges: list[int] = sorted(list(ranges))

        # Find the first range boundary that the value is less than
        for i, boundary in enumerate(sorted_ranges):
            if value < boundary:
                return i

        # If the value is greater than or equal to all boundaries, it belongs to the last range
        return len(sorted_ranges)

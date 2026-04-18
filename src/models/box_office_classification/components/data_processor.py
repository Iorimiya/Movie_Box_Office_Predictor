from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Optional, TypeAlias, TypedDict, Union

from numpy import array, float32, int_, percentile
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from typing_extensions import override

from src.data_handling.file_io import PickleFile
from src.data_handling.movie_collections import MovieSessionData, WeekData
from src.models.base.data_splitter import SplitDataset
from src.models.box_office_common.box_office_data_processor import (
    BoxOfficeDataConfig, BoxOfficeFeature, BoxOfficeSessionDataProcessor, BoxOfficeTrainingRawData
)

BoxOfficeClassificationTrainingProcessedData: TypeAlias = SplitDataset[NDArray[float32], NDArray[int_]]
BoxOfficeClassificationPredictionProcessedData: TypeAlias = NDArray[float32]


class BoxOfficeClassificationConfigDict(TypedDict, total=False):
    """
    Type definition for the Box Office Classification Model's configuration dictionary.

    This ensures type safety when handling the raw configuration dictionary
    before it is converted into specific dataclasses.

    :ivar model_id: The unique identifier for the model series.
    :ivar dataset_name: The name of the source structured dataset.
    :ivar training_week_len: The number of past weeks used as input.
    :ivar split_ratios: Ratios for data splitting (train, val, test).
    :ivar random_state: Seed for the random number generator.
    :ivar lstm_units: The number of units in the LSTM layer.
    :ivar dense_units: The number of units in the dense layer.
    :ivar dropout_rate: The dropout rate to apply after the LSTM layer.
    :ivar box_office_thresholds: The PR thresholds (e.g., (50, 80)) used for classification.
    :ivar learning_rate: The learning rate for the optimizer.
    :ivar clipnorm: The clipnorm value for gradient clipping.
    :ivar epochs: The number of epochs for training.
    :ivar batch_size: The batch size for training.
    :ivar early_stopping_patience: Number of epochs to wait for improvement.
    :ivar early_stopping_monitor: Metric to monitor for early stopping.
    :ivar early_stopping_min_delta: Minimum change to qualify as improvement.
    :ivar checkpoint_interval: The interval at which to save model checkpoints.
    :ivar f1_average_method: Averaging method for F1 score calculation.
    """
    model_id: str
    dataset_name: str
    training_week_len: int
    split_ratios: Optional[tuple[int, int, int]]
    random_state: Optional[int]
    lstm_units: int
    dense_units: int
    dropout_rate: float
    box_office_thresholds: tuple[int, ...]
    learning_rate: float
    clipnorm: float
    epochs: int
    batch_size: int
    early_stopping_patience: int
    early_stopping_monitor: str
    early_stopping_min_delta: float
    checkpoint_interval: int
    f1_average_method: str


@dataclass(frozen=True)
class BoxOfficeClassificationFeature(BoxOfficeFeature):
    """
    A structured container for the features of a single week used in the Box Office Classification Model.

    :ivar box_office: The box office revenue for the week.
    :ivar avg_sentiment: The average sentiment score of reviews for the week.
    :ivar reply_count: The total number of replies to reviews for the week.
    :ivar total_title_length: The total length of movie titles for the week.
    :ivar total_content_length: The total length of review content for the week.
    :ivar total_positive_reply_count: The total number of positive reactions for the week.
    :ivar total_negative_reply_count: The total number of negative reactions for the week.
    """
    box_office: int | float
    avg_sentiment: float
    reply_count: int
    total_title_length: int
    total_content_length: int
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
            self.total_title_length,
            self.total_content_length,
            self.total_positive_reply_count,
            self.total_negative_reply_count
        ]

    @classmethod
    @override
    def from_week_data(cls, week: WeekData) -> 'BoxOfficeClassificationFeature':
        """
        Extracts raw features from a WeekData object and populates a BoxOfficeClassificationFeature container.

        :param week: The WeekData object to extract features from.
        :return: A BoxOfficeClassificationFeature object containing the extracted features.
        """
        return cls(
            box_office=week.box_office,
            avg_sentiment=week.average_sentiment_score or 0.0,
            reply_count=week.total_reply_count,
            total_title_length=week.total_title_length,
            total_content_length=week.total_content_length,
            total_positive_reply_count=week.total_positive_reply_count,
            total_negative_reply_count=week.total_negative_reply_count
        )


class BoxOfficeClassificationDataProcessor(
    BoxOfficeSessionDataProcessor[
        BoxOfficeClassificationTrainingProcessedData,
        BoxOfficeClassificationPredictionProcessedData,
        NDArray[float32],
        NDArray[int_],
        BoxOfficeClassificationFeature,
        StandardScaler
    ]
):
    """
    Handles data processing for the Box Office Classification Model.

    :cvar ARTIFACT_FILE_NAME: The constant filename for the saved artifact.
    :ivar _box_office_threshold_percents: A tuple of PR percentages used for classification.
    :ivar _calculated_thresholds: A tuple containing the calculated box office amount thresholds.
    :ivar _scaler: The ``StandardScaler`` instance used for scaling data.
    """

    ARTIFACT_FILE_NAME: Final[str] = "artifact.pickle"
    _box_office_threshold_percents: Optional[tuple[int, ...]]
    _calculated_thresholds: Optional[tuple[float, ...]]

    @override
    def __init__(
        self,model_artifacts_path: Optional[Path] = None,box_office_thresholds: Optional[tuple[int, ...]] = None
    ) -> None:
        """
        Initializes the BoxOfficeClassificationProcessor.

        :param model_artifacts_path: Path to the directory for model artifacts.
        :param box_office_thresholds: The PR thresholds (e.g., (50, 80)) used for classification.
        """
        self._box_office_threshold_percents: Optional[tuple[int, ...]] = box_office_thresholds
        self._calculated_thresholds: Optional[tuple[float, ...]] = None
        super().__init__(model_artifacts_path=model_artifacts_path, feature_class=BoxOfficeClassificationFeature)

    @property
    @override
    def is_prepared(self) -> bool:
        """
        Checks if the processor is prepared for evaluation or continued training.

        Classification model requires both the scaler and the calculated thresholds.

        :return: True if both are available, False otherwise.
        """
        return super().is_prepared and self._calculated_thresholds is not None

    @override
    def save_artifacts(self) -> None:
        """
        Saves the StandardScaler and calculated PR thresholds to a single pickle file.

        :raises ValueError: If `model_artifacts_path` is not set or artifacts are not available.
        """
        if not self._model_artifacts_path:
            raise ValueError("model_artifacts_path is not set.")
        if self._scaler is None or self._calculated_thresholds is None:
            raise ValueError("Artifacts are not available to be saved.")

        self._model_artifacts_path.mkdir(parents=True, exist_ok=True)
        artifact_path: Path = self._model_artifacts_path / self.ARTIFACT_FILE_NAME
        self._logger.debug(f"Saving scaler and settings artifact to: {artifact_path}")
        artifacts: dict[str, Union[StandardScaler, tuple[float, ...]]] = {
            'scaler': self._scaler,
            'thresholds': self._calculated_thresholds
        }
        PickleFile(path=artifact_path).save(data=artifacts)

    @override
    def load_artifacts(self) -> None:
        """
        Loads the StandardScaler and PR thresholds from the artifact file.
        """
        if not self._model_artifacts_path:
            return

        artifact_path: Path = self._model_artifacts_path / self.ARTIFACT_FILE_NAME
        if artifact_path.exists():
            self._logger.debug(f"Loading classification artifacts from: {artifact_path}")
            try:
                artifacts: dict[str, Any] = PickleFile(path=artifact_path).load()
                self._scaler: Optional[StandardScaler] = artifacts.get('scaler')
                self._calculated_thresholds: Optional[tuple[float, ...]] = artifacts.get('thresholds')

                if not isinstance(self._scaler, StandardScaler) and self._scaler is not None:
                    self._logger.warning(
                        f"Loaded scaler is not a StandardScaler instance from {artifact_path}. Resetting scaler.")
                    self._scaler = None
                if not isinstance(self._calculated_thresholds, tuple) and self._calculated_thresholds is not None:
                    self._logger.warning(
                        f"Loaded thresholds is not a tuple instance from {artifact_path}. Resetting thresholds.")
                    self._calculated_thresholds = None

                if self._scaler and self._calculated_thresholds:
                    self._logger.debug("Classification artifacts loaded successfully.")
                else:
                    self._logger.warning(
                        "Some classification artifacts were missing or invalid after loading. Resetting to None.")
                    self._scaler = None
                    self._calculated_thresholds = None

            except (TypeError, ValueError) as e:
                self._logger.error(f"Failed to load artifacts: {e}")
                self._scaler = None
                self._calculated_thresholds = None

    @override
    def process_for_evaluation(
        self, raw_data: BoxOfficeTrainingRawData, config: BoxOfficeDataConfig
    ) -> tuple[NDArray[float32], NDArray[int_]]:
        """
        Processes a full raw dataset for evaluation without splitting it.

        This method follows a similar logic to the regression version, ensuring
        consistent preprocessing before model evaluation.

        :param raw_data: The raw list of `MovieSessionData` objects.
        :param config: A configuration object.
        :return: A tuple containing the processed features (x) and labels (y).
        :raises ValueError: If the configuration, scaler, or thresholds are missing.
        """
        if config is None:
            raise ValueError(
                "BoxOfficeDataConfig is required for processing box office classification data."
            )

        self._logger.debug("Processing full dataset for evaluation (no splitting).")
        sessions: list[MovieSessionData] = raw_data

        if not sessions:
            raise ValueError("No sessions data available for evaluation.")

        # Thresholds must be loaded from artifacts for consistent evaluation
        if not self.is_prepared:
            raise ValueError("DataProcessor is not prepared (scaler or thresholds missing) for evaluation.")

        x: NDArray[float32]
        y: NDArray[int_]
        x, y = self._create_xy_from_sessions(
            sessions=sessions, week_limit=config.training_week_len,
            thresholds=self._calculated_thresholds
        )

        # Scale the feature sequences
        x_scaled: NDArray[float32] = self._scale_feature_in_sequences(sequences=x)

        return x_scaled, y

    @staticmethod
    def _calculate_pr_thresholds(
        sessions: list[MovieSessionData], pr_values: tuple[int, ...]
    ) -> tuple[float, ...]:
        """
        Calculates box office thresholds based on provided percentile (PR) values from deduplicated session data.

        :param sessions: A list of MovieSessionData objects.
        :param pr_values: A tuple of percentile values (e.g., (50, 80) for PR50 and PR80).
        :return: A tuple containing the calculated percentile thresholds.
        """
        all_weeks_map: dict[tuple[int, Any], int] = {}
        for session in sessions:
            for week in session.weeks_data:
                all_weeks_map[(session.id, week.start_date)] = week.box_office

        all_box_offices: list[int] = list(all_weeks_map.values())
        if not all_box_offices:
            return tuple(0.0 for _ in pr_values)

        calculated_thresholds: list[float] = [float(percentile(a=all_box_offices, q=pr)) for pr in pr_values]
        return tuple(calculated_thresholds)

    def _create_xy_from_sessions(
        self, sessions: list[MovieSessionData], week_limit: int, thresholds: tuple[float, ...]
    ) -> tuple[NDArray[float32], NDArray[int_]]:
        """
        Creates input sequences (x) and classification labels (y) from movie sessions.

        :param sessions: A list of MovieSessionData objects.
        :param week_limit: The number of weeks to use as input features.
        :param thresholds: The calculated thresholds for labeling.
        :return: A tuple of (x, y) as NumPy arrays.
        """
        x_list: list[list[list[int | float]]] = []
        y_list: list[int] = []

        sorted_thresholds: tuple[float, ...] = tuple(sorted(thresholds))

        for session in sessions:
            if len(session.weeks_data) != week_limit + 1:
                continue
            numerical_features: list[list[int | float]] = self._convert_weeks_to_numerical_sequence(
                weeks=session.weeks_data[:week_limit]
            )

            target_bo: int | float = session.weeks_data[week_limit].box_office

            y_class: int = 0
            for i, threshold in enumerate(sorted_thresholds):
                if target_bo >= threshold:
                    y_class = i + 1
                else:
                    break

            x_list.append(numerical_features)
            y_list.append(y_class)

        return array(x_list, dtype=float32), array(y_list, dtype=int_)

    @override
    def _prepare_for_split(self, raw_data: BoxOfficeTrainingRawData, config: BoxOfficeDataConfig) -> tuple[NDArray[float32], NDArray[int_]]:
        """
        Prepares raw data for splitting by calculating thresholds and generating sequences.

        :param raw_data: The raw list of MovieSessionData objects.
        :param config: The data configuration.
        :return: A tuple containing the feature sequences (x) and classification labels (y).
        :raises ValueError: If box office thresholds percents are not provided.
        """
        if self._box_office_threshold_percents is None:
            raise ValueError(
                "Box office thresholds percents must be provided to the processor for training."
            )

        self._logger.debug("Preparing data for split: Calculating thresholds and generating sequences.")

        # 1. Calculate thresholds based on self._box_office_threshold_percents
        self._calculated_thresholds = self._calculate_pr_thresholds(
            sessions=raw_data, pr_values=self._box_office_threshold_percents
        )

        self._logger.info(f"Global PR Thresholds set: {self._calculated_thresholds}")

        # 2. Convert data to sequences and labels
        x: NDArray[float32]
        y: NDArray[int_]
        x, y = self._create_xy_from_sessions(
            sessions=raw_data,
            week_limit=config.training_week_len,
            thresholds=self._calculated_thresholds
        )

        return x, y

    @override
    def _post_process_splits(
        self, split_data: SplitDataset[NDArray[float32], NDArray[int_]], config: BoxOfficeDataConfig
    ) -> BoxOfficeClassificationTrainingProcessedData:
        """
        Fits the scaler on the training data and applies it to all data splits.

        :param split_data: The dataset splits containing train, val, and test sets.
        :param config: The data configuration.
        :return: The scaled and processed dataset splits.
        """
        self._scaler: Optional[StandardScaler] = StandardScaler()

        x_train: NDArray[float32] = split_data['x_train']
        if x_train.size > 0:
            # Fit scaler on training data (flatten to fit StandardScaler requirements)
            n_features: int = x_train.shape[2]
            self._scaler.fit(X=x_train.reshape(-1, n_features))

        # Apply scaling to all splits
        split_data['x_train'] = self._scale_feature_in_sequences(sequences=split_data['x_train'])
        split_data['x_val'] = self._scale_feature_in_sequences(sequences=split_data['x_val'])
        split_data['x_test'] = self._scale_feature_in_sequences(sequences=split_data['x_test'])

        return split_data

    @override
    def _scale_feature_in_sequences(self, sequences: NDArray[float32]) -> NDArray[float32]:
        """
        Standardizes the input sequences using the fitted scaler.

        This method reshapes the 3D sequence data into 2D for scaling and then
        reshapes it back to 3D.

        :param sequences: A 3D array of sequences (samples, timesteps, features).
        :return: The scaled sequences.
        :raises ValueError: If the scaler has not been fitted.
        """
        if not self._scaler:
            raise ValueError("Scaler is not fitted.")
        if sequences.size == 0:
            return sequences

        n_features: int = sequences.shape[2]
        flat_data: NDArray[float32] = sequences.reshape(-1, n_features)
        scaled_flat: NDArray[float32] = self._scaler.transform(X=flat_data)
        return scaled_flat.reshape(sequences.shape).astype(dtype=float32)

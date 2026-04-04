from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast, Final, Optional, TypedDict, TypeAlias, Union

from numpy import float32, int_, percentile
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from typing_extensions import override

from src.data_handling.file_io import PickleFile
from src.data_handling.movie_collections import MovieSessionData, WeekData
from src.models.base.data_splitter import SplitDataset
from src.models.box_office_common import (
    BoxOfficeDataConfig,
    BoxOfficeFeature,
    BoxOfficeSessionDataProcessor,
    BoxOfficeTrainingRawData
)

BoxOfficeClassificationTrainingProcessedData: TypeAlias = SplitDataset[NDArray[float32], NDArray[int_]]
BoxOfficeClassificationPredictionProcessedData: TypeAlias = NDArray[float32]


class BoxOfficeClassificationConfigDict(TypedDict, total=False):
    """
    Type definition for the Box Office Classification Model's configuration dictionary.
    """
    model_id: str
    dataset_name: str

    # 資料處理
    training_week_len: int
    split_ratios: Optional[tuple[int, int, int]]
    random_state: Optional[int]

    # 模型結構 (與 BiLSTM 嵌合)
    lstm_units: int
    dense_units: int
    dropout_rate: float
    num_classes: int

    # 最佳化與訓練
    learning_rate: float
    clipnorm: float
    epochs: int
    batch_size: int

    # 回調與監控
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
                :returns: A BoxOfficeClassificationFeature object containing the extracted features.
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

    :ivar _calculated_thresholds: A list containing the box office PR50 and PR80 thresholds [PR50, PR80]
                                  used to classify target box office values into labels (0, 1, 2).
    """

    ARTIFACT_FILE_NAME: Final[str] = "artifact.pickle"

    @override
    def __init__(self, model_artifacts_path: Optional[Path] = None):
        """
        Initializes the BoxOfficeClassificationProcessor.

        :param model_artifacts_path: Path to the directory for model artifacts.
        """
        self._calculated_thresholds: Optional[list[float]] = None
        super().__init__(model_artifacts_path=model_artifacts_path, feature_class=BoxOfficeClassificationFeature)

    @override
    def save_artifacts(self) -> None:
        """
        Saves the StandardScaler and calculated PR thresholds to a single pickle file.

        :raises ValueError: If `model_artifacts_path` is not set or artifacts are not available.
        """
        if not self.model_artifacts_path:
            raise ValueError("model_artifacts_path is not set.")
        if self.scaler is None or self._calculated_thresholds is None:
            raise ValueError("Artifacts are not available to be saved.")

        self.model_artifacts_path.mkdir(parents=True, exist_ok=True)
        artifact_path: Path = self.model_artifacts_path / self.ARTIFACT_FILE_NAME
        self.logger.debug(f"Saving scaler and settings artifact to: {artifact_path}")
        artifacts: dict[str, Union[StandardScaler, list[float]]] = {
            'scaler': self.scaler,
            'thresholds': self._calculated_thresholds
        }
        PickleFile(path=artifact_path).save(data=artifacts)

    @override
    def load_artifacts(self) -> None:
        """
        Loads the StandardScaler and PR thresholds from the artifact file.
        """
        if not self.model_artifacts_path:
            return

        artifact_path: Path = self.model_artifacts_path / self.ARTIFACT_FILE_NAME
        if artifact_path.exists():
            self.logger.debug(f"Loading classification artifacts from: {artifact_path}")
            try:
                artifacts: dict[str, Any] = PickleFile(path=artifact_path).load()
                self.scaler = artifacts.get('scaler')
                self._calculated_thresholds = artifacts.get('thresholds')

                if not isinstance(self.scaler, StandardScaler) and self.scaler is not None:
                    self.logger.warning(
                        f"Loaded scaler is not a StandardScaler instance from {artifact_path}. Resetting scaler.")
                    self.scaler = None
                if not isinstance(self._calculated_thresholds, list) and self._calculated_thresholds is not None:
                    self.logger.warning(
                        f"Loaded thresholds is not a list instance from {artifact_path}. Resetting thresholds.")
                    self._calculated_thresholds = None

                if self.scaler and self._calculated_thresholds:
                    self.logger.debug("Classification artifacts loaded successfully.")
                else:
                    self.logger.warning(
                        "Some classification artifacts were missing or invalid after loading. Resetting to None.")
                    self.scaler = None
                    self._calculated_thresholds = None

            except (TypeError, ValueError) as e:
                self.logger.error(f"Failed to load artifacts: {e}")
                self.scaler = None
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
        :returns: A tuple containing the processed features (x) and labels (y).
        :raises ValueError: If the configuration, scaler, or thresholds are missing.
        """

        # [Consistent with Regression] Validation of configuration
        if config is None:
            raise ValueError(
                "BoxOfficeDataConfig is required for processing box office classification data."
            )

        # [Consistent with Regression] Log processing start
        self.logger.debug("Processing full dataset for evaluation (no splitting).")
        sessions: list[MovieSessionData] = raw_data

        # [Consistent with Regression] Ensure data availability
        if not sessions:
            raise ValueError("No sessions data available for evaluation.")

        # [Specific to Classification] Thresholds must be loaded from artifacts for consistent evaluation
        if self._calculated_thresholds is None:
            raise ValueError("Thresholds must be loaded (from artifacts) to process data for evaluation.")

        # [Consistent with Regression] Sequence creation
        # Logic is identical: convert sessions to (x, y) pairs based on week_limit.
        # Classification additionally uses thresholds to determine y_class.
        x, y = self._create_xy_from_sessions(
            sessions=sessions,week_limit=config.training_week_len,thresholds=self._calculated_thresholds
        )

        # [Consistent with Regression] Ensure scaler availability
        if not self.scaler:
            raise ValueError("Scaler must be loaded to process data for evaluation.")

        # [Consistent with Regression] Scale the feature sequences
        # Note: Regression version uses MinMaxScaler, while Classification uses StandardScaler.
        x_scaled: NDArray[float32] = self._scale_feature_in_sequences(sequences=x)

        # [Specific to Classification] y labels (classes 0, 1, 2) do not require scaling,
        # unlike the regression version which scales the box office target value.
        return x_scaled, y

    @staticmethod
    def _calculate_pr_thresholds(sessions: list[MovieSessionData]) -> list[float]:
        """
        Calculates the global PR50 and PR80 box office thresholds from deduplicated session data.

        :param sessions: A list of MovieSessionData objects.
        :returns: A list containing [PR50, PR80] values.
        """

        all_weeks_map: dict[tuple[int, Any], int] = {}
        for session in sessions:
            for week in session.weeks_data:
                all_weeks_map[(session.id, week.start_date)] = week.box_office

        all_box_offices = list(all_weeks_map.values())
        if not all_box_offices:
            return [0.0, 0.0]

        p50: float = float(percentile(all_box_offices, 50))
        p80: float = float(percentile(all_box_offices, 80))

        return [p50, p80]

    def _create_xy_from_sessions(
        self, sessions: list[MovieSessionData], week_limit: int, thresholds: list[float]
    ) -> tuple[NDArray[float32], NDArray[int_]]:
        """
        Creates input sequences (x) and classification labels (y) from movie sessions.

        :param sessions: A list of MovieSessionData objects.
        :param week_limit: The number of weeks to use as input features.
        :param thresholds: The PR50 and PR80 thresholds for labeling.
        :returns: A tuple of (x, y) as NumPy arrays.
        """

        x_list = []
        y_list = []

        p50, p80 = thresholds

        for session in sessions:
            if len(session.weeks_data) != week_limit + 1:
                continue
            numerical_features = self._convert_weeks_to_numerical_sequence(weeks=session.weeks_data[:week_limit])

            target_bo = session.weeks_data[week_limit].box_office

            if target_bo < p50:
                y_class = 0
            elif target_bo < p80:
                y_class = 1
            else:
                y_class = 2

            x_list.append(numerical_features)
            y_list.append(y_class)

        return cast(NDArray[float32], cast(object, x_list)), cast(NDArray[int_], cast(object, y_list))

    @override
    def _prepare_for_split(
        self, raw_data: BoxOfficeTrainingRawData, config: BoxOfficeDataConfig
    ) -> tuple[NDArray[float32], NDArray[int_]]:
        """
        Prepares raw data for splitting by calculating thresholds and generating sequences.

        :param raw_data: The raw list of MovieSessionData objects.
        :param config: The data configuration.
        :returns: A tuple containing the feature sequences (x) and classification labels (y).
        """
        self.logger.debug("Preparing data for split: Calculating thresholds and generating sequences.")

        # 1. 執行封裝好的門檻計算邏輯
        self._calculated_thresholds = self._calculate_pr_thresholds(sessions=raw_data)

        self.logger.info(
            f"Global PR Thresholds set - PR50: {self._calculated_thresholds[0]:.0f}, "
            f"PR80: {self._calculated_thresholds[1]:.0f}"
        )

        # 2. 執行封裝好的資料轉換邏輯
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
        :returns: The scaled and processed dataset splits.
        """
        self.scaler = StandardScaler()

        x_train = split_data['x_train']
        if x_train.size > 0:
            # 僅進行 Fit (需展平以符合 StandardScaler 要求)
            n_features = x_train.shape[2]
            self.scaler.fit(x_train.reshape(-1, n_features))

        # 調用封裝好的縮放方法處理所有分割
        split_data['x_train'] = self._scale_feature_in_sequences(split_data['x_train'])
        split_data['x_val'] = self._scale_feature_in_sequences(split_data['x_val'])
        split_data['x_test'] = self._scale_feature_in_sequences(split_data['x_test'])

        return split_data

    @override
    def _scale_feature_in_sequences(self, sequences: NDArray[float32]) -> NDArray[float32]:
        """
        Standardizes the input sequences using the fitted scaler.
        """
        if not self.scaler:
            raise ValueError("Scaler is not fitted.")
        if sequences.size == 0:
            return sequences

        # 展平與還原維度的邏輯封裝在此
        n_features = sequences.shape[2]
        flat_data = sequences.reshape(-1, n_features)
        scaled_flat = self.scaler.transform(flat_data)
        return scaled_flat.reshape(sequences.shape).astype(float32)

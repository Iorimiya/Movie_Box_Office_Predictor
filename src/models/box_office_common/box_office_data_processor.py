from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Optional, Type, TypeAlias, TypeVar

from numpy import array, expand_dims, float32
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.model_selection import GroupShuffleSplit
from typing_extensions import override

from src.data_handling.box_office import BoxOffice
from src.data_handling.dataset import BaseDataset
from src.data_handling.movie_collections import MovieData, MovieSessionData, WeekData
from src.models.base.base_data_processor import ProcessedPredictionDataType, ProcessedTrainingDataType
from src.models.base.data_splitter import SplitDataset, X_Type, Y_Type
from src.models.base.gradient_data_processor import GradientDataConfig, GradientDataProcessor

BoxOfficeDataSource: TypeAlias = BaseDataset
BoxOfficeTrainingRawData: TypeAlias = list[MovieSessionData]
BoxOfficePredictionRawData: TypeAlias = MovieData


class BoxOfficeDataConfig(GradientDataConfig):
    """
    Configuration for the Box Office Model's data processing.

    Inherits splitting capabilities from GradientDataConfig and adds specific parameters
    of Box Office Regression Model.

    :ivar _training_week_len: The length of the training week window.
    """
    _training_week_len: int

    def __init__(
        self,
        *,
        training_week_len: int,
        split_ratios: Optional[tuple[int, int, int]] = None,
        random_state: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """
        Initializes the BoxOfficeDataConfig.

        :param training_week_len: The number of weeks of data to use for training.
        :param split_ratios: The ratio for splitting data (train, val, test).
        :param random_state: The seed for the random number generator.
        :param kwargs: Additional keyword arguments passed to the base class.
        """
        super().__init__(split_ratios=split_ratios, random_state=random_state, **kwargs)

        self._training_week_len: int = training_week_len

    @property
    def training_week_len(self) -> int:
        return self._training_week_len


@dataclass(frozen=True)
class BoxOfficeFeature(ABC):
    """
    An abstract base class for structured features derived from movie data.
    """

    @abstractmethod
    def as_numerical_list(self) -> list[int | float]:
        """
        Converts the structured features into a numerical list for model input.

        :return: A list of numerical features in a specific order.
        """
        pass

    @classmethod
    @abstractmethod
    def from_week_data(cls, week: WeekData) -> 'BoxOfficeFeature':
        """
        Factory method to create a Feature instance from WeekData.

        :param week: The week-specific data to extract features from.
        :return: A new instance of a concrete BoxOfficeFeature.
        """
        pass


FeatureClassType = TypeVar('FeatureClassType', bound=BoxOfficeFeature)
ScalerType = TypeVar('ScalerType', bound=BaseEstimator)


class BoxOfficeSessionDataProcessor(
    GradientDataProcessor[
        BoxOfficeDataSource,
        BoxOfficeTrainingRawData,
        ProcessedTrainingDataType,
        BoxOfficePredictionRawData,
        ProcessedPredictionDataType,
        BoxOfficeDataConfig,
        X_Type,
        Y_Type
    ],
    ABC,
    Generic[ProcessedTrainingDataType, ProcessedPredictionDataType, X_Type, Y_Type, FeatureClassType, ScalerType]
):
    """
    A common data processor for box office sequence data.

    It implements the core logic for loading movie sessions and processing them
    for training, ensuring strict isolation at the movie ID level during
    data splitting.
    """

    _feature_class: Type[FeatureClassType]
    _scaler: Optional[ScalerType]

    @override
    def __init__(self, feature_class: Type[FeatureClassType], model_artifacts_path: Optional[Path] = None) -> None:
        """
        Initializes the BoxOfficeSessionDataProcessor.

        :param feature_class: The concrete Feature class that this processor will use.
        :param model_artifacts_path: Path to the directory for model artifacts.
        """
        super().__init__(model_artifacts_path=model_artifacts_path)
        self._feature_class: Type[FeatureClassType] = feature_class
        self._scaler: Optional[ScalerType] = None
        self.load_artifacts()

    @property
    def scaler(self) -> Optional[ScalerType]:
        """
        Returns the fitted scaler instance.

        :return: The scaler object if fitted, otherwise None.
        """
        return self._scaler

    @property
    def is_prepared(self) -> bool:
        """
        Checks if the processor is prepared for evaluation or continued training.

        :return: True if prepared (scaler is loaded), False otherwise.
        """
        return self._scaler is not None

    @override
    def load_raw_data(
        self, source: BoxOfficeDataSource, config: Optional[BoxOfficeDataConfig] = None
    ) -> BoxOfficeTrainingRawData:
        """
        Loads and processes raw movie data into fixed-length sessions.

        :param source: The BaseDataset instance to load data from.
        :param config: The data configuration, used to determine the session length.
        :return: A list of MovieSessionData objects.
        :raises ValueError: If the config is not provided.
        """
        if config is None:
            raise ValueError("BoxOfficeDataConfig is required to determine session length.")

        self._logger.debug(f"Loading movie sessions from dataset: '{source.name}'")

        # The number of weeks needed is the training length + 1 for the target week
        number_of_weeks: int = config.training_week_len + 1
        sessions: list[MovieSessionData] = source.get_movie_sessions(number_of_weeks=number_of_weeks)

        if not sessions:
            self._logger.warning(f"No movie sessions loaded from dataset: {source.name}")

        return sessions

    @staticmethod
    def _split_by_group(
        x: NDArray[Any],
        y: NDArray[Any],
        groups: NDArray[Any],
        test_size: float,
        random_state: Optional[int]
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
        """
        Splits the data into two parts based on groups (Movie IDs) to ensure isolation.

        :param x: The feature array.
        :param y: The label array.
        :param groups: The array of group identifiers.
        :param test_size: The proportion of the dataset to include in the test split.
        :param random_state: The seed for the random number generator.
        :return: A tuple of (x_train, y_train, groups_train, x_test, y_test).
        """
        gss: GroupShuffleSplit = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx: NDArray[Any]
        test_idx: NDArray[Any]
        train_idx, test_idx = next(gss.split(X=x, y=y, groups=groups))

        return x[train_idx], y[train_idx], groups[train_idx], x[test_idx], y[test_idx]

    @override
    def process_for_training(
        self, raw_data: BoxOfficeTrainingRawData, config: BoxOfficeDataConfig
    ) -> ProcessedTrainingDataType:
        """
        Processes raw data for training using movie-level isolation for splitting.

        :param raw_data: The raw movie session data.
        :param config: The configuration for data processing and splitting.
        :return: The processed and split training data.
        """
        # 1. Prepare data (feature extraction)
        x_to_split: X_Type
        y_to_split: Y_Type
        x_to_split, y_to_split = self._prepare_for_split(raw_data=raw_data, config=config)

        # 2. Extract group labels (Movie IDs) for isolation
        movie_ids: NDArray[Any] = array([
            s.id for s in raw_data
            if len(s.weeks_data) == config.training_week_len + 1
        ])

        # 3. Two-stage grouped splitting (e.g., Train:Val:Test = 8:1:1)
        train_ratio: int
        val_ratio: int
        test_ratio: int
        train_ratio, val_ratio, test_ratio = config.split_ratios
        total_ratio: int = sum(config.split_ratios)

        # Step A: Split Test set
        x_train_val: NDArray[Any]
        y_train_val: NDArray[Any]
        groups_train_val: NDArray[Any]
        x_test: NDArray[Any]
        y_test: NDArray[Any]
        x_train_val, y_train_val, groups_train_val, x_test, y_test = self._split_by_group(
            x=x_to_split,
            y=y_to_split,
            groups=movie_ids,
            test_size=(test_ratio / total_ratio),
            random_state=config.random_state
        )

        # Step B: Split Validation set from the remainder
        x_train: NDArray[Any]
        y_train: NDArray[Any]
        x_val: NDArray[Any]
        y_val: NDArray[Any]
        x_train, y_train, _, x_val, y_val = self._split_by_group(
            x=x_train_val,
            y=y_train_val,
            groups=groups_train_val,
            test_size=(val_ratio / (train_ratio + val_ratio)),
            random_state=config.random_state
        )

        # 4. Wrap results
        split_data: SplitDataset = SplitDataset(
            x_train=x_train, y_train=y_train,
            x_val=x_val, y_val=y_val,
            x_test=x_test, y_test=y_test
        )

        # 5. Post-processing (Scaling)
        return self._post_process_splits(split_data=split_data, config=config)

    def _convert_weeks_to_numerical_sequence(self, weeks: list[WeekData]) -> list[list[int | float]]:
        """
        Converts a list of WeekData objects into a numerical sequence.

        :param weeks: A list of WeekData objects to be converted.
        :return: A list of lists, where each inner list represents the numerical features for a week.
        """
        return [self._feature_class.from_week_data(week=week).as_numerical_list() for week in weeks]

    @override
    def process_for_prediction(self, single_input: MovieData, config: BoxOfficeDataConfig) -> NDArray[float32]:
        """
        Processes a single movie's data for prediction.

        :param single_input: The raw movie data for prediction.
        :param config: The data configuration.
        :return: The processed feature array ready for the model's predict method.
        :raises ValueError: If the scaler is not set or input data length is insufficient.
        """
        if not self._scaler:
            raise ValueError("Scaler has not been set. Please train first or load an artifact.")

        box_office_history: list[BoxOffice] = single_input.box_office
        training_week_len: int = config.training_week_len
        if len(box_office_history) < training_week_len:
            raise ValueError(
                f"Input movie '{single_input.name}' has only {len(box_office_history)} weeks of data, "
                f"but the model requires {training_week_len} weeks."
            )
        latest_box_office_weeks: list[BoxOffice] = box_office_history[-training_week_len:]

        latest_weeks_data: list[WeekData] = WeekData.create_multiple_from_source_variable(
            weeks_data_source=latest_box_office_weeks,
            public_reviews_master_source=single_input.public_reviews,
            movie_id=single_input.id
        )
        numerical_sequence: list[list[int | float]] = \
            self._convert_weeks_to_numerical_sequence(weeks=latest_weeks_data)

        if len(numerical_sequence) != training_week_len:
            raise ValueError("Failed to create a numerical sequence of the required length.")

        unscaled_array: NDArray[float32] = expand_dims(
            a=array(numerical_sequence, dtype=float32), axis=0
        )
        scaled_array: NDArray[float32] = self._scale_feature_in_sequences(sequences=unscaled_array)
        return scaled_array

    @abstractmethod
    def _scale_feature_in_sequences(self, sequences: NDArray[float32]) -> NDArray[float32]:
        """
        Scales the numerical features within the input sequences.

        :param sequences: The numerical sequences to be scaled.
        :return: The scaled sequences.
        """
        pass

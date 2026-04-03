from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Optional, TypeAlias, TypeVar, Type

from numpy import array, expand_dims, float32
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.model_selection import GroupShuffleSplit
from typing_extensions import override

from src.data_handling.box_office import BoxOffice
from src.models.base.base_data_processor import ProcessedPredictionDataType, ProcessedTrainingDataType
from src.data_handling.dataset import BaseDataset
from src.data_handling.movie_collections import MovieData, MovieSessionData, WeekData
from src.models.base.data_splitter import SplitDataset, X_Type, Y_Type
from src.models.base.gradient_data_processor import GradientDataConfig
from src.models.base.gradient_data_processor import GradientDataProcessor

BoxOfficeDataSource: TypeAlias = BaseDataset
BoxOfficeTrainingRawData: TypeAlias = list[MovieSessionData]
BoxOfficePredictionRawData: TypeAlias = MovieData


class BoxOfficeDataConfig(GradientDataConfig):
    """
    Configuration for the Box Office Model's data processing.

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
        Initializes the BoxOfficeDataConfig.

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
class BoxOfficeFeature(ABC):

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
    票房序列資料的通用處理器。

    強制實作 load_raw_data (封裝 get_movie_sessions)
    以及 process_for_training (封裝 GroupShuffleSplit)。
    """

    _feature_class: Type[FeatureClassType]  # Store the concrete Feature class
    scaler: Optional[ScalerType]  # Store the concrete Scaler instance

    @override
    def __init__(self,
                 feature_class: Type[FeatureClassType],
                 model_artifacts_path: Optional[Path] = None):
        """
        Initializes the BoxOfficeSessionDataProcessor.

        :param feature_class: The concrete Feature class (e.g., BoxOfficeRegressionFeature)
                              that this processor will use for feature extraction.
        :param model_artifacts_path: Path to the directory for model artifacts.
        """
        super().__init__(model_artifacts_path=model_artifacts_path)
        self._feature_class = feature_class
        self.scaler = None
        self.load_artifacts()

    @override
    def load_raw_data(
        self, source: BoxOfficeDataSource, config: Optional[BoxOfficeDataConfig] = None
    ) -> BoxOfficeTrainingRawData:
        """
        Loads and processes raw movie data into fixed-length sessions using a Dataset instance.

        This method leverages the dataset's `get_movie_sessions` method, which
        efficiently loads and processes data based on its underlying storage
        (DB or YAML).

        :param source: The BaseDataset instance to load data from.
        :param config: The data configuration, used to determine the session length.
        :returns: A list of MovieSessionData objects.
        :raises ValueError: If the config is not provided.
        """
        if config is None:
            raise ValueError("BoxOfficeRegressionDataConfig is required to determine session length.")

        self.logger.debug(f"Loading movie sessions from dataset: '{source.name}'")

        # The number of weeks needed is the training length + 1 for the target week
        number_of_weeks = config.training_week_len + 1
        sessions: list[MovieSessionData] = source.get_movie_sessions(number_of_weeks=number_of_weeks)

        if not sessions:
            self.logger.warning(f"No movie sessions loaded from dataset: {source.name}")

        return sessions

    @override
    def process_for_training(
        self, raw_data: BoxOfficeTrainingRawData, config: BoxOfficeDataConfig
    ) -> ProcessedTrainingDataType:
        """
        實作與 latest.ipynb 一致的切分邏輯：確保電影 ID 級別的隔離。
        """

        # 1. 預備資料 (包含 PR 門檻計算與特徵提取)
        # 此處 X 的 shape 為 (N, SEQ_LEN, Features)
        x_to_split, y_to_split = self._prepare_for_split(raw_data=raw_data, config=config)

        # 2. 提取分組標籤 (電影 ID)
        # 這裡必須與 X 樣本對應，只有長度符合 SEQ_LEN + 1 的 Session 才會被轉換為樣本
        movie_ids = array([
            s.id for s in raw_data
            if len(s.weeks_data) == config.training_week_len + 1
        ])

        # 3. 兩階段分組切分 (模擬 split_ratios，例如 8:1:1)
        train_ratio, val_ratio, test_ratio = config.split_ratios

        # 第一步：切出測試集
        gss_test = GroupShuffleSplit(n_splits=1, test_size=(test_ratio / sum(config.split_ratios)),
                                     random_state=config.random_state)
        train_val_idx, test_idx = next(gss_test.split(x_to_split, y_to_split, groups=movie_ids))

        x_train_val, y_train_val = x_to_split[train_val_idx], y_to_split[train_val_idx]
        groups_train_val = movie_ids[train_val_idx]

        x_test, y_test = x_to_split[test_idx], y_to_split[test_idx]

        # 第二步：從剩下的資料中切出驗證集
        gss_val = GroupShuffleSplit(n_splits=1, test_size=(val_ratio / (train_ratio + val_ratio)),
                                    random_state=config.random_state)
        train_idx, val_idx = next(gss_val.split(x_train_val, y_train_val, groups=groups_train_val))

        x_train, y_train = x_train_val[train_idx], y_train_val[train_idx]
        x_val, y_val = x_train_val[val_idx], y_train_val[val_idx]

        # 4. 封裝結果
        split_data = SplitDataset(
            x_train=x_train, y_train=y_train,
            x_val=x_val, y_val=y_val,
            x_test=x_test, y_test=y_test
        )

        # 5. 後處理 (對應筆記本步驟 8: StandardScaler)
        return self._post_process_splits(split_data=split_data, config=config)

    def _convert_weeks_to_numerical_sequence(self, weeks: list[WeekData]) -> list[list[int | float]]:
        """
        Converts a list of WeekData objects into a numerical sequence.

        Each WeekData object is transformed into a list of features:
         [box_office, avg_sentiment, reply_count, total_positive_reply_count, total_negative_reply_count].

        :param weeks: A list of WeekData objects to be converted.
        :returns: A list of lists, where each inner list represents the numerical features for a week.
        """

        return [self._feature_class.from_week_data(week).as_numerical_list() for week in weeks]

    @override
    def process_for_prediction(
        self, single_input: MovieData, config: BoxOfficeDataConfig
    ) -> NDArray[float32]:
        """
        Processes a single movie's data for prediction using generic Feature and Scaler types.
        """
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

        latest_weeks_data: list[WeekData] = WeekData.create_multiple_from_source_variable(
            weeks_data_source=latest_box_office_weeks,
            public_reviews_master_source=single_input.public_reviews,
            movie_id=single_input.id
        )
        numerical_sequence: list[list[int | float]] = \
            self._convert_weeks_to_numerical_sequence(weeks=latest_weeks_data)  # 使用實例方法

        if len(numerical_sequence) != training_week_len:
            raise ValueError("Failed to create a numerical sequence of the required length.")

        unscaled_array: NDArray[float32] = expand_dims(
            array(numerical_sequence, dtype=float32), axis=0
        )
        scaled_array: NDArray[float32] = self._scale_feature_in_sequences(sequences=unscaled_array)
        return scaled_array

    @abstractmethod
    def _scale_feature_in_sequences(self, sequences: NDArray[float32]) -> NDArray[float32]:
        pass

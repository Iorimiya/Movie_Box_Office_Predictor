from abc import abstractmethod
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

from numpy.typing import NDArray
from typing_extensions import override

from src.core.logging_manager import LoggingManager
from src.models.base.base_data_processor import (
    BaseDataConfig,
    BaseDataProcessor,
    DataConfigType,
    PredictionRawDataType,
    ProcessedPredictionDataType,
    ProcessedTrainingDataType,
    RawDataSourceType,
    TrainingRawDataType
)
from src.models.base.data_splitter import DatasetSplitter, SplitDataset, X_Type, Y_Type


@dataclass(frozen=True, kw_only=True)
class SplittingConfig:
    """
    A mixin dataclass for configurations that involve data splitting.

    Provides common attributes for splitting data into train, validation, and test sets.
    These fields are optional as not all modes (e.g., Box Office Regression) require them.

    :ivar split_ratios: The ratio for splitting data.
    :ivar random_state: The seed for the random number generator.
    """
    split_ratios: tuple[int, int, int]
    random_state: int


class GradientDataConfig(BaseDataConfig):
    """
    Configuration for gradient-based training processes.

    It composes a SplittingConfig object internally.

    :ivar _splitting: The internal splitting configuration.
    """
    _splitting: Optional[SplittingConfig]

    def __init__(
        self,
        *,
        split_ratios: Optional[tuple[int, int, int]] = None,
        random_state: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """
        Initializes the GradientDataConfig.

        :param split_ratios: The ratio for splitting data (train, val, test).
        :param random_state: The seed for the random number generator.
        :param kwargs: Additional keyword arguments passed to the base class.
        :raises ValueError: If only one of ``split_ratios`` or ``random_state`` is provided.
        """
        super().__init__(**kwargs)
        if split_ratios is not None and random_state is not None:
            self._splitting: Optional[SplittingConfig] = SplittingConfig(
                split_ratios=split_ratios,
                random_state=random_state
            )
        elif split_ratios is not None or random_state is not None:
            raise ValueError("Both 'split_ratios' and 'random_state' must be provided together.")
        else:
            self._splitting: Optional[SplittingConfig] = None

    @property
    def split_ratios(self) -> Optional[tuple[int, int, int]]:
        return self._splitting.split_ratios if self._splitting is not None else None

    @property
    def random_state(self) -> Optional[int]:
        return self._splitting.random_state if self._splitting is not None else None

    @property
    def splittable(self) -> bool:
        return True if self._splitting is not None else False


GradientDataConfigType = TypeVar('GradientDataConfigType', bound=GradientDataConfig)


class GradientDataProcessor(
    BaseDataProcessor[
        RawDataSourceType,
        TrainingRawDataType,
        ProcessedTrainingDataType,
        PredictionRawDataType,
        ProcessedPredictionDataType,
        GradientDataConfigType
    ],
    Generic[
        RawDataSourceType,
        TrainingRawDataType,
        ProcessedTrainingDataType,
        PredictionRawDataType,
        ProcessedPredictionDataType,
        GradientDataConfigType,
        X_Type,
        Y_Type
    ]
):
    """
    An abstract base class for data processors that support evaluation on a full, unsplit dataset.

    This class extends :class:`~.BaseDataProcessor` by introducing a structured
    workflow for training data processing and adding an abstract method
    `process_for_evaluation`. This establishes a contract for processors used
    in evaluation contexts that require handling a complete dataset as a single
    test set. It also provides a template method pattern for processing training
    data, separating pre-split, splitting, and post-split logic.

    :ivar _logger: A logger instance for logging processing activities.
    :ivar _splitter: A :class:`~.DatasetSplitter` instance for splitting data.
    """
    _logger: Logger
    _splitter: DatasetSplitter[X_Type, Y_Type]

    @override
    def __init__(self, model_artifacts_path: Optional[Path] = None) -> None:
        """
        Initializes the EvaluableDataProcessor.

        This also initializes a shared logger and a generic :class:`~.DatasetSplitter`.

        :param model_artifacts_path: Path to the directory where model artifacts
                                     (like a scaler or tokenizer) are or will be stored.
        """
        super().__init__(model_artifacts_path=model_artifacts_path)
        self._logger: Logger = LoggingManager().get_logger(name='machine_learning')
        self._splitter: DatasetSplitter[X_Type, Y_Type] = DatasetSplitter[X_Type, Y_Type](logger=self._logger)

    @abstractmethod
    def _prepare_for_split(self, raw_data: TrainingRawDataType, config: DataConfigType) -> tuple[X_Type, Y_Type]:
        """
        Pre-processes raw data into feature (x) and label (y) arrays ready for splitting.

        This abstract method must be implemented by subclasses to perform initial
        transformations on the raw data, converting it into a numerical format
        (e.g., NumPy arrays) suitable for the data splitter.

        :param raw_data: The raw data loaded from the source.
        :param config: The data processing configuration.
        :return: A tuple containing the feature array (x) and the label array (y).
        """
        pass

    @abstractmethod
    def _post_process_splits(
        self, split_data: SplitDataset[X_Type, Y_Type], config: DataConfigType
    ) -> ProcessedTrainingDataType:
        """
        Performs final processing on the data after it has been split.

        This is where model-specific tools like tokenizers or scalers should be
        fitted (on the training set) and applied to all splits.

        :param split_data: The TypedDict containing the train, validation, and test splits.
        :param config: The data processing configuration.
        :return: The final, fully processed data ready for model training.
        """
        pass

    @override
    def process_for_training(
        self, raw_data: TrainingRawDataType, config: GradientDataConfigType
    ) -> ProcessedTrainingDataType:
        """
        A template method that processes raw data for model training.

        It follows a fixed workflow:
        1. Prepare data for splitting (`_prepare_for_split`).
        2. Split the data into train, val, and test sets.
        3. Perform post-split processing (`_post_process_splits`).

        :param raw_data: The raw data loaded by `load_raw_data`.
        :param config: A configuration object containing parameters for the training process,
                       such as split ratios and random state.
        :return: The processed data, ready to be fed into a model.
        :raises ValueError: If `split_ratios` in the config is not provided.
        """
        self._logger.debug("Starting data processing for training.")

        if config.split_ratios is None:
            raise ValueError(
                "The 'split_ratios' parameter must be provided in the configuration for the training process."
            )

        # Delegate pre-split processing to subclass
        self._logger.debug("Preparing data for splitting...")
        x_to_split: X_Type
        y_to_split: Y_Type
        x_to_split, y_to_split = self._prepare_for_split(raw_data=raw_data, config=config)

        # Perform the split (common logic)
        self._logger.debug("Splitting data into train, validation, and test sets...")
        # noinspection PyTypeChecker
        split_data: SplitDataset[X_Type, Y_Type] = self._splitter.split(
            x_data=x_to_split,
            y_data=y_to_split,
            split_ratios=config.split_ratios,
            random_state=config.random_state,
            shuffle=True
        )

        # Delegate post-split processing to subclass
        self._logger.debug("Performing post-split processing (scaling/tokenizing)...")
        processed_data: ProcessedTrainingDataType = self._post_process_splits(split_data=split_data, config=config)

        self._logger.debug("Data processing for training finished.")
        return processed_data

    # noinspection PyTypeHints
    @abstractmethod
    def process_for_evaluation(
        self, raw_data: TrainingRawDataType, config: Optional[GradientDataConfigType]
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """
        Processes a full raw dataset for evaluation without splitting it.

        :param raw_data: The raw data to be processed for evaluation.
        :param config: An optional configuration object containing necessary parameters.
        :return: A tuple containing the full processed features (x) and labels (y).
        """
        pass

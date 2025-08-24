from abc import abstractmethod
from logging import Logger
from pathlib import Path
from typing import Generic, Optional

from numpy.typing import NDArray
from typing_extensions import override

from src.core.logging_manager import LoggingManager
from src.models.base.base_data_processor import (
    BaseDataProcessor,
    RawDataSourceType,
    RawDataType,
    ProcessedTrainingDataType,
    PredictionDataType,
    ProcessedPredictionDataType,
    DataConfigType
)
from src.models.base.data_splitter import DatasetSplitter, SplitDataset, X_Type, Y_Type


class EvaluableDataProcessor(
    BaseDataProcessor[
        RawDataSourceType,
        RawDataType,
        ProcessedTrainingDataType,
        PredictionDataType,
        ProcessedPredictionDataType,
        DataConfigType
    ],
    Generic[
        RawDataSourceType,
        RawDataType,
        ProcessedTrainingDataType,
        PredictionDataType,
        ProcessedPredictionDataType,
        DataConfigType,
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

    :ivar logger: A logger instance for logging processing activities.
    :ivar splitter: A :class:`~.DatasetSplitter` instance for splitting data.
    """
    logger: Logger
    splitter: DatasetSplitter[X_Type, Y_Type]

    @override
    def __init__(self, model_artifacts_path: Optional[Path] = None) -> None:
        """
        Initializes the EvaluableDataProcessor.

        This also initializes a shared logger and a generic :class:`~.DatasetSplitter`.

        :param model_artifacts_path: Path to the directory where model artifacts
                                     (like a scaler or tokenizer) are or will be stored.
        """
        super().__init__(model_artifacts_path=model_artifacts_path)
        self.logger: Logger = LoggingManager().get_logger('machine_learning')
        self.splitter: DatasetSplitter[X_Type, Y_Type] = DatasetSplitter[X_Type, Y_Type](logger=self.logger)

    @abstractmethod
    def _prepare_for_split(self, raw_data: RawDataType, config: DataConfigType) -> tuple[X_Type, Y_Type]:
        """
        Pre-processes raw data into feature (x) and label (y) arrays ready for splitting.

        This abstract method must be implemented by subclasses to perform initial
        transformations on the raw data, converting it into a numerical format
        (e.g., NumPy arrays) suitable for the data splitter.

        :param raw_data: The raw data loaded from the source.
        :param config: The data processing configuration.
        :returns: A tuple containing the feature array (x) and the label array (y).
        """
        pass

    @abstractmethod
    def _post_process_splits(self, split_data: SplitDataset[X_Type, Y_Type],
                             config: DataConfigType) -> ProcessedTrainingDataType:
        """
        Performs final processing on the data after it has been split.

        This is where model-specific tools like tokenizers or scalers should be
        fitted (on the training set) and applied to all splits.

        :param split_data: The TypedDict containing the train, validation, and test splits.
        :param config: The data processing configuration.
        :returns: The final, fully processed data ready for model training.
        """
        pass

    @override
    def process_for_training(self, raw_data: RawDataType, config: DataConfigType) -> ProcessedTrainingDataType:
        """
        A template method that processes raw data for model training.

        It follows a fixed workflow:
        1. Prepare data for splitting (`_prepare_for_split`).
        2. Split the data into train, val, and test sets.
        3. Perform post-split processing (`_post_process_splits`).

        :param raw_data: The raw data loaded by `load_raw_data`.
        :param config: A configuration object containing parameters for the training process,
                       such as split ratios and random state.
        :returns: The processed data, ready to be fed into a model.
        :raises ValueError: If the sum of `split_ratios` in the config is zero.
        """
        self.logger.info("--- Starting data processing for training ---")

        # Delegate pre-split processing to subclass
        self.logger.info("Step 1: Preparing data for splitting...")
        x_to_split, y_to_split = self._prepare_for_split(raw_data=raw_data, config=config)

        # Perform the split (common logic)
        self.logger.info("Step 2: Splitting data into train, validation, and test sets...")
        split_data: SplitDataset[X_Type, Y_Type] = self.splitter.split(
            x_data=x_to_split,
            y_data=y_to_split,
            split_ratios=config.split_ratios,
            random_state=config.random_state,
            shuffle=True  # Assuming shuffle is standard for training
        )

        # Delegate post-split processing to subclass
        self.logger.info("Step 3: Performing post-split processing (scaling/tokenizing)...")
        processed_data: ProcessedTrainingDataType = self._post_process_splits(split_data=split_data, config=config)

        self.logger.info("--- Data processing for training finished ---")
        return processed_data

    @abstractmethod
    def process_for_evaluation(self, raw_data: RawDataType, config: Optional[DataConfigType]) \
        -> tuple[NDArray[any], NDArray[any]]:
        """
        Processes a full raw dataset for evaluation without splitting it.


        :param raw_data: The raw data to be processed for evaluation.
        :param config: An optional configuration object containing necessary parameters.
        :returns: A tuple containing the full processed features (x) and labels (y).
        """
        pass

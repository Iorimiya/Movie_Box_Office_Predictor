from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Generic, Optional, TypeVar

from numpy.typing import NDArray

from src.core.logging_manager import LoggingManager
from src.data_handling.file_io import PickleFile
from src.models.base.base_data_processor import BaseDataProcessor
from src.models.base.base_model_core import BaseModelCore
from src.models.base.base_pipeline import BaseTrainingPipeline
from src.models.base.keras_setup import keras_base

History = keras_base.callbacks.History


@dataclass(frozen=True)
class BaseEvaluationConfig:
    """
    A base dataclass for model evaluation configurations.

    Defines common attributes required for evaluating a model, such as the
    model's identity, the dataset to use, and flags for which metrics to compute.

    :ivar model_id: The unique identifier for the model series.
    :ivar model_epoch: The specific training epoch of the model to evaluate.
    :ivar dataset_name: The name of the dataset file to use for evaluation.
    :ivar evaluate_on_full_dataset: If True, evaluates on the entire dataset
                                    without splitting. If False, reproduces
                                    the original test split.
    :ivar split_ratios: The train/val/test split ratios. Required only for
                        reproducibility mode.
    :ivar random_state: The random seed for data splitting. Required only for
                        reproducibility mode.
    :ivar calculate_loss: Flag to calculate loss on the test set.
    :ivar calculate_f1_score: Flag to calculate F1-score on the test set.
    :ivar f1_average_method: The averaging method for F1 score calculation.
    """
    model_id: str
    model_epoch: int
    dataset_name: str
    evaluate_on_full_dataset: bool
    split_ratios: Optional[tuple[int, int, int]]
    random_state: Optional[int]
    calculate_loss: bool
    calculate_f1_score: bool
    f1_average_method: str


@dataclass(frozen=True)
class BaseEvaluationResult:
    """
    A base dataclass for structured evaluation results.

    Defines the common attributes that all model evaluation results must have.

    :ivar model_id: The unique identifier for the evaluated model series.
    :ivar model_epoch: The specific epoch of the evaluated model.
    :ivar test_loss: The loss calculated on the test set.
    :ivar f1_score: The F1-score calculated on the test set.
    :ivar training_loss_history: A list of training loss values from the original run.
    :ivar validation_loss_history: A list of validation loss values from the original run.
    """
    model_id: str
    model_epoch: int
    test_loss: Optional[float]
    f1_score: Optional[float]
    training_loss_history: list[float]
    validation_loss_history: list[float]


DataProcessorType = TypeVar('DataProcessorType', bound=BaseDataProcessor)
ModelCoreType = TypeVar('ModelCoreType', bound=BaseModelCore)
EvaluationConfigType = TypeVar('EvaluationConfigType', bound=BaseEvaluationConfig)
EvaluationResultType = TypeVar('EvaluationResultType', bound=BaseEvaluationResult)


class BaseEvaluator(
    Generic[DataProcessorType, ModelCoreType, EvaluationConfigType, EvaluationResultType],
    ABC
):
    """
    An abstract base class for a model evaluator.

    This class defines a standardized workflow for evaluating a trained model.
    It is responsible for loading a model and its associated artifacts,
    processing a dataset for evaluation, and computing performance metrics.
    The specific logic is delegated to subclasses.

    :ivar logger: A logger instance for logging evaluation progress.
    """
    logger: Logger

    def __init__(self) -> None:
        """
        Initializes the BaseEvaluator.
        """
        self.logger = LoggingManager().get_logger('machine_learning')

    @abstractmethod
    def _setup_components(
        self, model_id: str, model_epoch: int
    ) -> tuple[DataProcessorType, ModelCoreType, Path]:
        """
        Sets up and loads the necessary data processor and model core.

        :param model_id: The unique identifier for the model series.
        :param model_epoch: The specific training epoch of the model to load.
        :returns: A tuple containing the initialized data processor, model core,
                  and the path to the model artifacts directory.
        """
        pass

    @abstractmethod
    def _prepare_test_data(
        self, data_processor: DataProcessorType, config: EvaluationConfigType
    ) -> tuple[NDArray[any], NDArray[any]]:
        """
        Loads and processes data to retrieve the test set for evaluation.

        :param data_processor: The initialized data processor.
        :param config: The configuration object for the evaluation run.
        :returns: A tuple containing the evaluation features (x_eval) and labels (y_eval).
        """
        pass

    @abstractmethod
    def _calculate_metrics(
        self,
        model_core: ModelCoreType,
        data_processor: DataProcessorType,
        x_test: NDArray[any],
        y_test: NDArray[any],
        config: EvaluationConfigType
    ) -> dict[str, Optional[float]]:
        """
        Calculates all requested metrics based on the configuration.

        This is the core calculation hook for subclasses. It should return a
        dictionary containing all computed metrics.

        :param model_core: The loaded model core.
        :param data_processor: The loaded data processor (needed for things like scalers).
        :param x_test: The test features.
        :param y_test: The test labels.
        :param config: The evaluation configuration object.
        :returns: A dictionary mapping metric names to their calculated values.
        """
        pass

    @abstractmethod
    def _compile_final_result(
        self,
        config: EvaluationConfigType,
        metrics: dict[str, Optional[float]],
        training_history: list[float],
        validation_history: list[float]
    ) -> EvaluationResultType:
        """
        Compiles the final, structured result object for the evaluation.

        :param config: The original evaluation configuration.
        :param metrics: A dictionary of calculated metrics from `_calculate_metrics`.
        :param training_history: The loaded training loss history.
        :param validation_history: The loaded validation loss history.
        :returns: The final, model-specific evaluation result object.
        """
        pass

    # --- 變成具體方法的 load_training_history ---
    def load_training_history(self, history_file_path: Path) -> tuple[list[float], list[float]]:
        """
        Loads the training and validation loss history from a pickle file.

        :param history_file_path: The path to the history file.
        :returns: A tuple containing the training loss list and validation loss list.
        """
        self.logger.info(f"Step 2: Loading training history from '{history_file_path}'...")
        history: History = PickleFile(path=history_file_path).load()
        training_loss: list[float] = history.history.get('loss', [])
        validation_loss: list[float] = history.history.get('val_loss', [])
        return training_loss, validation_loss

    def run(self, config: EvaluationConfigType) -> EvaluationResultType:
        """
        Executes the standardized evaluation pipeline.

        This template method orchestrates the evaluation process by calling a
        series of abstract methods that must be implemented by subclasses.

        :param config: The configuration object for the evaluation run.
        :returns: A structured result object containing all evaluation metrics.
        """
        self.logger.info(
            f"--- Starting evaluation for model '{config.model_id}' at epoch {config.model_epoch} ---"
        )

        # Setup components (delegated to subclass)
        data_processor, model_core, artifacts_path = self._setup_components(
            model_id=config.model_id, model_epoch=config.model_epoch
        )

        # Load history (common logic)
        history_path = artifacts_path / BaseTrainingPipeline.HISTORY_FILE_NAME
        training_loss, validation_loss = self.load_training_history(history_file_path=history_path)

        # Prepare test data (delegated to subclass)
        x_test, y_test = self._prepare_test_data(data_processor=data_processor, config=config)

        # Calculate all metrics (delegated to subclass)
        self.logger.info("Step 4: Calculating requested metrics on the test set...")
        calculated_metrics = self._calculate_metrics(
            model_core=model_core,
            data_processor=data_processor,
            x_test=x_test,
            y_test=y_test,
            config=config
        )

        # Compile final result (delegated to subclass)
        self.logger.info("Step 5: Compiling final evaluation results...")
        final_result = self._compile_final_result(
            config=config,
            metrics=calculated_metrics,
            training_history=training_loss,
            validation_history=validation_loss
        )

        self.logger.info(f"--- Evaluation finished for model '{config.model_id}' ---")
        return final_result

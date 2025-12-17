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
from src.models.base.display import ClassificationSummaryMixin
from src.models.base.keras_setup import keras_base
from src.utilities.metrics import RegressionReportDict, ClassificationReportDict

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
    A universal base dataclass for any model evaluation result.

    This class is purified to only contain attributes that are guaranteed
    to exist for any model evaluation, regardless of the model type or task.

    :ivar model_id: The unique identifier for the model series.
    :ivar model_epoch: The specific training epoch of the model evaluated.
    """
    model_id: str
    model_epoch: int


@dataclass(frozen=True)
class GradientBasedEvaluationResult(BaseEvaluationResult):
    """
    A base dataclass for evaluation results of models trained via gradient descent.

    This intermediate class captures the common artifacts produced by the
    training process of differentiable models, specifically the history of
    loss values over epochs. It serves as a bridge between the generic base
    result and task-specific results (regression or classification).

    :ivar training_loss_history: A list of training loss values recorded at each epoch.
    :ivar validation_loss_history: A list of validation loss values recorded at each epoch.
    """
    training_loss_history: list[float]
    validation_loss_history: list[float]


@dataclass(frozen=True)
class RegressionEvaluationResult(GradientBasedEvaluationResult):
    """
    A base dataclass for evaluation results of trainable regression models.

    It inherits the universal identifiers and adds fields common to models
    trained with a loss function, and holds a report for regression-specific
    test metrics.

    :ivar regression_report: A structured dictionary containing regression metrics
                             like MSE, MAE, and R² score, calculated on the test set.
    """
    regression_report: Optional[RegressionReportDict]


@dataclass(frozen=True)
class ClassificationEvaluationResult(GradientBasedEvaluationResult, ClassificationSummaryMixin):
    """
    A base dataclass for evaluation results of classification models.

    It inherits universal identifiers from BaseEvaluationResult and summary
    generation behaviors from ClassificationSummaryMixin.

    :ivar classification_report: A structured dictionary containing metrics
                                 like accuracy, F1-score, and confusion matrix.
    """
    classification_report: Optional[ClassificationReportDict]

    def get_summary_string(self) -> str:
        """
        Generates a formatted summary string for the classification result.
        """
        lines: list[str] = [f"Evaluation Summary for Model '{self.model_id}' (Epoch {self.model_epoch}):"]
        if self.classification_report:
            lines.append(f"  - Accuracy: {self.classification_report['accuracy']:.2%}")
            lines.append(f"  - F1-Score: {self.classification_report['f1_score']:.4f}")
            lines.append("\n" + self.classification_report['report_string'])

            matrix_str:str = self.format_confusion_matrix_string(
                matrix=self.classification_report['confusion_matrix'],
                names=self.classification_report['target_names'] or []
            )
            lines.append(f"\nConfusion Matrix:\n{matrix_str}")
        else:
            lines.append("  - No classification metrics were calculated.")

        return "\n".join(lines)


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
    def _create_evaluation_result(
            self,
            *,
            config: EvaluationConfigType,
            model_core: ModelCoreType,
            data_processor: DataProcessorType,
            x_test: NDArray[any],
            y_test: NDArray[any],
            training_history: list[float],
            validation_history: list[float]
    ) -> EvaluationResultType:
        """
        Creates the final, structured result object for the evaluation.

        This method is the primary hook for subclasses. It should orchestrate
        all necessary metric calculations and compile the final, model-specific
        evaluation result object, often by calling a factory method on the
        result class itself (e.g., `Result.create(...)`).

        :param config: The original evaluation configuration.
        :param model_core: The loaded model core.
        :param data_processor: The loaded data processor.
        :param x_test: The prepared test features.
        :param y_test: The prepared test labels.
        :param training_history: The loaded training loss history.
        :param validation_history: The loaded validation loss history.
        :returns: The final, model-specific evaluation result object.
        """
        pass

    def load_training_history(self, history_file_path: Path) -> tuple[list[float], list[float]]:
        """
        Loads the training and validation loss history from a pickle file.

        This method now correctly handles loading a dictionary that was saved
        from a Keras History object's `.history` attribute.

        :param history_file_path: The path to the history file.
        :returns: A tuple containing the training loss list and validation loss list.
        :raises FileNotFoundError: If the history file does not exist.
        """
        self.logger.info(f"Loading training history from '{history_file_path}'...")
        if not history_file_path.exists():
            raise FileNotFoundError(f"Training history file not found at: {history_file_path}")

        history_dict: dict[str, list[float]] = PickleFile(path=history_file_path).load()

        # Directly access the keys from the loaded dictionary.
        training_loss: list[float] = history_dict.get('loss', [])
        validation_loss: list[float] = history_dict.get('val_loss', [])

        return training_loss, validation_loss

    def run(self, config: EvaluationConfigType) -> EvaluationResultType:
        """
        Executes the standardized evaluation pipeline.

        This template method orchestrates the evaluation process by preparing
        all components and data, then delegating the final evaluation step
        to the `_create_evaluation_result` method.

        :param config: The configuration object for the evaluation run.
        :returns: A structured result object containing all evaluation metrics.
        """
        self.logger.info(
            f"--- Starting evaluation for model '{config.model_id}' at epoch {config.model_epoch} ---"
        )

        data_processor: DataProcessorType
        model_core:ModelCoreType
        artifacts_path:Path
        # Setup components (delegated to subclass)
        data_processor, model_core, artifacts_path = self._setup_components(
            model_id=config.model_id, model_epoch=config.model_epoch
        )

        # Load history (common logic)
        history_path:Path = artifacts_path / BaseTrainingPipeline.HISTORY_FILE_NAME
        training_loss:list[float]
        validation_loss:list[float]
        training_loss, validation_loss = self.load_training_history(history_file_path=history_path)

        # Prepare test data (delegated to subclass)
        x_test: NDArray[any]
        y_test: NDArray[any]
        x_test, y_test = self._prepare_test_data(data_processor=data_processor, config=config)

        # Delegate the entire evaluation and compilation to the subclass
        self.logger.info("Creating final evaluation result...")
        final_result:EvaluationResultType = self._create_evaluation_result(
            config=config,
            model_core=model_core,
            data_processor=data_processor,
            x_test=x_test,
            y_test=y_test,
            training_history=training_loss,
            validation_history=validation_loss
        )

        self.logger.info(f"--- Evaluation finished for model '{config.model_id}' ---")
        return final_result

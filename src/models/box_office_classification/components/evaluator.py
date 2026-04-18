from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Any, Optional, TypeAlias

from numpy import float32, int_
from numpy.typing import NDArray
from typing_extensions import override

from src.core.project_config import ProjectModelType
from src.models.base.evaluation import (
    ClassificationEvaluationConfig,
    ClassificationEvaluationResult
)
from src.models.base.keras_setup import keras_base
from src.models.box_office_classification.components.data_processor import BoxOfficeClassificationDataProcessor
from src.models.box_office_classification.components.model_core import (
    BoxOfficeClassificationModelCore,
    BoxOfficeClassificationPredictParams,
)
from src.models.box_office_common.box_office_evaluator import (
    BoxOfficeBaseEvaluationConfig,
    BoxOfficeBaseEvaluator,
)
from src.utilities.metrics import ClassificationReportDict, MultiClassClassificationMetricsCalculator

# noinspection PyUnresolvedReferences
History: TypeAlias = keras_base.callbacks.History


class BoxOfficeClassificationEvaluationConfig(BoxOfficeBaseEvaluationConfig, ClassificationEvaluationConfig):
    """
    Configuration for running a Box Office Classification Model evaluation.

    Inherits common evaluation parameters from BoxOfficeBaseEvaluationConfig and ClassificationEvaluationConfig.

    :ivar box_office_thresholds: The PR thresholds (e.g., (50, 80)) used for classification.
    """

    def __init__(
        self,
        *,
        calculate_classification_metrics: bool,
        box_office_thresholds: tuple[int, ...],
        f1_average_method: str = 'macro',
        **kwargs: Any
    ):
        """
        Initializes the BoxOfficeClassificationEvaluationConfig.

        :param calculate_classification_metrics: Flag to enable classification-based metrics.
        :param box_office_thresholds: The PR thresholds (e.g., (50, 80)) used for classification.
        :param f1_average_method: The averaging method for F1 score calculation.
        :param kwargs: Additional keyword arguments passed to the base class.
        """
        super().__init__(
            calculate_classification_metrics=calculate_classification_metrics,
            f1_average_method=f1_average_method,
            **kwargs
        )
        self.box_office_thresholds: tuple[int, ...] = box_office_thresholds


@dataclass(frozen=True)
class BoxOfficeClassificationEvaluationResult(ClassificationEvaluationResult):
    """
    The specific evaluation result for the Box Office Classification Model.

    This class encapsulates metrics for a multi-class classification task.
    """

    @classmethod
    def create(
        cls,
        *,
        config: "BoxOfficeClassificationEvaluationConfig",
        model_core: "BoxOfficeClassificationModelCore",
        x_test: NDArray[float32],
        y_test: NDArray[int_],
        training_history: list[float],
        validation_history: list[float],
        logger: Logger
    ) -> "BoxOfficeClassificationEvaluationResult":
        """
        Factory method to create a complete evaluation result.

        :param config: The evaluation configuration.
        :param model_core: The trained model core.
        :param x_test: The test features.
        :param y_test: The test labels (class indices or one-hot).
        :param training_history: The training loss history.
        :param validation_history: The validation loss history.
        :param logger: The logger instance.
        :return: A fully populated BoxOfficeClassificationEvaluationResult instance.
        """
        logger.debug("Calculating classification metrics on the test set...")

        classification_report: Optional[ClassificationReportDict] = None
        if config.calculate_classification_metrics:
            predict_params: BoxOfficeClassificationPredictParams = BoxOfficeClassificationPredictParams(verbose=0)
            # Model output is expected to be class probabilities (softmax)
            y_pred_probs: NDArray[float32] = model_core.predict(data=x_test, params=predict_params)

            # Dynamically generate label map based on thresholds
            # e.g., (50, 80) -> {0: 'Range 0', 1: 'Range 1', 2: 'Range 2'}
            num_classes: int = len(config.box_office_thresholds) + 1
            label_map: dict[int, str] = {i: f"Range {i}" for i in range(num_classes)}

            metrics_calculator: MultiClassClassificationMetricsCalculator = MultiClassClassificationMetricsCalculator(
                label_map=label_map,
                f1_average_method=config.f1_average_method
            )
            classification_report = metrics_calculator.generate_report(y_true=y_test, y_pred=y_pred_probs)

            logger.debug(f"  - Accuracy: {classification_report['accuracy']:.2%}")
            logger.debug(f"  - F1-Score (avg='{config.f1_average_method}'): {classification_report['f1_score']:.4f}")

        return cls(
            model_id=config.model_id,
            model_epoch=config.model_epoch,
            training_loss_history=training_history,
            validation_loss_history=validation_history,
            classification_report=classification_report
        )


class BoxOfficeClassificationEvaluator(
    BoxOfficeBaseEvaluator[
        BoxOfficeClassificationDataProcessor,
        BoxOfficeClassificationModelCore,
        BoxOfficeClassificationEvaluationConfig,
        BoxOfficeClassificationEvaluationResult
    ]
):
    """
    Evaluates a trained Box Office Classification Model.

    Inherits shared box office evaluation logic from BoxOfficeBaseEvaluator.
    """

    @property
    @override
    def _project_model_type(self) -> ProjectModelType:
        """
        Returns the model type associated with this evaluator.

        :return: ProjectModelType.BOX_OFFICE_CLASSIFICATION.
        """
        return ProjectModelType.BOX_OFFICE_CLASSIFICATION

    @override
    def _create_data_processor_instance(self, model_artifacts_path: Path) -> BoxOfficeClassificationDataProcessor:
        """
        Creates a DataProcessor instance for classification.

        :param model_artifacts_path: Path to the directory for model artifacts.
        :return: A BoxOfficeClassificationDataProcessor instance.
        """
        return BoxOfficeClassificationDataProcessor(model_artifacts_path=model_artifacts_path)

    @override
    def _create_model_core_instance(self, model_file_path: Path) -> BoxOfficeClassificationModelCore:
        """
        Creates a ModelCore instance for classification.

        :param model_file_path: Path to the saved Keras model file.
        :return: A BoxOfficeClassificationModelCore instance.
        """
        return BoxOfficeClassificationModelCore(model_path=model_file_path)

    @override
    def _create_evaluation_result(
        self,
        *,
        config: BoxOfficeClassificationEvaluationConfig,
        model_core: BoxOfficeClassificationModelCore,
        data_processor: BoxOfficeClassificationDataProcessor,
        x_test: NDArray[float32],
        y_test: NDArray[int_],
        training_history: list[float],
        validation_history: list[float]
    ) -> BoxOfficeClassificationEvaluationResult:
        """
        Creates the final BoxOfficeClassificationEvaluationResult by calling its factory method.

        :param config: The evaluation configuration.
        :param model_core: The trained model core.
        :param data_processor: The data processor (provided for consistency).
        :param x_test: The test features.
        :param y_test: The test labels.
        :param training_history: The training loss history.
        :param validation_history: The validation loss history.
        :return: The fully populated evaluation result object.
        """
        final_result: BoxOfficeClassificationEvaluationResult = BoxOfficeClassificationEvaluationResult.create(
            config=config,
            model_core=model_core,
            x_test=x_test,
            y_test=y_test,
            training_history=training_history,
            validation_history=validation_history,
            logger=self._logger
        )
        self._logger.debug(final_result.get_summary_string())
        return final_result

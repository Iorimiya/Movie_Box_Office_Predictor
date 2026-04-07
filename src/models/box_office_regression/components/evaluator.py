from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Any, Optional, TypeAlias

from numpy import array, float32, float64
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler
from typing_extensions import override

from src.core.project_config import ProjectModelType
from src.models.base.evaluation import (
    ClassificationSummaryMixin,
    RegressionEvaluationConfig,
    RegressionEvaluationResult,
)
from src.models.base.keras_setup import keras_base
from src.models.box_office_common.box_office_evaluator import (
    BoxOfficeBaseEvaluationConfig,
    BoxOfficeBaseEvaluator
)
from src.models.box_office_regression.components.data_processor import BoxOfficeRegressionDataProcessor
from src.models.box_office_regression.components.model_core import (
    BoxOfficeRegressionModelCore,
    BoxOfficeRegressionPredictParams,
)
from src.utilities.metrics import (
    ClassificationReportDict,
    PairwiseClassificationMetricsCalculator,
    PointwiseClassificationMetricsCalculator,
    RegressionMetricsCalculator,
    RegressionReportDict
)

# noinspection PyUnresolvedReferences
History: TypeAlias = keras_base.callbacks.History


class BoxOfficeRegressionEvaluationConfig(BoxOfficeBaseEvaluationConfig, RegressionEvaluationConfig):
    """
    Configuration for running a Box Office Regression Model evaluation.

    Inherits common evaluation parameters from BoxOfficeBaseEvaluationConfig and RegressionEvaluationConfig.

    :ivar calculate_classification_metrics: Flag to enable classification-based metrics.
    :ivar classification_method: The strategy for classification ('range' or 'trend').
    :ivar box_office_ranges: A tuple defining the upper boundaries of box office ranges.
    :ivar f1_average_method: The averaging method for F1 score calculation.
    """

    def __init__(
        self,
        *,
        calculate_loss: bool,
        calculate_classification_metrics: bool,
        classification_method: Optional[str] = None,
        box_office_ranges: tuple[int, ...] = (1_000_000, 10_000_000, 90_000_000),
        f1_average_method: str = 'macro',
        **kwargs: Any
    ) -> None:
        """
        Initializes the BoxOfficeRegressionEvaluationConfig.

        :param calculate_loss: Flag to calculate loss (e.g., MSE) on the test set.
        :param calculate_classification_metrics: Flag to enable classification-based metrics.
        :param classification_method: The strategy for classification ('range' or 'trend').
        :param box_office_ranges: A tuple defining the upper boundaries of box office ranges.
        :param f1_average_method: The averaging method for F1 score calculation.
        :param kwargs: Additional keyword arguments passed to the base class.
        """
        super().__init__(calculate_loss=calculate_loss, **kwargs)
        self.calculate_classification_metrics: bool = calculate_classification_metrics
        self.classification_method: Optional[str] = classification_method
        self.box_office_ranges: tuple[int, ...] = box_office_ranges
        self.f1_average_method: str = f1_average_method


@dataclass(frozen=True)
class BoxOfficeRegressionEvaluationResult(RegressionEvaluationResult, ClassificationSummaryMixin):
    """
    The specific evaluation result for the Box Office Regression Model.

    This class represents the most detailed evaluation result. It 'is-a'
    RegressionEvaluationResult and also 'mixes-in' the
    ClassificationSummaryMixin to gain summary generation capabilities.

    :ivar range_classification_report: A report for the pointwise classification task.
    :ivar trend_classification_report: A report for the pairwise classification task.
    """
    range_classification_report: Optional[ClassificationReportDict]
    trend_classification_report: Optional[ClassificationReportDict]

    @classmethod
    def create(
        cls,
        *,
        config: "BoxOfficeRegressionEvaluationConfig",
        model_core: "BoxOfficeRegressionModelCore",
        data_processor: "BoxOfficeRegressionDataProcessor",
        x_test: NDArray[Any],
        y_test: NDArray[Any],
        training_history: list[float],
        validation_history: list[float],
        logger: Logger
    ) -> "BoxOfficeRegressionEvaluationResult":
        """
        Factory method to create a complete evaluation result.

        This method orchestrates all metric calculations by delegating to the
        appropriate calculators and composes the final result object.

        :param config: The evaluation configuration.
        :param model_core: The trained model core.
        :param data_processor: The data processor used for scaling/unscaling.
        :param x_test: The test features.
        :param y_test: The test labels.
        :param training_history: The training loss history.
        :param validation_history: The validation loss history.
        :param logger: The logger instance.
        :return: A fully populated BoxOfficeRegressionEvaluationResult instance.
        """
        logger.debug(msg="Calculating requested metrics on the test set...")

        # Regression Metrics
        regression_report: Optional[RegressionReportDict] = None
        if config.calculate_loss:
            regression_report: Optional[RegressionReportDict] = cls._calculate_regression_report(
                model_core=model_core, x_test=x_test, y_test=y_test, logger=logger
            )

        # Classification Metrics (requires unscaling)
        range_report: Optional[ClassificationReportDict] = None
        trend_report: Optional[ClassificationReportDict] = None

        if config.calculate_classification_metrics:
            unscaled_pred: list[float]
            unscaled_actual: list[float]
            unscaled_inputs: list[float]
            unscaled_pred, unscaled_actual, unscaled_inputs = cls._get_unscaled_predictions(
                model_core=model_core, scaler=data_processor.scaler, x_test=x_test, y_test=y_test, logger=logger
            )

            # Dispatch based on the classification method strategy
            if config.classification_method == 'range':
                range_report: Optional[ClassificationReportDict] = cls._calculate_range_report(
                    predictions=unscaled_pred, actual=unscaled_actual, config=config, logger=logger
                )
            elif config.classification_method == 'trend':
                trend_report: Optional[ClassificationReportDict] = cls._calculate_trend_report(
                    predictions=unscaled_pred, actual=unscaled_actual, last_inputs=unscaled_inputs, logger=logger
                )

        return cls(
            model_id=config.model_id,
            model_epoch=config.model_epoch,
            training_loss_history=training_history,
            validation_loss_history=validation_history,
            regression_report=regression_report,
            range_classification_report=range_report,
            trend_classification_report=trend_report
        )

    @override
    def get_summary_string(self) -> str:
        """
        Generates a summary string for the regression and classification results.

        :return: The formatted summary string.
        """
        # Call the parent's get_summary_string to get the regression metrics
        base_summary: str = super().get_summary_string()
        lines: list[str] = [base_summary]

        if self.range_classification_report:
            lines.append(f"  - Range Accuracy: {self.range_classification_report['accuracy']:.2%}")
            lines.append(f"  - Range F1-Score: {self.range_classification_report['f1_score']:.4f}")

        if self.trend_classification_report:
            lines.append(f"  - Trend Accuracy: {self.trend_classification_report['accuracy']:.2%}")
            lines.append(f"  - Trend F1-Score: {self.trend_classification_report['f1_score']:.4f}")

        return "\n".join(lines)

    @staticmethod
    def _calculate_regression_report(
        model_core: "BoxOfficeRegressionModelCore", x_test: NDArray[Any], y_test: NDArray[Any], logger: Logger
    ) -> RegressionReportDict:
        """
        Calculates standard regression metrics.

        :param model_core: The model core to use for prediction.
        :param x_test: The test features.
        :param y_test: The test labels.
        :param logger: The logger instance.
        :return: A dictionary containing MSE, MAE, and R2 score.
        """
        logger.debug(msg="Calculating regression metrics (MSE, MAE, R²)...")
        predict_params: BoxOfficeRegressionPredictParams = BoxOfficeRegressionPredictParams(verbose=0)
        y_pred_scaled: NDArray[Any] = model_core.predict(data=x_test, params=predict_params)
        regression_calculator: RegressionMetricsCalculator = RegressionMetricsCalculator()
        report: RegressionReportDict = regression_calculator.generate_report(y_true=y_test, y_pred=y_pred_scaled)

        logger.debug(msg=f"  - Test MSE Loss: {report['mse']:.6f}")
        logger.debug(msg=f"  - Test MAE: {report['mae']:.6f}")
        logger.debug(msg=f"  - Test R² Score: {report['r2_score']:.4f}")
        return report

    @staticmethod
    def _get_unscaled_predictions(
        model_core: "BoxOfficeRegressionModelCore", scaler: MinMaxScaler, x_test: NDArray[float32],
        y_test: NDArray[float64],
        logger: Logger
    ) -> tuple[list[float], list[float], list[float]]:
        """
        Generates model predictions and inverse-transforms them to their original scale.

        :param model_core: The trained model core for generating predictions.
        :param scaler: The `MinMaxScaler` instance used for scaling.
        :param x_test: The scaled input test data.
        :param y_test: The scaled target test data.
        :param logger: The logger instance.
        :return: A tuple containing unscaled predictions, actual values, and last week inputs.
        """
        logger.debug(msg="Generating unscaled predictions for classification metrics...")
        predict_params: BoxOfficeRegressionPredictParams = BoxOfficeRegressionPredictParams(verbose=0)
        y_pred_scaled: NDArray[Any] = model_core.predict(data=x_test, params=predict_params)
        unscaled_predictions: list[float] = scaler.inverse_transform(X=y_pred_scaled).flatten().tolist()
        unscaled_actual: list[float] = scaler.inverse_transform(X=y_test.reshape(-1, 1)).flatten().tolist()
        last_week_input_scaled: NDArray[float32] = x_test[:, -1, 0].reshape(-1, 1)
        unscaled_last_week_inputs: list[float] = scaler.inverse_transform(X=last_week_input_scaled).flatten().tolist()
        return unscaled_predictions, unscaled_actual, unscaled_last_week_inputs

    @staticmethod
    def _calculate_range_report(
        predictions: list[float], actual: list[float], config: "BoxOfficeRegressionEvaluationConfig", logger: Logger
    ) -> ClassificationReportDict:
        """
        Calculates classification metrics based on box office revenue ranges.

        :param predictions: The unscaled predicted values.
        :param actual: The unscaled actual values.
        :param config: The evaluation configuration containing range definitions.
        :param logger: The logger instance.
        :return: A dictionary containing classification metrics for ranges.
        """
        logger.debug(msg="Calculating pointwise metrics (Range Accuracy, F1-Score)...")

        def value_to_label_fn(value: float) -> int:
            return BoxOfficeRegressionDataProcessor.get_range_index(value=value, ranges=config.box_office_ranges)

        range_labels: list[str] = BoxOfficeRegressionEvaluationResult._generate_range_labels(
            ranges=config.box_office_ranges
        )
        label_map: dict[int, str] = {i: label for i, label in enumerate(range_labels)}
        metrics_calculator: PointwiseClassificationMetricsCalculator = PointwiseClassificationMetricsCalculator(
            value_to_label_fn=value_to_label_fn, label_map=label_map, f1_average_method=config.f1_average_method
        )
        report: ClassificationReportDict = \
            metrics_calculator.generate_report(y_true=array(actual), y_pred=array(predictions))
        logger.debug(msg=f"  - Range Accuracy: {report['accuracy']:.2%}")
        logger.debug(msg=f"  - F1-Score (Range, avg='{config.f1_average_method}'): {report['f1_score']:.4f}")

        matrix_str: str = BoxOfficeRegressionEvaluationResult.format_confusion_matrix_string(
            matrix=report['confusion_matrix'],
            names=report.get('target_names') or []
        )
        logger.debug(
            msg=f"Full range classification report:\n{report['report_string']}\n\nConfusion Matrix:\n{matrix_str}")
        return report

    @staticmethod
    def _calculate_trend_report(
        predictions: list[float], actual: list[float], last_inputs: list[float], logger: Logger
    ) -> ClassificationReportDict:
        """
        Calculates classification metrics based on the trend (increase/decrease).

        :param predictions: The unscaled predicted values.
        :param actual: The unscaled actual values.
        :param last_inputs: The unscaled values from the previous time step.
        :param logger: The logger instance.
        :return: A dictionary containing classification metrics for trends.
        """
        logger.debug(msg="Calculating pairwise metrics (Trend Accuracy, F1-Score)...")

        def trend_fn(value: float, reference: float) -> int:
            return 1 if value > reference else 0

        metrics_calculator: PairwiseClassificationMetricsCalculator = PairwiseClassificationMetricsCalculator(
            value_pair_to_label_fn=trend_fn, reference_values=array(last_inputs),
            label_map={0: 'Decrease/Stay', 1: 'Increase'}, f1_average_method='binary'
        )
        report: ClassificationReportDict = \
            metrics_calculator.generate_report(y_true=array(actual), y_pred=array(predictions))
        logger.debug(msg=f"  - Trend Accuracy: {report['accuracy']:.2%}")
        logger.debug(msg=f"  - F1-Score (Trend, avg='binary'): {report['f1_score']:.4f}")

        matrix_str: str = BoxOfficeRegressionEvaluationResult.format_confusion_matrix_string(
            matrix=report['confusion_matrix'],
            names=report.get('target_names') or []
        )
        logger.debug(
            msg=f"Full trend classification report:\n{report['report_string']}\n\nConfusion Matrix:\n{matrix_str}")
        return report

    @staticmethod
    def _generate_range_labels(ranges: tuple[int, ...]) -> list[str]:
        """
        Generates human-readable string labels for the defined box office ranges.

        :param ranges: A tuple of integer boundaries for the ranges.
        :return: A list of string labels.
        """
        if not ranges:
            return []
        sorted_ranges: list[int] = sorted(list(ranges))
        labels: list[str] = []

        def fmt(n: int) -> str:
            if n >= 1_000_000_000:
                return f"{n / 1_000_000_000:.1f}B"
            if n >= 1_000_000:
                return f"{n / 1_000_000:.1f}M"
            if n >= 1_000:
                return f"{n / 1_000:.1f}K"
            return str(n)

        labels.append(f"< {fmt(sorted_ranges[0])}")
        for i in range(len(sorted_ranges) - 1):
            labels.append(f"{fmt(sorted_ranges[i])} - {fmt(sorted_ranges[i + 1])}")
        labels.append(f">= {fmt(sorted_ranges[-1])}")
        return labels


class BoxOfficeRegressionEvaluator(
    BoxOfficeBaseEvaluator[
        BoxOfficeRegressionDataProcessor,
        BoxOfficeRegressionModelCore,
        BoxOfficeRegressionEvaluationConfig,
        BoxOfficeRegressionEvaluationResult
    ]
):
    """
    Evaluates a trained Box Office Regression Model.

    Inherits shared box office evaluation logic from BoxOfficeBaseEvaluator.

    :ivar _logger: A logger instance for evaluation.
    """

    @property
    @override
    def _project_model_type(self) -> ProjectModelType:
        """
        Returns the model type associated with this evaluator.

        :return: ProjectModelType.BOX_OFFICE_REGRESSION.
        """
        return ProjectModelType.BOX_OFFICE_REGRESSION

    @override
    def _create_data_processor_instance(self, model_artifacts_path: Path) -> BoxOfficeRegressionDataProcessor:
        """
        Creates a DataProcessor instance for regression.

        :param model_artifacts_path: Path to the directory for model artifacts.
        :return: A BoxOfficeRegressionDataProcessor instance.
        """
        return BoxOfficeRegressionDataProcessor(model_artifacts_path=model_artifacts_path)

    @override
    def _create_model_core_instance(self, model_file_path: Path) -> BoxOfficeRegressionModelCore:
        """
        Creates a ModelCore instance for regression.

        :param model_file_path: Path to the saved Keras model file.
        :return: A BoxOfficeRegressionModelCore instance.
        """
        return BoxOfficeRegressionModelCore(model_path=model_file_path)

    @override
    def _create_evaluation_result(
        self,
        *,
        config: BoxOfficeRegressionEvaluationConfig,
        model_core: BoxOfficeRegressionModelCore,
        data_processor: BoxOfficeRegressionDataProcessor,
        x_test: NDArray[Any],
        y_test: NDArray[Any],
        training_history: list[float],
        validation_history: list[float]
    ) -> BoxOfficeRegressionEvaluationResult:
        """
        Creates the final BoxOfficeRegressionEvaluationResult by calling its factory method.

        :param config: The evaluation configuration.
        :param model_core: The trained model core.
        :param data_processor: The data processor used for scaling/unscaling.
        :param x_test: The test features.
        :param y_test: The test labels.
        :param training_history: The training loss history.
        :param validation_history: The validation loss history.
        :return: The fully populated evaluation result object.
        """
        final_result: BoxOfficeRegressionEvaluationResult = BoxOfficeRegressionEvaluationResult.create(
            config=config,
            model_core=model_core,
            data_processor=data_processor,
            x_test=x_test,
            y_test=y_test,
            training_history=training_history,
            validation_history=validation_history,
            logger=self._logger
        )
        self._logger.debug(msg=final_result.get_summary_string())
        return final_result

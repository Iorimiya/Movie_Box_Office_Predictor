from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Optional

from numpy import array, float32, float64
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler
from typing_extensions import override

from src.core.project_config import ProjectModelType, ProjectPaths
from src.data_handling.movie_collections import MovieData
from src.models.base.evaluation import (
    BaseEvaluator,
    BaseEvaluationConfig,
    RegressionEvaluationResult,
    ClassificationSummaryMixin,
)
from src.models.base.keras_setup import keras_base
from src.models.prediction.components.data_processor import (
    PredictionDataConfig,
    PredictionDataProcessor,
    PredictionDataSource,
    PredictionTrainingProcessedData,
)
from src.models.prediction.components.model_core import (
    PredictionModelCore,
    PredictionPredictConfig,
)
from src.utilities.metrics import (
    ClassificationReportDict,
    PairwiseClassificationMetricsCalculator,
    PointwiseClassificationMetricsCalculator,
    RegressionMetricsCalculator,
    RegressionReportDict
)

History = keras_base.callbacks.History


@dataclass(frozen=True)
class PredictionEvaluationConfig(BaseEvaluationConfig):
    """
    Configuration for running a prediction model evaluation.

    Inherits common evaluation parameters from BaseEvaluationConfig.

    :ivar training_week_len: The number of past weeks used for prediction.
    :ivar calculate_trend_accuracy: Flag to calculate trend prediction accuracy.
    :ivar calculate_range_accuracy: Flag to calculate range prediction accuracy.
    :ivar box_office_ranges: A tuple defining the upper boundaries of box office ranges.
    :ivar f1_average_method: The averaging method for F1 score calculation.
    """

    training_week_len: int = 4
    calculate_trend_accuracy: bool = False
    calculate_range_accuracy: bool = False
    box_office_ranges: tuple[int, ...] = (1_000_000, 10_000_000, 90_000_000)
    f1_average_method: str = 'macro'


@dataclass(frozen=True)
class PredictionEvaluationResult(RegressionEvaluationResult, ClassificationSummaryMixin):
    """
    The specific evaluation result for the Prediction model.

    This class represents the most detailed evaluation result. It 'is-a'
    RegressionEvaluationResult and also 'mixes-in' the
    ClassificationSummaryMixin to gain summary generation capabilities.
    It holds multiple reports and provides behaviors for its own creation
    and presentation.

    :ivar range_classification_report: A report for the pointwise classification
                                       task (e.g., box office ranges).
    :ivar trend_classification_report: A report for the pairwise classification
                                       task (e.g., box office trend).
    """
    range_classification_report: Optional[ClassificationReportDict]
    trend_classification_report: Optional[ClassificationReportDict]

    @classmethod
    def create(
        cls,
        *,
        config: "PredictionEvaluationConfig",
        model_core: "PredictionModelCore",
        data_processor: "PredictionDataProcessor",
        x_test: NDArray[any],
        y_test: NDArray[any],
        training_history: list[float],
        validation_history: list[float],
        logger: Logger
    ) -> "PredictionEvaluationResult":
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
        :return: A fully populated PredictionEvaluationResult instance.
        """
        logger.info("Calculating requested metrics on the test set...")

        # Regression Metrics
        regression_report: Optional[RegressionReportDict] = None
        if config.calculate_loss:
            regression_report = cls._calculate_regression_report(
                model_core=model_core, x_test=x_test, y_test=y_test, logger=logger
            )

        # Classification Metrics (requires unscaling)
        range_report: Optional[ClassificationReportDict] = None
        trend_report: Optional[ClassificationReportDict] = None

        # Use the explicit flag to determine if unscaling is needed
        if config.calculate_classification_metrics:
            unscaled_pred: list[float]
            unscaled_actual: list[float]
            unscaled_inputs: list[float]
            unscaled_pred, unscaled_actual, unscaled_inputs = cls._get_unscaled_predictions(
                model_core=model_core, scaler=data_processor.scaler, x_test=x_test, y_test=y_test, logger=logger
            )
            if config.calculate_range_accuracy:
                range_report = cls._calculate_range_report(
                    predictions=unscaled_pred, actual=unscaled_actual, config=config, logger=logger
                )
            if config.calculate_trend_accuracy:
                trend_report = cls._calculate_trend_report(
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

    @property
    def summary_string(self) -> str:
        """
        Returns a formatted, multi-line summary string of all key metrics.
        """
        lines: list[str] = [f"Evaluation Summary for Model '{self.model_id}' (Epoch {self.model_epoch}):"]
        if self.regression_report:
            lines.append(f"  - MSE Loss: {self.regression_report['mse']:.6f}")
            lines.append(f"  - MAE:      {self.regression_report['mae']:.6f}")
            lines.append(f"  - R² Score: {self.regression_report['r2_score']:.4f}")
        if self.range_classification_report:
            lines.append(f"  - Range Accuracy: {self.range_classification_report['accuracy']:.2%}")
            lines.append(f"  - Range F1-Score: {self.range_classification_report['f1_score']:.4f}")
        if self.trend_classification_report:
            lines.append(f"  - Trend Accuracy: {self.trend_classification_report['accuracy']:.2%}")
            lines.append(f"  - Trend F1-Score: {self.trend_classification_report['f1_score']:.4f}")
        return "\n".join(lines)

    @staticmethod
    def _calculate_regression_report(
        model_core: "PredictionModelCore", x_test: NDArray[any], y_test: NDArray[any], logger: Logger
    ) -> RegressionReportDict:
        """
        Calculates standard regression metrics.

        :param model_core: The model core to use for prediction.
        :param x_test: The test features.
        :param y_test: The test labels.
        :param logger: The logger instance.
        :return: A dictionary containing MSE, MAE, and R2 score.
        """
        logger.info("Calculating regression metrics (MSE, MAE, R²)...")
        y_pred_scaled: NDArray[any] = model_core.predict(data=x_test, config=PredictionPredictConfig(verbose=0))
        regression_calculator:RegressionMetricsCalculator = RegressionMetricsCalculator()
        report:RegressionReportDict = regression_calculator.generate_report(y_true=y_test, y_pred=y_pred_scaled)

        logger.info(f"  - Test MSE Loss: {report['mse']:.6f}")
        logger.info(f"  - Test MAE: {report['mae']:.6f}")
        logger.info(f"  - Test R² Score: {report['r2_score']:.4f}")
        return report

    @staticmethod
    def _get_unscaled_predictions(
        model_core: PredictionModelCore, scaler: MinMaxScaler, x_test: NDArray[float32], y_test: NDArray[float64],
        logger: Logger
    ) -> tuple[list[float], list[float], list[float]]:
        """
        Generates model predictions and inverse-transforms them to their original scale.

        This method also un-scales the actual labels and the last box office value
        from each input sequence, which is needed for trend accuracy calculation.

        :param model_core: The trained model core for generating predictions.
        :param scaler: The `MinMaxScaler` instance used for scaling.
        :param x_test: The scaled input test data.
        :param y_test: The scaled target test data.
        :returns: A tuple containing:
                  - A list of unscaled predicted box office values.
                  - A list of unscaled actual box office values.
                  - A list of unscaled box office values from the last input week.
        """
        logger.info("Generating unscaled predictions for classification metrics...")
        y_pred_scaled: NDArray[any] = model_core.predict(data=x_test, config=PredictionPredictConfig(verbose=0))
        unscaled_predictions: list[float] = scaler.inverse_transform(y_pred_scaled).flatten().tolist()
        unscaled_actual: list[float] = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten().tolist()
        last_week_input_scaled: NDArray[float32] = x_test[:, -1, 0].reshape(-1, 1)
        unscaled_last_week_inputs: list[float] = scaler.inverse_transform(last_week_input_scaled).flatten().tolist()
        return unscaled_predictions, unscaled_actual, unscaled_last_week_inputs

    @staticmethod
    def _calculate_range_report(
        predictions: list[float], actual: list[float], config: "PredictionEvaluationConfig", logger: Logger
    ) -> ClassificationReportDict:
        """
        Calculates classification metrics based on box office revenue ranges.

        :param predictions: The unscaled predicted values.
        :param actual: The unscaled actual values.
        :param config: The evaluation configuration containing range definitions.
        :param logger: The logger instance.
        :return: A dictionary containing classification metrics for ranges.
        """
        logger.info("Calculating pointwise metrics (Range Accuracy, F1-Score)...")

        def value_to_label_fn(value: float) -> int:
            return PredictionDataProcessor.get_range_index(value=value, ranges=config.box_office_ranges)

        range_labels:list[str] = PredictionEvaluationResult._generate_range_labels(ranges=config.box_office_ranges)
        label_map:dict[int,str] = {i: label for i, label in enumerate(range_labels)}
        metrics_calculator:PointwiseClassificationMetricsCalculator = PointwiseClassificationMetricsCalculator(
            value_to_label_fn=value_to_label_fn, label_map=label_map, f1_average_method=config.f1_average_method
        )
        report:ClassificationReportDict = \
            metrics_calculator.generate_report(y_true=array(actual), y_pred=array(predictions))
        logger.info(f"  - Range Accuracy: {report['accuracy']:.2%}")
        logger.info(f"  - F1-Score (Range, avg='{config.f1_average_method}'): {report['f1_score']:.4f}")

        matrix_str:str = PredictionEvaluationResult.format_confusion_matrix_string(
            matrix=report['confusion_matrix'],
            names=report.get('target_names') or []
        )
        logger.info(f"Full range classification report:\n{report['report_string']}\n\nConfusion Matrix:\n{matrix_str}")
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
        logger.info("Calculating pairwise metrics (Trend Accuracy, F1-Score)...")

        def trend_fn(value: float, reference: float) -> int:
            return 1 if value > reference else 0

        metrics_calculator:PairwiseClassificationMetricsCalculator = PairwiseClassificationMetricsCalculator(
            value_pair_to_label_fn=trend_fn, reference_values=array(last_inputs),
            label_map={0: 'Decrease/Stay', 1: 'Increase'}, f1_average_method='binary'
        )
        report:ClassificationReportDict = \
            metrics_calculator.generate_report(y_true=array(actual), y_pred=array(predictions))
        logger.info(f"  - Trend Accuracy: {report['accuracy']:.2%}")
        logger.info(f"  - F1-Score (Trend, avg='binary'): {report['f1_score']:.4f}")

        matrix_str:str = PredictionEvaluationResult.format_confusion_matrix_string(
            matrix=report['confusion_matrix'],
            names=report.get('target_names') or []
        )
        logger.info(f"Full trend classification report:\n{report['report_string']}\n\nConfusion Matrix:\n{matrix_str}")
        return report

    @staticmethod
    def _generate_range_labels(ranges: tuple[int, ...]) -> list[str]:
        """
        Generates human-readable string labels for the defined box office ranges.

        :param ranges: A tuple of integer boundaries for the ranges.
        :return: A list of string labels (e.g., "< 1.0M", "1.0M - 10.0M").
        """
        if not ranges:
            return []
        sorted_ranges:list[int] = sorted(list(ranges))
        labels: list[str] = []

        def fmt(n: int) -> str:
            if n >= 1_000_000_000: return f"{n / 1_000_000_000:.1f}B"
            if n >= 1_000_000: return f"{n / 1_000_000:.1f}M"
            if n >= 1_000: return f"{n / 1_000:.1f}K"
            return str(n)

        labels.append(f"< {fmt(sorted_ranges[0])}")
        for i in range(len(sorted_ranges) - 1):
            labels.append(f"{fmt(sorted_ranges[i])} - {fmt(sorted_ranges[i + 1])}")
        labels.append(f">= {fmt(sorted_ranges[-1])}")
        return labels


class PredictionEvaluator(
    BaseEvaluator[PredictionDataProcessor, PredictionModelCore, PredictionEvaluationConfig, PredictionEvaluationResult]
):
    """
    Evaluates a trained box office prediction model.

    This evaluator is now a lightweight coordinator. It sets up components,
    prepares data, and then delegates the entire metric calculation and result
    compilation process to the `PredictionEvaluationResult.create` factory method.
    """

    @override
    def _setup_components(
        self, model_id: str, model_epoch: int
    ) -> tuple[PredictionDataProcessor, PredictionModelCore, Path]:
        """
        Sets up and loads the necessary data processor and model core for evaluation.

        :param model_id: The unique identifier for the model series.
        :param model_epoch: The specific training epoch of the model to load.
        :return: A tuple containing the initialized data processor, model core,
                 and the path to the model artifacts directory.
        :raises FileNotFoundError: If the scaler artifact cannot be found.
        """
        self.logger.info("Loading model and data processor artifacts...")
        artifacts_path: Path = ProjectPaths.get_model_root_path(
            model_id=model_id, model_type=ProjectModelType.PREDICTION
        )
        model_file_path: Path = artifacts_path / f"{model_id}_{model_epoch:04d}.keras"

        data_processor: PredictionDataProcessor = PredictionDataProcessor(model_artifacts_path=artifacts_path)
        if not data_processor.scaler:
            raise FileNotFoundError(f"Could not load scaler artifact from: {artifacts_path}")

        model_core: PredictionModelCore = PredictionModelCore(model_path=model_file_path)
        return data_processor, model_core, artifacts_path

    @override
    def _prepare_test_data(
        self, data_processor: PredictionDataProcessor, config: PredictionEvaluationConfig
    ) -> tuple[NDArray[float32], NDArray[float64]]:
        """
        Loads and processes data to retrieve the evaluation set.

        :param data_processor: The initialized data processor.
        :param config: The configuration object for the evaluation run.
        :return: A tuple containing the evaluation features (x_eval) and labels (y_eval).
        :raises ValueError: If reproducibility mode is selected but split parameters are missing.
        """
        self.logger.info("Loading and processing evaluation dataset...")
        data_source: PredictionDataSource = PredictionDataSource(dataset_name=config.dataset_name)
        raw_data: list[MovieData] = data_processor.load_raw_data(source=data_source)

        processing_config: PredictionDataConfig = PredictionDataConfig(
            training_week_len=config.training_week_len,
            split_ratios=config.split_ratios,
            random_state=config.random_state
        )

        if config.evaluate_on_full_dataset:
            self.logger.info("Evaluation mode: Processing the full dataset as the test set.")
            x_eval, y_eval = data_processor.process_for_evaluation(
                raw_data=raw_data, config=processing_config
            )
            return x_eval, y_eval
        else:
            self.logger.info("Evaluation mode: Reproducing the original test split.")

            if config.split_ratios is None or config.random_state is None:
                raise ValueError(
                    "For reproducibility mode (evaluate_on_full_dataset=False), "
                    "'split_ratios' and 'random_state' must be provided in the configuration."
                )

            processed_data: PredictionTrainingProcessedData = data_processor.process_for_training(
                raw_data=raw_data, config=processing_config
            )
            return processed_data['x_test'], processed_data['y_test']

    @override
    def _create_evaluation_result(
        self,
        *,
        config: PredictionEvaluationConfig,
        model_core: PredictionModelCore,
        data_processor: PredictionDataProcessor,
        x_test: NDArray[any],
        y_test: NDArray[any],
        training_history: list[float],
        validation_history: list[float]
    ) -> PredictionEvaluationResult:
        """
        Creates the final PredictionEvaluationResult by calling its factory method.

        :param config: The evaluation configuration.
        :param model_core: The trained model core.
        :param data_processor: The data processor used for scaling/unscaling.
        :param x_test: The test features.
        :param y_test: The test labels.
        :param training_history: The training loss history.
        :param validation_history: The validation loss history.
        :return: The fully populated evaluation result object.
        """
        # The Evaluator's only job is now to call the factory method.
        final_result:PredictionEvaluationResult = PredictionEvaluationResult.create(
            config=config,
            model_core=model_core,
            data_processor=data_processor,
            x_test=x_test,
            y_test=y_test,
            training_history=training_history,
            validation_history=validation_history,
            logger=self.logger
        )
        # We can log the summary here, after the result is created.
        self.logger.info(final_result.summary_string)
        return final_result

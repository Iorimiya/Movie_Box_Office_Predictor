from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

from numpy import array, float32, float64, int_
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler
from typing_extensions import override

from src.core.project_config import ProjectPaths, ProjectModelType
from src.data_handling.movie_collections import MovieData
from src.models.base.base_evaluator import BaseEvaluator, BaseEvaluationResult, BaseEvaluationConfig
from src.models.base.keras_setup import keras_base
from src.models.prediction.components.data_processor import (
    PredictionDataProcessor,
    PredictionDataSource,
    PredictionDataConfig,
    PredictionTrainingProcessedData,
)
from src.models.prediction.components.model_core import (
    PredictionModelCore,
    PredictionEvaluateConfig,
    PredictionPredictConfig,
)

from src.utilities.metrics import PointwiseClassificationMetrics, PairwiseClassificationMetrics

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
class PredictionEvaluationResult(BaseEvaluationResult):
    """
    A structured result of a prediction model evaluation run.

    Inherits common fields from BaseEvaluationResult.

    :ivar trend_accuracy: The trend prediction accuracy on the test set.
    :ivar test_accuracy: The range prediction accuracy on the test set.
    """
    trend_accuracy: Optional[float]
    test_accuracy: Optional[float]


class PredictionEvaluator(
    BaseEvaluator[PredictionDataProcessor, PredictionModelCore, PredictionEvaluationConfig, PredictionEvaluationResult]
):
    """
    Evaluates a trained box office prediction model.

    This evaluator loads a specific model checkpoint and its corresponding scaler,
    recreates the test dataset, and computes various performance metrics such as
    MSE loss, trend accuracy, and delegates classification-based metrics to a
    centralized framework.
    """

    @override
    def _setup_components(
        self, model_id: str, model_epoch: int
    ) -> tuple[PredictionDataProcessor, PredictionModelCore, Path]:
        """
        Sets up and loads the necessary data processor and model core for evaluation.

        This method constructs the paths to the model and its artifacts, then
        initializes the data processor and the model core by loading them from
        the specified files.

        :param model_id: The unique identifier of the model to be loaded.
        :param model_epoch: The specific epoch of the model to load.
        :returns: A tuple containing the initialized data processor, the loaded model core,
                  and the path to the model's artifacts directory.
        :raises FileNotFoundError: If the scaler artifact cannot be found or loaded.
        """
        self.logger.info("Step 1: Loading model and data processor artifacts...")
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

        This method supports two modes based on `config.evaluate_on_full_dataset`:
        - If `False` (default), it reproduces the original test split from the dataset.
        - If `True`, it processes the entire dataset as a single evaluation set.

        :param data_processor: The initialized data processor with its scaler loaded.
        :param config: The configuration object for the evaluation run.
        :returns: A tuple containing the evaluation features (x_eval) and labels (y_eval).
        :raises ValueError: If `evaluate_on_full_dataset` is `False` but `split_ratios`
                            or `random_state` are not provided in the config.
        """
        self.logger.info("Step 3: Loading and processing evaluation dataset...")
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
    def _calculate_metrics(
        self,
        model_core: PredictionModelCore,
        data_processor: PredictionDataProcessor,
        x_test: NDArray[float32],
        y_test: NDArray[float64],
        config: PredictionEvaluationConfig
    ) -> dict[str, Optional[float]]:
        """
        Calculates various performance metrics based on the evaluation configuration.

        This method orchestrates the calculation of MSE loss, trend accuracy,
        and delegates classification-based metrics (range accuracy, F1-score)
        to the centralized metrics framework.

        :param model_core: The trained model core to use for predictions.
        :param data_processor: The data processor containing the fitted scaler.
        :param x_test: The test features.
        :param y_test: The test labels (scaled).
        :param config: The configuration directing which metrics to calculate.
        :returns: A dictionary mapping metric names to their calculated values.
        """
        metrics: dict[str, Optional[float]] = {
            'test_loss': None,
            'trend_accuracy': None,
            'test_accuracy': None,
            'f1_score': None
        }

        if config.calculate_f1_score and not (config.calculate_range_accuracy or config.calculate_trend_accuracy):
            raise ValueError(
                "The 'calculate_f1_score' flag cannot be used alone. "
                "It must be combined with either 'calculate_range_accuracy' or 'calculate_trend_accuracy'."
            )
        if config.calculate_loss:
            metrics['test_loss'] = self._calculate_mse_loss(model_core=model_core, x_test=x_test, y_test=y_test)

        needs_unscaling: bool = any([
            config.calculate_trend_accuracy,
            config.calculate_range_accuracy,
            config.calculate_f1_score
        ])

        if needs_unscaling:
            unscaled_pred, unscaled_actual, unscaled_inputs = self._get_unscaled_predictions(
                model_core=model_core,
                scaler=data_processor.scaler,
                x_test=x_test,
                y_test=y_test
            )
            if config.calculate_trend_accuracy:
                trend_metrics: dict[str, Optional[float]] = self._calculate_trend_metrics(
                    predictions=unscaled_pred,
                    actual=unscaled_actual,
                    last_inputs=unscaled_inputs,
                    calculate_f1=config.calculate_f1_score  # Pass the f1 flag
                )
                metrics['trend_accuracy'] = trend_metrics.get('accuracy')
                # Only populate f1_score if it was requested for this method
                if config.calculate_f1_score:
                    metrics['f1_score'] = trend_metrics.get('f1_score')

            if config.calculate_range_accuracy:
                range_metrics: dict[str, Optional[float]] = self._calculate_range_metrics(
                    predictions=unscaled_pred,
                    actual=unscaled_actual,
                    config=config,
                    calculate_f1=config.calculate_f1_score  # Pass the f1 flag
                )
                metrics['test_accuracy'] = range_metrics.get('accuracy')
                # Only populate f1_score if it was requested for this method
                if config.calculate_f1_score:
                    metrics['f1_score'] = range_metrics.get('f1_score')

        return metrics

    @override
    def _compile_final_result(
        self,
        config: PredictionEvaluationConfig,
        metrics: dict[str, Optional[float]],
        training_history: list[float],
        validation_history: list[float]
    ) -> PredictionEvaluationResult:
        """
        Compiles the final result object from the calculated metrics and history.

        :param config: The original evaluation configuration.
        :param metrics: A dictionary of calculated performance metrics.
        :param training_history: A list of training loss values from the model's history.
        :param validation_history: A list of validation loss values from the model's history.
        :returns: A populated `PredictionEvaluationResult` object.
        """
        return PredictionEvaluationResult(
            model_id=config.model_id,
            model_epoch=config.model_epoch,
            test_loss=metrics.get('test_loss'),
            trend_accuracy=metrics.get('trend_accuracy'),
            test_accuracy=metrics.get('test_accuracy'),
            f1_score=metrics.get('f1_score'),
            training_loss_history=training_history,
            validation_loss_history=validation_history
        )

    def _calculate_mse_loss(
        self, model_core: PredictionModelCore, x_test: NDArray[float32], y_test: NDArray[float64]
    ) -> float:
        """
        Calculates the Mean Squared Error (MSE) loss on the test set.

        :param model_core: The model to evaluate.
        :param x_test: The test features.
        :param y_test: The test labels.
        :returns: The MSE loss value.
        """
        self.logger.info("Step 4a: Calculating MSE loss on the test set...")
        eval_config: PredictionEvaluateConfig = PredictionEvaluateConfig(verbose=0)
        loss: float = model_core.evaluate(x_test=x_test, y_test=y_test, config=eval_config)
        self.logger.info(f"  - Test MSE Loss: {loss:.6f}")
        return loss

    def _get_unscaled_predictions(
        self, model_core: PredictionModelCore, scaler: MinMaxScaler,
        x_test: NDArray[float32], y_test: NDArray[float64]
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
        self.logger.info("Step 4b: Generating unscaled predictions for accuracy metrics...")
        predict_config: PredictionPredictConfig = PredictionPredictConfig(verbose=0)
        y_pred_scaled: NDArray[Any] = model_core.predict(data=x_test, config=predict_config)

        unscaled_predictions: list[float] = scaler.inverse_transform(y_pred_scaled).flatten().tolist()
        unscaled_actual: list[float] = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten().tolist()

        # Unscale the last box office value from each input sequence
        last_week_input_scaled: NDArray[float32] = x_test[:, -1, 0].reshape(-1, 1)
        unscaled_last_week_inputs: list[float] = scaler.inverse_transform(last_week_input_scaled).flatten().tolist()

        return unscaled_predictions, unscaled_actual, unscaled_last_week_inputs

    def _calculate_range_metrics(
        self,
        predictions: list[float],
        actual: list[float],
        config: PredictionEvaluationConfig,
        calculate_f1: bool
    ) -> dict[str, Optional[float]]:
        """
        Calculates classification metrics based on predefined box office ranges.

        This method uses the `PointwiseClassificationMetrics` framework to compute
        accuracy and, optionally, the F1-score.

        :param predictions: A list of unscaled predicted box office values.
        :param actual: A list of unscaled actual box office values.
        :param config: The evaluation configuration containing box office ranges.
        :param calculate_f1: A boolean flag indicating whether to compute the F1-score.
        :returns: A dictionary with 'accuracy' and optional 'f1_score'.
        """
        self.logger.info("Calculating pointwise metrics (Range Accuracy, F1-Score)...")

        def value_to_label_fn(value: float) -> int:
            return PredictionDataProcessor.get_range_index(value=value, ranges=config.box_office_ranges)

        range_labels: list[str] = self._generate_range_labels(ranges=config.box_office_ranges)
        label_map: dict[int, str] = {i: label for i, label in enumerate(range_labels)}

        metrics_calculator = PointwiseClassificationMetrics(
            value_to_label_fn=value_to_label_fn,
            label_map=label_map,
            f1_average_method=config.f1_average_method
        )

        report: dict[str, Any] = metrics_calculator.generate_report(
            y_true=array(actual),
            y_pred=array(predictions)
        )

        accuracy: float = report.get('accuracy', 0.0)
        self.logger.info(f"  - Range Accuracy: {accuracy:.2%}")

        f1: Optional[float] = None
        if calculate_f1:
            f1 = report.get('f1_score', 0.0)
            self.logger.info(f"  - F1-Score (Range, average='{config.f1_average_method}'): {f1:.4f}")

        conf_matrix: Optional[NDArray[int_]] = report.get('confusion_matrix')
        target_names: Optional[list[str]] = report.get('target_names')
        if conf_matrix is not None and target_names:
            matrix_str: str = self._format_confusion_matrix_string(matrix=conf_matrix, names=target_names)
            self.logger.info(
                f"Full range classification report:\n{report.get('report_string')}\n\nConfusion Matrix:\n{matrix_str}")
        else:
            self.logger.info(f"Full range classification report:\n{report.get('report_string')}")

        return {'accuracy': accuracy, 'f1_score': f1}

    def _calculate_trend_metrics(
        self,
        predictions: list[float],
        actual: list[float],
        last_inputs: list[float],
        calculate_f1: bool
    ) -> dict[str, Optional[float]]:
        """
        Calculates classification metrics based on the trend (increase/decrease).

        This method uses the `PairwiseClassificationMetrics` framework to compute
        accuracy and, optionally, the F1-score for trend prediction.

        :param predictions: A list of unscaled predicted box office values.
        :param actual: A list of unscaled actual box office values.
        :param last_inputs: A list of reference values from the last input week.
        :param calculate_f1: A boolean flag indicating whether to compute the F1-score.
        :returns: A dictionary with 'accuracy' and optional 'f1_score'.
        """
        self.logger.info("Calculating pairwise metrics (Trend Accuracy, F1-Score)...")

        def trend_value_pair_to_label_fn(value: float, reference: float) -> int:
            """Returns 1 if value > reference (increase), else 0."""
            return 1 if value > reference else 0

        trend_metrics_calculator = PairwiseClassificationMetrics(
            value_pair_to_label_fn=trend_value_pair_to_label_fn,
            reference_values=array(last_inputs),
            label_map={0: 'Decrease/Stay', 1: 'Increase'},
            f1_average_method='binary'
        )

        report: dict[str, Any] = trend_metrics_calculator.generate_report(
            y_true=array(actual),
            y_pred=array(predictions)
        )

        accuracy: float = report.get('accuracy', 0.0)
        self.logger.info(f"  - Trend Accuracy: {accuracy:.2%}")

        f1: Optional[float] = None
        if calculate_f1:
            f1 = report.get('f1_score', 0.0)
            self.logger.info(f"  - F1-Score (Trend, average='binary'): {f1:.4f}")

        conf_matrix: Optional[NDArray[int_]] = report.get('confusion_matrix')
        target_names: Optional[list[str]] = report.get('target_names')
        if conf_matrix is not None and target_names:
            matrix_str: str = self._format_confusion_matrix_string(matrix=conf_matrix, names=target_names)
            self.logger.info(
                f"Full trend classification report:\n{report.get('report_string')}\n\nConfusion Matrix:\n{matrix_str}")
        else:
            self.logger.info(f"Full trend classification report:\n{report.get('report_string')}")

        return {'accuracy': accuracy, 'f1_score': f1}


    @staticmethod
    def _generate_range_labels(ranges: tuple[int, ...]) -> list[str]:
        """
        Generates human-readable labels from a tuple of box office range boundaries.

        For example, an input of (1_000_000, 10_000_000) would produce:
        ['< 1.0M', '1.0M - 10.0M', '>= 10.0M']

        :param ranges: A sorted tuple of integer boundaries.
        :return: A list of formatted string labels for each range.
        """
        if not ranges:
            return []

        sorted_ranges: list[int] = sorted(list(ranges))
        labels: list[str] = []

        def format_number(n: int) -> str:
            if n >= 1_000_000_000:
                return f"{n / 1_000_000_000:.1f}B"
            if n >= 1_000_000:
                return f"{n / 1_000_000:.1f}M"
            if n >= 1_000:
                return f"{n / 1_000:.1f}K"
            return str(n)

        labels.append(f"< {format_number(sorted_ranges[0])}")

        for i in range(len(sorted_ranges) - 1):
            lower_bound_str: str = format_number(sorted_ranges[i])
            upper_bound_str: str = format_number(sorted_ranges[i + 1])
            labels.append(f"{lower_bound_str} - {upper_bound_str}")

        labels.append(f">= {format_number(sorted_ranges[-1])}")

        return labels

    @staticmethod
    def _format_confusion_matrix_string(matrix: NDArray[int_], names: list[str]) -> str:
        """
        Formats a confusion matrix into a human-readable string for logging.

        :param matrix: The confusion matrix as a NumPy array.
        :param names: A list of string names for the classes, corresponding to the matrix axes.
        :return: A formatted, multi-line string representation of the confusion matrix.
        """
        if matrix.size == 0 or not names:
            return "  [Confusion Matrix is empty or has no labels]"

        # Determine column widths for alignment
        header_col_width: int = max(len(name) for name in names)
        cell_width: int = max(
            len(str(cell)) for cell in matrix.flatten()
        )
        # Ensure cell width is at least as wide as the longest name
        cell_width = max(cell_width, max(len(name) for name in names)) + 2

        # Header row
        header: str = f"{'':<{header_col_width}} |" + "".join([f"{name:^{cell_width}}" for name in names])
        separator: str = '-' * (header_col_width + 1) + '-' * (cell_width * len(names))

        # Build the string
        lines: list[str] = [
            f"{'True / Pred':<{header_col_width}} | {'Predicted Labels':^{cell_width * len(names) - 1}}", header,
            separator]
        for i, name in enumerate(names):
            row_str: str = f"{name:<{header_col_width}} |"
            for j in range(len(names)):
                row_str += f"{matrix[i, j]:^{cell_width}}"
            lines.append(row_str)

        # Indent all lines for better log readability
        return "\n".join(["  " + line for line in lines])

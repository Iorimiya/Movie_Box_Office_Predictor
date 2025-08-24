from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from typing_extensions import override

from src.core.project_config import ProjectPaths, ProjectModelType
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
    """

    training_week_len: int
    calculate_trend_accuracy: bool
    calculate_range_accuracy: bool
    box_office_ranges: tuple[int, ...] = (1_000_000, 10_000_000, 90_000_000)


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
    MSE loss, trend accuracy, range accuracy, and F1-score.
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
        artifacts_path = ProjectPaths.get_model_root_path(
            model_id=model_id, model_type=ProjectModelType.PREDICTION
        )
        model_file_path = artifacts_path / f"{model_id}_{model_epoch:04d}.keras"

        data_processor = PredictionDataProcessor(model_artifacts_path=artifacts_path)
        if not data_processor.scaler:
            raise FileNotFoundError(f"Could not load scaler artifact from: {artifacts_path}")

        model_core = PredictionModelCore(model_path=model_file_path)
        return data_processor, model_core, artifacts_path

    @override
    def _prepare_test_data(
        self, data_processor: PredictionDataProcessor, config: PredictionEvaluationConfig
    ) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
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
        data_source = PredictionDataSource(dataset_name=config.dataset_name)
        raw_data = data_processor.load_raw_data(source=data_source)

        processing_config = PredictionDataConfig(
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
        x_test: NDArray[np.float32],
        y_test: NDArray[np.float64],
        config: PredictionEvaluationConfig
    ) -> dict[str, Optional[float]]:
        """
        Calculates various performance metrics based on the evaluation configuration.

        This method orchestrates the calculation of different metrics such as
        MSE loss, trend accuracy, range accuracy, and F1-score. It un-scales
        predictions and actual values as needed for certain metrics.

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

        if config.calculate_loss:
            metrics['test_loss'] = self._calculate_mse_loss(model_core, x_test, y_test)

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
                metrics['trend_accuracy'] = self._calculate_trend_accuracy(
                    predictions=unscaled_pred, actual=unscaled_actual, last_inputs=unscaled_inputs
                )
            if config.calculate_range_accuracy:
                metrics['test_accuracy'] = self._calculate_range_accuracy(
                    predictions=unscaled_pred, actual=unscaled_actual, ranges=config.box_office_ranges
                )
            if config.calculate_f1_score:
                metrics['f1_score'] = self._calculate_f1_score(
                    predictions=unscaled_pred, actual=unscaled_actual, config=config
                )
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
        self, model_core: PredictionModelCore, x_test: NDArray[np.float32], y_test: NDArray[np.float64]
    ) -> float:
        """
        Calculates the Mean Squared Error (MSE) loss on the test set.

        :param model_core: The model to evaluate.
        :param x_test: The test features.
        :param y_test: The test labels.
        :returns: The MSE loss value.
        """
        self.logger.info("Step 4a: Calculating MSE loss on the test set...")
        eval_config = PredictionEvaluateConfig(verbose=0)
        loss: float = model_core.evaluate(x_test=x_test, y_test=y_test, config=eval_config)
        self.logger.info(f"  - Test MSE Loss: {loss:.6f}")
        return loss

    def _get_unscaled_predictions(
        self, model_core: PredictionModelCore, scaler: MinMaxScaler,
        x_test: NDArray[np.float32], y_test: NDArray[np.float64]
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
        predict_config = PredictionPredictConfig(verbose=0)
        y_pred_scaled = model_core.predict(data=x_test, config=predict_config)

        unscaled_predictions = scaler.inverse_transform(y_pred_scaled).flatten().tolist()
        unscaled_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten().tolist()

        # Unscale the last box office value from each input sequence
        last_week_input_scaled = x_test[:, -1, 0].reshape(-1, 1)
        unscaled_last_week_inputs = scaler.inverse_transform(last_week_input_scaled).flatten().tolist()

        return unscaled_predictions, unscaled_actual, unscaled_last_week_inputs

    @staticmethod
    def _get_range_index(value: float, ranges: tuple[int, ...]) -> int:
        """
        Determines the index of the range a given value falls into.

        :param value: The box office value to classify.
        :param ranges: A tuple of upper boundaries defining the ranges.
        :returns: The integer index of the corresponding range.
        """
        # This logic is now centralized.
        thresholds: list[float] = [-float('inf')] + sorted(list(ranges)) + [float('inf')]
        for i in range(len(thresholds) - 1):
            if thresholds[i] <= value < thresholds[i + 1]:
                return i
        # Handle edge case where value might be exactly the last boundary or infinity
        return len(thresholds) - 2

    def _calculate_trend_accuracy(
        self, predictions: list[float], actual: list[float], last_inputs: list[float]
    ) -> float:
        """
        Calculates the trend prediction accuracy.

        This metric measures how often the model correctly predicts whether the
        box office will increase or decrease compared to the last known week.

        :param predictions: A list of unscaled predicted box office values.
        :param actual: A list of unscaled actual box office values.
        :param last_inputs: A list of unscaled box office values from the last input week.
        :returns: The trend accuracy, a float between 0.0 and 1.0.
        """
        correct_predictions = 0
        for pred, actual, last_input in zip(predictions, actual, last_inputs):
            pred_trend = 1 if pred > last_input else 0
            actual_trend = 1 if actual > last_input else 0
            if pred_trend == actual_trend:
                correct_predictions += 1
        accuracy = correct_predictions / len(predictions) if predictions else 0.0
        self.logger.info(f"  - Trend Accuracy: {accuracy:.2%}")
        return accuracy

    def _calculate_range_accuracy(
        self, predictions: list[float], actual: list[float], ranges: tuple[int, ...]
    ) -> float:
        """
        Calculates the range prediction accuracy.

        This metric measures how often the predicted box office value falls into
        the same predefined revenue range as the actual value.

        :param predictions: A list of unscaled predicted box office values.
        :param actual: A list of unscaled actual box office values.
        :param ranges: A tuple of upper boundaries defining the box office ranges.
        :returns: The range accuracy, a float between 0.0 and 1.0.
        """
        correct_predictions: int = 0
        for pred, actual in zip(predictions, actual):
            if self._get_range_index(pred, ranges) == self._get_range_index(actual, ranges):
                correct_predictions += 1
        accuracy: float = correct_predictions / len(predictions) if predictions else 0.0
        self.logger.info(f"  - Range Accuracy: {accuracy:.2%}")
        return accuracy

    def _calculate_f1_score(
        self, predictions: list[float], actual: list[float], config: PredictionEvaluationConfig
    ) -> float:
        """
        Calculates the F1-score for the range prediction task.

        This treats the range prediction as a multi-class classification problem
        and computes the F1-score based on the method specified in the config.

        :param predictions: A list of unscaled predicted box office values.
        :param actual: A list of unscaled actual box office values.
        :param config: The evaluation configuration, used to access box office ranges
                       and the F1 averaging method.
        :returns: The calculated F1-score.
        """
        y_pred_labels: list[int] = [self._get_range_index(p, config.box_office_ranges) for p in predictions]
        y_true_labels: list[int] = [self._get_range_index(a, config.box_office_ranges) for a in actual]

        score: float = f1_score(y_true_labels, y_pred_labels, average=config.f1_average_method, zero_division=0)
        self.logger.info(f"  - F1-Score (average='{config.f1_average_method}'): {score:.4f}")
        return score

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from numpy.typing import NDArray
from sklearn.metrics import f1_score
from typing_extensions import override

from src.core.project_config import ProjectPaths, ProjectModelType
from src.models.base.base_evaluator import BaseEvaluator, BaseEvaluationResult, BaseEvaluationConfig
from src.models.base.keras_setup import keras_base
from src.models.sentiment.components.data_processor import (
    SentimentTrainingProcessedData,
    SentimentDataProcessor,
    SentimentDataSource, SentimentDataConfig
)
from src.models.sentiment.components.model_core import (
    SentimentEvaluateConfig,
    SentimentModelCore,
    SentimentPredictConfig
)

History = keras_base.callbacks.History


@dataclass(frozen=True)
class SentimentEvaluationConfig(BaseEvaluationConfig):
    """
    Configuration for running a sentiment model evaluation.

    Inherits common evaluation parameters from BaseEvaluationConfig.

    :ivar vocabulary_size: The vocabulary size used during original training.
    :ivar calculate_accuracy: Flag to calculate accuracy on the test set.
    """
    vocabulary_size: int
    calculate_accuracy: bool
    f1_average_method: str = 'binary'


@dataclass(frozen=True)
class SentimentEvaluationResult(BaseEvaluationResult):  # <-- Inherit from BaseEvaluationResult
    """
    A structured result of a sentiment model evaluation run.

    Inherits common fields from BaseEvaluationResult.

    :ivar test_accuracy: The accuracy calculated on the test set.
    """
    test_accuracy: float


class SentimentEvaluator(
    BaseEvaluator[SentimentDataProcessor, SentimentModelCore, SentimentEvaluationConfig, SentimentEvaluationResult]
):
    """
    Evaluates a trained sentiment analysis model.

    This evaluator loads a specific epoch of a trained model, its tokenizer,
    and its training history. It then calculates performance metrics like loss,
    accuracy, and F1-score on a test set, and combines them with the
    historical training/validation loss for a comprehensive report.
    """

    @override
    def _setup_components(
        self, model_id: str, model_epoch: int
    ) -> tuple[SentimentDataProcessor, SentimentModelCore, Path]:
        """
        Sets up and loads the necessary data processor and model core.

        :param model_id: The unique identifier for the model series.
        :param model_epoch: The specific training epoch of the model to load.
        :returns: A tuple containing the initialized data processor, model core,
                  and the path to the model artifacts directory.
        :raises FileNotFoundError: If the tokenizer artifact cannot be found.
        """
        self.logger.info("Step 1: Loading model and data processor artifacts...")
        model_artifacts_path: Path = ProjectPaths.get_model_root_path(
            model_id=model_id, model_type=ProjectModelType.SENTIMENT
        )
        model_file_path: Path = model_artifacts_path / f"{model_id}_{model_epoch:04d}.keras"

        data_processor = SentimentDataProcessor(model_artifacts_path=model_artifacts_path)
        model_core = SentimentModelCore(model_path=model_file_path)

        if not data_processor.tokenizer:
            raise FileNotFoundError(
                f"Could not load tokenizer from path: {model_artifacts_path}"
            )
        return data_processor, model_core, model_artifacts_path

    @override
    def _prepare_test_data(
        self,
        data_processor: SentimentDataProcessor,
        config: SentimentEvaluationConfig

    ) -> tuple[NDArray[any], NDArray[any]]:
        """
        Loads and processes data to retrieve the test set.

        This method supports two modes based on `config.evaluate_on_full_dataset`:
        - If False (default), it reproduces the original test split.
        - If True, it processes the entire dataset as a single evaluation set.

        :param data_processor: The initialized data processor.
        :param config: The configuration object for the evaluation run.
        :returns: A tuple containing the evaluation features (x_eval) and labels (y_eval).
        """
        self.logger.info("Step 3: Loading and processing evaluation dataset...")
        data_source = SentimentDataSource(file_name=config.dataset_name)
        raw_data = data_processor.load_raw_data(source=data_source)

        if config.evaluate_on_full_dataset:
            # --- Mode 2: Exploratory (New Dataset) ---
            self.logger.info("Evaluation mode: Processing the full dataset as the test set.")
            x_eval, y_eval = data_processor.process_for_evaluation(raw_data=raw_data)
            return x_eval, y_eval
        else:
            # --- Mode 1: Reproducibility (Original Dataset) ---
            self.logger.info("Evaluation mode: Reproducing the original test split.")

            # Validate that necessary parameters for this mode are present
            if config.split_ratios is None or config.random_state is None:
                raise ValueError(
                    "For reproducibility mode (evaluate_on_full_dataset=False), "
                    "'split_ratios' and 'random_state' must be provided in the configuration."
                )

            # Construct the specific config needed by the data processor
            processing_config = SentimentDataConfig(
                vocabulary_size=config.vocabulary_size,
                split_ratios=config.split_ratios,
                random_state=config.random_state
            )

            # Re-run processing to get the exact same test split
            processed_data: SentimentTrainingProcessedData = data_processor.process_for_training(
                raw_data=raw_data, config=processing_config
            )
            return processed_data['x_test'], processed_data['y_test']

    @override
    def _calculate_metrics(
        self,
        model_core: SentimentModelCore,
        data_processor: SentimentDataProcessor,
        x_test: NDArray[any],
        y_test: NDArray[any],
        config: SentimentEvaluationConfig
    ) -> dict[str, Optional[float]]:
        """
        Calculates loss, accuracy, and F1-score for the sentiment model.
        """
        metrics: dict[str, Optional[float]] = {
            'test_loss': None,
            'test_accuracy': None,
            'f1_score': None
        }

        if config.calculate_loss or config.calculate_accuracy:
            eval_config = SentimentEvaluateConfig(verbose=0)
            eval_results: list[float] = model_core.evaluate(x_test=x_test, y_test=y_test, config=eval_config)
            metrics['test_loss'] = eval_results[0]
            metrics['test_accuracy'] = eval_results[1]
            self.logger.info(f"  - Test Loss: {metrics['test_loss']:.4f}")
            self.logger.info(f"  - Test Accuracy: {metrics['test_accuracy']:.4f}")

        if config.calculate_f1_score:
            predict_config = SentimentPredictConfig(verbose=0)
            y_pred_probs: NDArray[any] = model_core.predict(data=x_test, config=predict_config)
            y_pred_labels: NDArray[any] = (y_pred_probs > 0.5).astype("int32")
            f1 = f1_score(
                y_true=y_test,
                y_pred=y_pred_labels,
                average=config.f1_average_method,
                zero_division=0
            )
            metrics['f1_score'] = f1
            self.logger.info(f"  - F1-Score (average='{config.f1_average_method}'): {f1:.4f}")

        return metrics

    @override
    def _compile_final_result(
        self,
        config: SentimentEvaluationConfig,
        metrics: dict[str, Optional[float]],
        training_history: list[float],
        validation_history: list[float]
    ) -> SentimentEvaluationResult:
        """
        Compiles the final result object for the sentiment evaluation.
        """
        return SentimentEvaluationResult(
            model_id=config.model_id,
            model_epoch=config.model_epoch,
            test_loss=metrics.get('test_loss'),
            test_accuracy=metrics.get('test_accuracy'),
            f1_score=metrics.get('f1_score'),
            training_loss_history=training_history,
            validation_loss_history=validation_history
        )

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from keras.src.callbacks import History
from numpy.typing import NDArray
from sklearn.metrics import f1_score
from typing_extensions import override

from src.core.project_config import ProjectPaths, ProjectModelType
from src.data_handling.file_io import PickleFile
from src.models.base.base_evaluator import BaseEvaluator
from src.models.sentiment.components.data_processor import (
    SentimentTrainingProcessedData,
    SentimentDataProcessor,
    SentimentDataSource,
    SentimentDataConfig
)
from src.models.sentiment.components.model_core import (
    SentimentEvaluateConfig,
    SentimentModelCore,
    SentimentPredictConfig
)


@dataclass(frozen=True)
class SentimentEvaluationConfig:
    """
    Configuration for running a sentiment model evaluation.

    :ivar model_id: The unique identifier for the model series.
    :ivar model_epoch: The specific training epoch of the model to evaluate.
    :ivar evaluation_dataset_file_name: The name of the CSV file in the
                                        `sentiment_analysis_resources` directory
                                        to use for evaluation.
    :ivar data_processing_config: The configuration that was used for the
                                  original data processing during training.
                                  This is needed to ensure the test split is
                                  recreated identically.
    """
    model_id: str
    model_epoch: int
    evaluation_dataset_file_name: str
    data_processing_config: SentimentDataConfig


@dataclass(frozen=True)
class SentimentEvaluationResult:
    """
    A structured result of a sentiment model evaluation run.

    :ivar model_id: The unique identifier for the evaluated model series.
    :ivar model_epoch: The specific epoch of the evaluated model.
    :ivar test_loss: The loss calculated on the test set.
    :ivar test_accuracy: The accuracy calculated on the test set.
    :ivar f1_score: The F1-score calculated on the test set.
    :ivar training_loss_history: A list of training loss values from each epoch
                                 of the original training run.
    :ivar validation_loss_history: A list of validation loss values from each
                                   epoch of the original training run.
    """
    model_id: str
    model_epoch: int
    test_loss: float
    test_accuracy: float
    f1_score: float
    training_loss_history: list[float]
    validation_loss_history: list[float]


@dataclass(frozen=True)
class _TestMetrics:
    """
    A private dataclass to hold the calculated metrics from the test set.

    This structure serves as an internal data container to pass evaluation
    results within the SentimentEvaluator.

    :ivar loss: The calculated loss value on the test set.
    :ivar accuracy: The calculated accuracy on the test set.
    :ivar f1: The calculated F1-score on the test set.
    """
    loss: float
    accuracy: float
    f1: float


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
    HISTORY_FILE_NAME: Final[str] = "training_history.pkl"

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
        model_file_path: Path = model_artifacts_path / f"{model_id}_{model_epoch}.keras"

        data_processor = SentimentDataProcessor(model_artifacts_path=model_artifacts_path)
        model_core = SentimentModelCore(model_path=model_file_path)

        if not data_processor.tokenizer:
            raise FileNotFoundError(
                f"Could not load tokenizer from path: {model_artifacts_path}"
            )
        return data_processor, model_core, model_artifacts_path

    def load_training_history(self, history_file_path: Path) -> tuple[list[float], list[float]]:
        """
        Loads the training and validation loss history from a pickle file.

        :param history_file_path: The path to the 'training_history.pkl' file.
        :returns: A tuple containing the training loss list and validation loss list.
        """
        self.logger.info(f"Step 2: Loading training history from '{history_file_path}'...")
        history_loader = PickleFile(path=history_file_path)
        history: History = history_loader.load()
        training_loss: list[float] = history.history.get('loss', [])
        validation_loss: list[float] = history.history.get('val_loss', [])
        return training_loss, validation_loss

    def _prepare_test_data(
        self,
        data_processor: SentimentDataProcessor,
        file_name: str,
        processing_config: SentimentDataConfig
    ) -> tuple[NDArray[any], NDArray[any]]:
        """
        Loads and processes data to retrieve the test set.

        :param data_processor: The initialized data processor.
        :param file_name: The name of the evaluation dataset file.
        :param processing_config: The configuration used for the original training run.
        :returns: A tuple containing the test features (x_test) and labels (y_test).
        """
        self.logger.info("Step 3: Loading and processing evaluation dataset...")
        data_source = SentimentDataSource(file_name=file_name)
        raw_data = data_processor.load_raw_data(source=data_source)

        # Re-run processing to get the exact same test split
        processed_data: SentimentTrainingProcessedData = data_processor.process_for_training(
            raw_data=raw_data, config=processing_config
        )
        return processed_data['x_test'], processed_data['y_test']

    def _calculate_test_metrics(
        self, model_core: SentimentModelCore, x_test: NDArray[any], y_test: NDArray[any]
    ) -> _TestMetrics:
        """
        Calculates loss, accuracy, and F1-score on the test set.

        :param model_core: The loaded model core.
        :param x_test: The test features.
        :param y_test: The test labels.
        :returns: A dataclass containing the calculated metrics.
        """
        self.logger.info("Step 4: Calculating metrics on the test set...")
        # Loss and Accuracy
        eval_config = SentimentEvaluateConfig(verbose=0)
        eval_results: list[float] = model_core.evaluate(x_test=x_test, y_test=y_test, config=eval_config)
        test_loss: float = eval_results[0]
        test_accuracy: float = eval_results[1]
        self.logger.info(f"  - Test Loss: {test_loss:.4f}")
        self.logger.info(f"  - Test Accuracy: {test_accuracy:.4f}")

        # F1-Score
        predict_config = SentimentPredictConfig(verbose=0)
        y_pred_probs: NDArray[any] = model_core.predict(data=x_test, config=predict_config)
        y_pred_labels: NDArray[any] = (y_pred_probs > 0.5).astype("int32")
        f1: float = f1_score(y_true=y_test, y_pred=y_pred_labels, average='binary')
        self.logger.info(f"  - F1-Score: {f1:.4f}")

        return _TestMetrics(loss=test_loss, accuracy=test_accuracy, f1=f1)

    @override
    def run(self, config: SentimentEvaluationConfig) -> SentimentEvaluationResult:
        """
        Executes the evaluation pipeline for the sentiment model.

        :param config: The configuration object for the evaluation run.
        :returns: A structured result object containing all evaluation metrics.
        :raises FileNotFoundError: If the model, tokenizer, or history artifacts
                                   for the specified model_id and epoch are not found.
        """
        self.logger.info(
            f"--- Starting SENTIMENT evaluation for model '{config.model_id}' at epoch {config.model_epoch} ---"
        )

        data_processor, model_core, artifacts_path = self._setup_components(
            model_id=config.model_id, model_epoch=config.model_epoch
        )

        history_path = artifacts_path / self.HISTORY_FILE_NAME
        training_loss, validation_loss = self.load_training_history(history_file_path=history_path)

        x_test, y_test = self._prepare_test_data(
            data_processor=data_processor,
            file_name=config.evaluation_dataset_file_name,
            processing_config=config.data_processing_config
        )

        metrics = self._calculate_test_metrics(model_core=model_core, x_test=x_test, y_test=y_test)

        self.logger.info("Step 5: Compiling final evaluation results...")
        final_result = SentimentEvaluationResult(
            model_id=config.model_id,
            model_epoch=config.model_epoch,
            test_loss=metrics.loss,
            test_accuracy=metrics.accuracy,
            f1_score=metrics.f1,
            training_loss_history=training_loss,
            validation_loss_history=validation_loss
        )

        self.logger.info("--- SENTIMENT evaluation finished ---")
        return final_result

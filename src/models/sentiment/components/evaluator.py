from dataclasses import dataclass

from typing_extensions import override


from src.models.base.base_evaluator import BaseEvaluator
from src.models.sentiment.components.data_processor import (
    SentimentDataProcessor,
    SentimentTrainingConfig
)
from src.models.sentiment.components.model_core import (
    SentimentModelCore
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
    data_processing_config: SentimentTrainingConfig


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
    def run(self, config: SentimentEvaluationConfig) -> SentimentEvaluationResult:
        """
        Executes the evaluation pipeline for the sentiment model.

        :param config: The configuration object for the evaluation run.
        :returns: A structured result object containing all evaluation metrics.
        :raises FileNotFoundError: If the model, tokenizer, or history artifacts
                                   for the specified model_id and epoch are not found.
        """

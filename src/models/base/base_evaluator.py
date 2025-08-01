from abc import ABC, abstractmethod
from logging import Logger
from typing import  Generic, TypeVar

from src.core.logging_manager import LoggingManager
from src.models.base.base_data_processor import BaseDataProcessor
from src.models.base.base_model_core import BaseModelCore

DataProcessorType = TypeVar('DataProcessorType', bound=BaseDataProcessor)
ModelCoreType = TypeVar('ModelCoreType', bound=BaseModelCore)
EvaluationConfigType = TypeVar('EvaluationConfigType')
EvaluationResultType = TypeVar('EvaluationResultType')


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
    def run(self, config: EvaluationConfigType) -> EvaluationResultType:
        """
        Executes the evaluation pipeline based on the provided configuration.

        Subclasses MUST implement this method to define the specific sequence
        of operations for an evaluation run. This typically involves:
        1. Determining the path to the trained model artifacts.
        2. Instantiating and loading the appropriate DataProcessor and ModelCore.
        3. Loading and processing the evaluation dataset.
        4. Executing one or more evaluation strategies (e.g., calculating loss,
           accuracy, F1-score).
        5. Compiling the results into a structured format.

        :param config: A structured configuration object containing all necessary
                       parameters for the evaluation run.
        :returns: A structured object or dictionary containing the evaluation results.
        """
        pass

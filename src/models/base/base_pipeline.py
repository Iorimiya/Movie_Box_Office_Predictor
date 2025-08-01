from abc import ABC, abstractmethod
from logging import Logger
from typing import Generic, TypeVar

from src.core.logging_manager import LoggingManager
from src.models.base.base_data_processor import BaseDataProcessor
from src.models.base.base_model_core import BaseModelCore

DataProcessorType = TypeVar('DataProcessorType', bound=BaseDataProcessor)
ModelCoreType = TypeVar('ModelCoreType', bound=BaseModelCore)
PipelineConfigType = TypeVar('PipelineConfigType')


class BaseTrainingPipeline(
    Generic[DataProcessorType, ModelCoreType, PipelineConfigType],
    ABC
):
    """
    An abstract base class for a model training pipeline.

    This class orchestrates the end-to-end training process by coordinating
    a DataProcessor and a ModelCore. It uses dependency injection to receive
    these components, promoting modularity and testability. The pipeline's
    behavior is driven by a generic configuration object.

    :ivar logger: A logger instance for logging pipeline progress.
    :ivar data_processor: An instance of a DataProcessor subclass.
    :ivar model_core: An instance of a ModelCore subclass.
    """
    logger: Logger
    data_processor: DataProcessorType
    model_core: ModelCoreType

    def __init__(self, data_processor: DataProcessorType, model_core: ModelCoreType) -> None:
        """
        Initializes the BaseTrainingPipeline with its required components.

        :param data_processor: An instance of a class that inherits from BaseDataProcessor.
        :param model_core: An instance of a class that inherits from BaseModelCore.
        """
        self.logger = LoggingManager().get_logger('machine_learning')
        self.data_processor = data_processor
        self.model_core = model_core

    @abstractmethod
    def run(self, config: PipelineConfigType) -> None:
        """
        Executes the training pipeline based on the provided configuration.

        Subclasses MUST implement this method to define the specific sequence
        of operations for a training run. This typically involves:
        1. Creating data source and processing configuration objects.
        2. Calling the data_processor to load and process data.
        3. Creating a model build configuration object.
        4. Calling the model_core to build or load the model.
        5. Creating a model training configuration object.
        6. Calling the model_core to train the model.
        7. Saving all resulting artifacts (model, scaler, tokenizer, etc.).

        :param config: A structured configuration object containing all necessary
                       parameters for the entire training run.
        """
        pass

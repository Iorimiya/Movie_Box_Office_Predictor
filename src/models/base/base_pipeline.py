from abc import ABC, abstractmethod
from logging import Logger
from pathlib import Path
from typing import Generic, TypeVar, Optional

from keras.src.callbacks import History
from src.data_handling.file_io import PickleFile

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

    @abstractmethod
    def _check_required_artifacts_for_continuation(self) -> None:
        """
        Checks if all necessary artifacts for continuing training are loaded.

        Subclasses must implement this to verify their specific artifacts
        (e.g., tokenizer, scaler) are present in the data processor.

        :raises FileNotFoundError: If a required artifact is not found.
        """
        pass

    @abstractmethod
    def _create_model_core(self, model_path: Path) -> ModelCoreType:
        """
        Creates a specific ModelCore instance from a saved model file.

        :param model_path: The path to the .keras model file.
        :returns: An instance of the specific ModelCore subclass.
        """
        pass

    def _setup_for_continuation(
        self,
        artifacts_folder: Path,
        model_id: str,
        continue_from_epoch: int
    ) -> ModelCoreType:
        """
        Handles the common logic for setting up a continued training run.

        This template method loads artifacts, checks for their presence,
        locates the model checkpoint, and re-initializes the model core.

        :param artifacts_folder: The root directory for model artifacts.
        :param model_id: The unique identifier of the model.
        :param continue_from_epoch: The epoch number to continue from.
        :returns: A new, loaded ModelCore instance.
        :raises FileNotFoundError: If artifacts or the model checkpoint are not found.
        """
        self.logger.info(f"Setting up for continued training from epoch {continue_from_epoch}...")

        # Load data processor artifacts
        self.data_processor.load_artifacts()

        # Delegate specific artifact check to subclass
        self._check_required_artifacts_for_continuation()

        # Common model path logic
        model_to_load_path = artifacts_folder / f"{model_id}_{continue_from_epoch:04d}.keras"
        if not model_to_load_path.exists():
            raise FileNotFoundError(f"Checkpoint to continue from not found: {model_to_load_path}")

        # Delegate specific ModelCore creation to subclass
        loaded_model_core = self._create_model_core(model_path=model_to_load_path)
        self.logger.info(f"Successfully loaded model from: {model_to_load_path}")

        return loaded_model_core

    @abstractmethod
    def _get_history_filename(self) -> str:
        """
        Returns the specific filename for the training history pickle file.

        :returns: The name of the history file (e.g., "training_history.pkl").
        """
        pass

    def _merge_histories(
        self,
        new_history: History,
        history_save_path: Path,
        continue_from_epoch: Optional[int]
    ) -> History:
        """
        Merges a new training history with an existing one if applicable.

        This base implementation handles the standard merging logic. Subclasses
        can override this to add specific metrics (like F1 scores).

        :param new_history: The History object from the latest training run.
        :param history_save_path: The path to the history file.
        :param continue_from_epoch: The epoch number the training continued from.
        :returns: The final, potentially merged, History object to be saved.
        """
        if continue_from_epoch and history_save_path.exists():
            self.logger.info(f"Loading existing history from {history_save_path} to append new results.")
            old_history: History = PickleFile(path=history_save_path).load()
            for key, value in new_history.history.items():
                old_history.history.setdefault(key, []).extend(value)
            return old_history
        else:
            return new_history

    # --- NEW TEMPLATE METHOD FOR ARTIFACT SAVING ---
    def _save_run_artifacts(
        self,
        config: PipelineConfigType,
        history: History,
        artifacts_folder: Path,
        continue_from_epoch: Optional[int]
    ) -> None:
        """
        Handles the common logic for saving all artifacts at the end of a run.

        This template method saves the training history, data processor artifacts,
        and the final model state.

        :param config: The master configuration object for the run.
        :param history: The History object from the completed training.
        :param artifacts_folder: The root directory for model artifacts.
        :param continue_from_epoch: The epoch number the training continued from, if any.
        """
        self.logger.info("Saving all run artifacts...")

        # 1. Save History (delegating merging logic)
        history_filename: str = self._get_history_filename()
        history_save_path: Path = artifacts_folder / history_filename
        history_to_save: History = self._merge_histories(
            new_history=history,
            history_save_path=history_save_path,
            continue_from_epoch=continue_from_epoch
        )
        PickleFile(path=history_save_path).save(data=history_to_save)
        self.logger.info(f"Training history saved to: {history_save_path}")

        # 2. Save Data Processor Artifacts (only on a new run)
        if not continue_from_epoch:
            self.data_processor.save_artifacts()
            self.logger.info(f"Data processor artifacts saved in: {self.data_processor.model_artifacts_path}")

        # 3. Save Final Model State (if not already saved by a checkpoint)
        # We need to access model_id and epochs from the config, which requires a bit of care
        # since PipelineConfigType is a generic. We assume it has these attributes.
        final_model_save_path: Path = artifacts_folder / f"{config.model_id}_{config.epochs:04d}.keras"
        if not final_model_save_path.exists():
            self.model_core.save(file_path=final_model_save_path)
            self.logger.info(f"Final model state for epoch {config.epochs} saved to: {final_model_save_path}")


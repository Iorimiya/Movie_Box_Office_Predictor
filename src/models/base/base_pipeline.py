from abc import ABC, abstractmethod
from logging import Logger
from pathlib import Path
from typing import Any, Final, Generic, Optional, TypeAlias, TypeVar

from src.core.logging_manager import LoggingManager
from src.data_handling.file_io import PickleFile
from src.models.base.base_data_processor import BaseDataProcessor
from src.models.base.base_model_core import BaseModelCore
from src.models.base.callbacks import F1ScoreHistory
from src.models.base.keras_setup import keras_base

# noinspection PyUnresolvedReferences
History: TypeAlias = keras_base.callbacks.History
# noinspection PyUnresolvedReferences
ModelCheckpoint: TypeAlias = keras_base.callbacks.ModelCheckpoint
# noinspection PyUnresolvedReferences
EarlyStopping: TypeAlias = keras_base.callbacks.EarlyStopping
# noinspection PyUnresolvedReferences
Callback: TypeAlias = keras_base.callbacks.Callback

DataProcessorType = TypeVar('DataProcessorType', bound=BaseDataProcessor)
ModelCoreType = TypeVar('ModelCoreType', bound=BaseModelCore)
PipelineConfigType = TypeVar('PipelineConfigType')


class BaseTrainingPipeline(
    Generic[DataProcessorType, ModelCoreType, PipelineConfigType],
    ABC
):
    """
    An abstract base class for a model training pipeline.

    This class defines the basic structure and shared utilities for training models.
    It does not define a specific 'run' workflow, leaving that to domain-specific
    intermediate classes or concrete subclasses.

    :ivar HISTORY_FILE_NAME: The fixed filename for the training history file.
    :ivar _logger: The logger instance for the pipeline.
    :ivar _data_processor: The data processor instance.
    :ivar _model_core: The model core instance.
    """

    HISTORY_FILE_NAME: Final[str] = "training_history.pkl"

    _logger: Logger
    _data_processor: DataProcessorType
    _model_core: ModelCoreType

    def __init__(self, data_processor: DataProcessorType, model_core: ModelCoreType) -> None:
        """
        Initializes the BaseTrainingPipeline with its required components.

        :param data_processor: The data processor responsible for handling artifacts and data loading.
        :param model_core: The model core responsible for the Keras model lifecycle.
        """
        self._logger: Logger = LoggingManager().get_logger('machine_learning')
        self._data_processor: DataProcessorType = data_processor
        self._model_core: ModelCoreType = model_core

    @abstractmethod
    def run(self, config: PipelineConfigType, continue_from_epoch: Optional[int] = None) -> None:
        """
        Executes the training pipeline.

        :param config: The configuration object for the pipeline.
        :param continue_from_epoch: Optional epoch number to resume training from.
        """
        pass

    @abstractmethod
    def _check_required_artifacts_for_continuation(self) -> None:
        """
        Checks if the required artifacts for continuing training are available.

        :raises FileNotFoundError: If required artifacts are missing.
        """
        pass

    @abstractmethod
    def _create_model_core(self, model_path: Path) -> ModelCoreType:
        """
        Creates a specific ModelCore instance from a saved model file.

        :param model_path: Path to the saved Keras model file.
        :return: A new instance of ModelCore with the loaded model.
        """
        pass

    def _setup_for_continuation(self, artifacts_folder: Path, model_id: str, continue_from_epoch: int) -> ModelCoreType:
        """
        Handles the common logic for setting up a continued training run.

        :param artifacts_folder: The directory where artifacts are stored.
        :param model_id: The identifier for the model.
        :param continue_from_epoch: The epoch number to resume from.
        :return: A loaded ModelCore instance ready for training.
        :raises FileNotFoundError: If the specified checkpoint file does not exist.
        """
        self._logger.info(f"Setting up for continued training from epoch {continue_from_epoch}...")

        # Load data processor artifacts
        self._data_processor.load_artifacts()
        self._check_required_artifacts_for_continuation()

        model_to_load_path: Path = artifacts_folder / f"{model_id}_{continue_from_epoch:04d}.keras"
        if not model_to_load_path.exists():
            raise FileNotFoundError(f"Checkpoint to continue from not found: {model_to_load_path}")

        loaded_model_core: ModelCoreType = self._create_model_core(model_path=model_to_load_path)
        self._logger.info(f"Successfully loaded model from: {model_to_load_path}")

        return loaded_model_core

    def get_history_filename(self) -> str:
        """
        Returns the filename for the training history.

        :return: The history filename.
        """
        return self.HISTORY_FILE_NAME

    def _merge_histories(
        self, new_history: History, history_save_path: Path, continue_from_epoch: Optional[int], **kwargs: Any
    ) -> dict[str, Any]:
        """
        Merges a new training history with an existing one if applicable.

        :param new_history: The history object from the recent training run.
        :param history_save_path: The path to the existing history file.
        :param continue_from_epoch: The epoch from which training was resumed.
        :param kwargs: Additional arguments, such as 'f1_history_callback'.
        :return: A dictionary containing the merged history.
        """
        history_to_save: dict[str, Any]

        # Handle custom callback data (like F1 score) passed via kwargs
        f1_history_callback: Optional[F1ScoreHistory] = kwargs.get('f1_history_callback')
        if f1_history_callback:
            new_history.history['val_f1_score'] = f1_history_callback.f1_scores

        if continue_from_epoch and history_save_path.exists():
            self._logger.info(f"Loading existing history from {history_save_path} to append new results.")
            old_history_data: dict[str, Any] = PickleFile(path=history_save_path).load()
            for key, value in new_history.history.items():
                if key not in old_history_data:
                    old_history_data[key] = []
                old_history_data[key].extend(value)
            history_to_save = old_history_data
        else:
            history_to_save = new_history.history

        return history_to_save

    def _save_run_artifacts(
        self,
        config: PipelineConfigType,
        history: History,
        artifacts_folder: Path,
        callbacks: list[Callback],
        continue_from_epoch: Optional[int],
        **kwargs: Any
    ) -> None:
        """
        Handles the common logic for saving all artifacts at the end of a run.

        :param config: The configuration object for the pipeline.
        :param history: The training history.
        :param artifacts_folder: The directory to save artifacts.
        :param callbacks: The list of Keras callbacks used during training.
        :param continue_from_epoch: The epoch from which training was resumed.
        :param kwargs: Additional arguments for history merging.
        """
        self._logger.info("Saving all run artifacts...")

        # Determine the correct final epoch for saving
        final_epoch: int
        early_stopping_callback: Optional[EarlyStopping] = next(
            (cb for cb in callbacks if isinstance(cb, EarlyStopping)),
            None
        )

        if early_stopping_callback and early_stopping_callback.stopped_epoch > 0:
            final_epoch = early_stopping_callback.best_epoch + 1
            self._logger.info(f"Early stopping triggered. Best model at epoch {final_epoch}.")
        else:
            epochs_run: int = len(history.history.get(key='loss', default=[]))
            final_epoch = epochs_run + (continue_from_epoch or 0)
            self._logger.info(f"Training completed. Final model at epoch {final_epoch}.")

        # Save Final Model State
        model_id: str = getattr(config, 'model_id', 'model')
        final_model_save_path: Path = artifacts_folder / f"{model_id}_{final_epoch:04d}.keras"
        self._model_core.save(file_path=final_model_save_path)
        self._logger.info(f"Final model state saved to: {final_model_save_path}")

        # Save History
        history_filename: str = self.get_history_filename()
        history_save_path: Path = artifacts_folder / history_filename
        history_to_save: dict[str, Any] = self._merge_histories(
            new_history=history,
            history_save_path=history_save_path,
            continue_from_epoch=continue_from_epoch,
            **kwargs
        )
        PickleFile(path=history_save_path).save(data=history_to_save)

        # Save Data Processor Artifacts (only on a new run)
        if not continue_from_epoch:
            self._data_processor.save_artifacts()

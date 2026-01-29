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

    This class orchestrates the end-to-end training process by coordinating
    a DataProcessor and a ModelCore. It uses dependency injection to receive
    these components, promoting modularity and testability. The pipeline's
    behavior is driven by a generic configuration object.

    :ivar logger: A logger instance for logging pipeline progress.
    :ivar data_processor: An instance of a DataProcessor subclass.
    :ivar model_core: An instance of a ModelCore subclass.
    """

    HISTORY_FILE_NAME: Final[str] = "training_history.pkl"

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

    def get_history_filename(self) -> str:
        """
        Returns the specific filename for the training history pickle file.

        :returns: The name of the history file (e.g., "training_history.pkl").
        """
        return self.HISTORY_FILE_NAME

    def _merge_histories(
        self,
        new_history: History,
        history_save_path: Path,
        continue_from_epoch: Optional[int],
        **kwargs: Any
    ) -> dict[str, Any]:
        """
        Merges a new training history with an existing one if applicable.

        This implementation is enhanced to handle custom callback data, such as
        F1 scores from an F1ScoreHistory callback, passed via kwargs.

        :param new_history: The History object from the latest training run.
        :param history_save_path: The path to the history file.
        :param continue_from_epoch: The epoch number the training continued from.
        :param kwargs: Catches extra arguments passed from subclasses, like 'f1_history_callback'.
        :returns: The final, potentially merged, history dictionary to be saved.
        """
        history_to_save: dict[str, Any]

        # The new_history.history object might not contain val_f1_score if it was never triggered
        # in the first epoch. We should get it from the callback directly.
        f1_history_callback: Optional[F1ScoreHistory] = kwargs.get('f1_history_callback')
        if f1_history_callback:
            new_history.history['val_f1_score'] = f1_history_callback.f1_scores

        if continue_from_epoch and history_save_path.exists():
            self.logger.info(f"Loading existing history from {history_save_path} to append new results.")
            old_history_data: dict[str, Any] = PickleFile(path=history_save_path).load()
            for key, value in new_history.history.items():
                if key not in old_history_data:
                    old_history_data[key] = []
                old_history_data[key].extend(value)
            history_to_save = old_history_data
        else:
            history_to_save = new_history.history

        return history_to_save

    # noinspection PyTypeHints
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

        This template method saves the training history, data processor artifacts,
        and the final model state. It intelligently determines the correct epoch
        number for the saved model, especially when EarlyStopping is used.

        :param config: The master configuration object for the run.
        :param history: The History object from the completed training.
        :param artifacts_folder: The root directory for model artifacts.
        :param callbacks: The list of Keras callbacks used during training.
        :param continue_from_epoch: The epoch number the training continued from, if Any.
        :param kwargs: Extra arguments to be passed to helper methods like _merge_histories.
        """
        self.logger.info("Saving all run artifacts...")

        # Determine the correct final epoch for saving
        final_epoch: int
        early_stopping_callback: Optional[EarlyStopping] = next(
            (cb for cb in callbacks if isinstance(cb, EarlyStopping)),  # Search in the provided list
            None
        )

        if early_stopping_callback and early_stopping_callback.stopped_epoch > 0:
            # Early stopping was triggered. The model weights were restored to the best epoch.
            # The 'best_epoch' attribute is 0-indexed, so we add 1 for the filename.
            final_epoch = early_stopping_callback.best_epoch + 1
            self.logger.info(
                f"Early stopping was triggered. The best model was at epoch {final_epoch}. "
                f"Saving model with this epoch number."
            )
        else:
            # Training completed all epochs without early stopping.
            # The number of epochs run is the length of the 'loss' history list.
            epochs_run: int = len(history.history.get('loss', []))
            final_epoch = epochs_run + (continue_from_epoch or 0)
            self.logger.info(
                f"Training completed all configured epochs. Saving final model for epoch {final_epoch}."
            )

        # Save Final Model State
        # We save the model that is currently in memory. If early stopping with
        # restore_best_weights=True was used, this is the best model.
        # We assume the config object has 'model_id'
        model_id: str = getattr(config, 'model_id', 'model')
        final_model_save_path: Path = artifacts_folder / f"{model_id}_{final_epoch:04d}.keras"
        self.model_core.save(file_path=final_model_save_path)
        self.logger.info(f"Final model state saved to: {final_model_save_path}")

        # --- Save History ---
        history_filename: str = self.get_history_filename()
        history_save_path: Path = artifacts_folder / history_filename
        history_to_save: dict[str, Any] = self._merge_histories(
            new_history=history,
            history_save_path=history_save_path,
            continue_from_epoch=continue_from_epoch,
            **kwargs
        )
        PickleFile(path=history_save_path).save(data=history_to_save)
        self.logger.info(f"Training history saved to: {history_save_path}")

        # --- Save Data Processor Artifacts (only on a new run) ---
        if not continue_from_epoch:
            self.data_processor.save_artifacts()
            self.logger.info(f"Data processor artifacts saved in: {self.data_processor.model_artifacts_path}")

    def _setup_checkpoint_callback(
        self,
        config: PipelineConfigType,
        num_train_samples: int,
        artifacts_folder: Path
    ) -> Optional[ModelCheckpoint]:
        """
        Sets up the ModelCheckpoint callback based on the pipeline configuration.

        This helper method centralizes the logic for creating a checkpoint
        callback, calculating the save frequency based on the number of
        training samples and the specified interval in epochs.

        :param config: The master configuration object for the run, which must
                       have `checkpoint_interval`, `batch_size`, and `model_id` attributes.
        :param num_train_samples: The number of samples in the training dataset.
        :param artifacts_folder: The directory where checkpoint files will be saved.
        :returns: A configured `ModelCheckpoint` instance if checkpointing is
                  enabled in the config, otherwise `None`.
        """
        if not getattr(config, 'checkpoint_interval', None):
            return None

        save_frequency_in_batches: int | str
        if num_train_samples == 0:
            self.logger.warning("Training data is empty. Checkpoint callback will fall back to saving every epoch.")
            save_frequency_in_batches = 'epoch'
        else:
            # Calculate steps per epoch using ceiling division
            batch_size: int = getattr(config, 'batch_size')
            checkpoint_interval: int = getattr(config, 'checkpoint_interval')
            steps_per_epoch: int = (num_train_samples + batch_size - 1) // batch_size
            save_frequency_in_batches = checkpoint_interval * steps_per_epoch
            self.logger.info(
                f"Checkpoint interval of {checkpoint_interval} epochs "
                f"translates to a save frequency of {save_frequency_in_batches} batches "
                f"(steps_per_epoch: {steps_per_epoch})."
            )

        model_id: str = getattr(config, 'model_id')
        checkpoint_filepath: Path = artifacts_folder / f"{model_id}_{{epoch:04d}}.keras"
        model_checkpoint_callback: ModelCheckpoint = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            save_freq=save_frequency_in_batches,
            verbose=1
        )

        self.logger.info(
            f"Model checkpointing enabled. Saving every {getattr(config, 'checkpoint_interval')} epochs."
        )
        return model_checkpoint_callback

    def _setup_early_stopping_callback(self, config: PipelineConfigType) -> Optional[EarlyStopping]:
        """
        Sets up the EarlyStopping callback based on the pipeline configuration.

        This helper method centralizes the logic for creating an early stopping
        callback if it is enabled in the configuration.

        :param config: The master configuration object for the run, which must
                       have `early_stopping_patience`, `early_stopping_monitor`,
                       and `early_stopping_min_delta` attributes.
        :returns: A configured `EarlyStopping` instance if enabled, otherwise `None`.
        """
        patience: Optional[int] = getattr(config, 'early_stopping_patience', None)
        if patience is None:
            return None

        monitor: str = getattr(config, 'early_stopping_monitor', 'val_loss')
        min_delta: float = getattr(config, 'early_stopping_min_delta', 0.0)
        mode: str = 'max' if 'f1' in monitor or 'accuracy' in monitor else 'auto'

        self.logger.info(
            f"Early stopping enabled: monitoring '{monitor}' with patience={patience}."
        )
        early_stopping_callback: EarlyStopping = EarlyStopping(
            monitor=monitor,
            patience=patience,
            min_delta=min_delta,
            verbose=1,
            mode=mode,
            restore_best_weights=True  # Always restore best weights
        )
        return early_stopping_callback

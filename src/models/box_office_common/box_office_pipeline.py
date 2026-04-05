from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

from numpy.typing import NDArray
from typing_extensions import override

from src.core.project_config import ProjectModelType, ProjectPaths
from src.models.base.base_model_core import BaseModelCore
from src.models.base.base_pipeline import BaseTrainingPipeline, Callback, EarlyStopping, History, ModelCheckpoint
from src.models.base.callbacks import F1ScoreHistory
from src.models.box_office_common.box_office_data_processor import (
    BoxOfficeDataConfig,
    BoxOfficeDataSource,
    BoxOfficeSessionDataProcessor
)

DataProcessorType = TypeVar('DataProcessorType', bound=BoxOfficeSessionDataProcessor)
ModelCoreType = TypeVar('ModelCoreType', bound=BaseModelCore)
PipelineConfigType = TypeVar('PipelineConfigType')


class BoxOfficeTrainingPipeline(
    Generic[DataProcessorType, ModelCoreType, PipelineConfigType],
    BaseTrainingPipeline[DataProcessorType, ModelCoreType, PipelineConfigType],
    ABC
):
    """
    A domain-specific intermediate class for Box Office training pipelines.

    This class provides the template 'run' method that is tailored for the
    sequence-based movie box office prediction domain.
    """

    @override
    def run(self, config: PipelineConfigType, continue_from_epoch: Optional[int] = None) -> None:
        """
        The template method defining the skeleton of the box office training process.
        """
        model_id: str = getattr(config, 'model_id', 'unknown')
        self.logger.debug(f"Starting training pipeline for model: {model_id}")

        # 1. Setup Artifacts Folder
        artifacts_folder: Path = ProjectPaths.get_model_root_path(
            model_id=model_id, model_type=self._get_model_type()
        )
        artifacts_folder.mkdir(parents=True, exist_ok=True)

        # 2. Setup for Continuation (if applicable)
        if continue_from_epoch:
            self.model_core = self._setup_for_continuation(
                artifacts_folder=artifacts_folder,
                model_id=model_id,
                continue_from_epoch=continue_from_epoch
            )

        # 3. Data Loading and Processing
        self.logger.debug("Loading and processing data...")
        data_config: BoxOfficeDataConfig = self._create_data_config(config)
        data_source: BoxOfficeDataSource = self._create_data_source(config)

        # Box Office Specific: load_raw_data requires data_config for session length
        raw_data = self.data_processor.load_raw_data(source=data_source, config=data_config)
        processed_data = self.data_processor.process_for_training(raw_data=raw_data, config=data_config)

        # 4. Task-Specific Data Preparation (e.g., Label Transformation)
        x_train, y_train, x_val, y_val = self._prepare_training_data(processed_data, config)

        # 5. Build Model (if new run)
        if not continue_from_epoch:
            num_features: int = x_train.shape[2]
            build_config = self._create_build_config(config, num_features)
            self.model_core.build(config=build_config)
            self.logger.debug("Model building complete.")

        # 6. Callbacks Orchestration
        monitoring_callbacks: list[Callback] = []

        f1_callback = self._setup_f1_score_callback(config, (x_val, y_val))
        if f1_callback:
            monitoring_callbacks.append(f1_callback)

        early_stopping = self._setup_early_stopping_callback(config)
        if early_stopping:
            monitoring_callbacks.append(early_stopping)

        checkpoint = self._setup_checkpoint_callback(config, len(x_train), artifacts_folder)
        if checkpoint:
            monitoring_callbacks.append(checkpoint)

        # 7. Model Training
        self.logger.debug("Starting model training...")
        fit_params = self._create_fit_params(config, (x_val, y_val), monitoring_callbacks, continue_from_epoch or 0)

        history: History = self.model_core.train(x_train=x_train, y_train=y_train, params=fit_params)
        self.logger.debug("Model training complete.")

        # 8. Save Artifacts
        self._save_run_artifacts(
            config=config,
            history=history,
            artifacts_folder=artifacts_folder,
            callbacks=monitoring_callbacks,
            continue_from_epoch=continue_from_epoch,
            f1_history_callback=f1_callback
        )
        self.logger.debug(f"Training pipeline for {model_id} finished successfully.")

    # --- Hook Methods to be implemented by Task-Specific Subclasses ---

    @abstractmethod
    def _get_model_type(self) -> ProjectModelType:
        pass

    @abstractmethod
    def _create_data_config(self, config: PipelineConfigType) -> BoxOfficeDataConfig:
        pass

    @abstractmethod
    def _create_data_source(self, config: PipelineConfigType) -> BoxOfficeDataSource:
        pass

    @abstractmethod
    def _prepare_training_data(self, processed_data: Any, config: PipelineConfigType) -> tuple[
        NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Returns (x_train, y_train, x_val, y_val)"""
        pass

    @abstractmethod
    def _create_build_config(self, config: PipelineConfigType, num_features: int) -> Any:
        pass

    @abstractmethod
    def _create_fit_params(self, config: PipelineConfigType, validation_data: tuple, callbacks: list,
                           initial_epoch: int) -> Any:
        pass

    @abstractmethod
    def _setup_f1_score_callback(self, config: PipelineConfigType, validation_data: tuple) -> Optional[F1ScoreHistory]:
        pass

    @staticmethod
    def _setup_checkpoint_callback(
        config: PipelineConfigType,
        num_train_samples: int,
        artifacts_folder: Path
    ) -> Optional[ModelCheckpoint]:
        interval = getattr(config, 'checkpoint_interval', None)
        if not interval:
            return None

        batch_size: int = getattr(config, 'batch_size')
        steps_per_epoch: int = (num_train_samples + batch_size - 1) // batch_size
        save_freq = interval * steps_per_epoch

        model_id: str = getattr(config, 'model_id')
        checkpoint_filepath: Path = artifacts_folder / f"{model_id}_{{epoch:04d}}.keras"
        return ModelCheckpoint(filepath=checkpoint_filepath, save_freq=save_freq, verbose=1)

    @staticmethod
    def _setup_early_stopping_callback(config: PipelineConfigType) -> Optional[EarlyStopping]:
        patience: Optional[int] = getattr(config, 'early_stopping_patience', None)
        if patience is None:
            return None

        monitor: str = getattr(config, 'early_stopping_monitor', 'val_loss')
        min_delta: float = getattr(config, 'early_stopping_min_delta', 0.0)
        mode: str = 'max' if 'f1' in monitor or 'accuracy' in monitor else 'auto'

        return EarlyStopping(monitor=monitor, patience=patience, min_delta=min_delta, mode=mode,
                             restore_best_weights=True)

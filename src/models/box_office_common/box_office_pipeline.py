from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

from numpy.typing import NDArray
from typing_extensions import override

from src.core.project_config import ProjectModelType, ProjectPaths
from src.data_handling.dataset import DatabaseDataset, YamlDataset
from src.models.base.base_model_core import BaseModelCore
from src.models.base.base_pipeline import BaseTrainingPipeline, Callback, EarlyStopping, History, ModelCheckpoint
from src.models.base.callbacks import F1ScoreHistory
from src.models.box_office_common.box_office_data_processor import (
    BoxOfficeDataConfig,
    BoxOfficeDataSource,
    BoxOfficeSessionDataProcessor
)
from src.models.box_office_common.box_office_evaluator import DataSourceType

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

    :ivar _data_source_type: The type of data source to use (Database or YAML).
    """
    _data_source_type: DataSourceType

    def __init__(
        self,
        data_processor: DataProcessorType,
        model_core: ModelCoreType,
        data_source_type: DataSourceType = DataSourceType.DATABASE
    ) -> None:
        """
        Initializes the BoxOfficeTrainingPipeline.

        :param data_processor: The data processor instance.
        :param model_core: The model core instance.
        :param data_source_type: The type of data source to use (Database or YAML).
        """
        super().__init__(data_processor=data_processor, model_core=model_core)
        self._data_source_type: DataSourceType = data_source_type

    @override
    def run(self, config: PipelineConfigType, continue_from_epoch: Optional[int] = None) -> None:
        """
        The template method defining the skeleton of the box office training process.

        :param config: The configuration object for the pipeline.
        :param continue_from_epoch: Optional epoch number to resume training from.
        """
        model_id: str = getattr(config, 'model_id', 'unknown')
        self._logger.debug(f"Starting training pipeline for model: {model_id}")

        # 1. Setup Artifacts Folder
        artifacts_folder: Path = ProjectPaths.get_model_root_path(
            model_id=model_id, model_type=self.model_type
        )
        artifacts_folder.mkdir(parents=True, exist_ok=True)

        # 2. Setup for Continuation (if applicable)
        if continue_from_epoch:
            # Re-assign the model core with the loaded one for continuation
            self._model_core = self._setup_for_continuation(
                artifacts_folder=artifacts_folder,
                model_id=model_id,
                continue_from_epoch=continue_from_epoch
            )

        # 3. Data Loading and Processing
        self._logger.debug("Loading and processing data...")
        data_config: BoxOfficeDataConfig = self._create_data_config(config=config)
        data_source: BoxOfficeDataSource = self._create_data_source(config=config)

        # Box Office Specific: load_raw_data requires data_config for session length
        raw_data: Any = self._data_processor.load_raw_data(source=data_source, config=data_config)
        processed_data: Any = self._data_processor.process_for_training(raw_data=raw_data, config=data_config)

        # 4. Task-Specific Data Preparation (e.g., Label Transformation)
        x_train: NDArray[Any]
        y_train: NDArray[Any]
        x_val: NDArray[Any]
        y_val: NDArray[Any]
        x_train, y_train, x_val, y_val = self._prepare_training_data(processed_data=processed_data, config=config)

        # 5. Build Model (if new run)
        if not continue_from_epoch:
            num_features: int = x_train.shape[2]
            build_config: Any = self._create_build_config(config=config, num_features=num_features)
            self._model_core.build(config=build_config)
            self._logger.debug("Model building complete.")

        # 6. Callbacks Orchestration
        monitoring_callbacks: list[Callback] = []

        f1_callback: Optional[F1ScoreHistory] = self._setup_f1_score_callback(
            config=config, validation_data=(x_val, y_val)
        )
        if f1_callback:
            monitoring_callbacks.append(f1_callback)

        early_stopping: Optional[EarlyStopping] = self._setup_early_stopping_callback(config=config)
        if early_stopping:
            monitoring_callbacks.append(early_stopping)

        checkpoint: Optional[ModelCheckpoint] = self._setup_checkpoint_callback(
            config=config,
            num_train_samples=len(x_train),
            artifacts_folder=artifacts_folder
        )
        if checkpoint:
            monitoring_callbacks.append(checkpoint)

        # 7. Model Training
        self._logger.debug("Starting model training...")
        fit_params: Any = self._create_fit_params(
            config=config,
            validation_data=(x_val, y_val),
            callbacks=monitoring_callbacks,
            initial_epoch=continue_from_epoch or 0
        )

        history: History = self._model_core.train(x_train=x_train, y_train=y_train, params=fit_params)
        self._logger.debug("Model training complete.")

        # 8. Save Artifacts
        self._save_run_artifacts(
            config=config,
            history=history,
            artifacts_folder=artifacts_folder,
            callbacks=monitoring_callbacks,
            continue_from_epoch=continue_from_epoch,
            f1_history_callback=f1_callback
        )
        self._logger.debug(f"Training pipeline for {model_id} finished successfully.")

    @property
    @abstractmethod
    def model_type(self) -> ProjectModelType:
        """
        Returns the type of the model associated with this pipeline.
        """
        pass

    @abstractmethod
    def _create_data_config(self, config: PipelineConfigType) -> BoxOfficeDataConfig:
        """
        Creates a data configuration object from the pipeline configuration.
        """
        pass

    def _create_data_source(self, config: PipelineConfigType) -> BoxOfficeDataSource:
        """
        Creates a data source instance based on the _data_source_type.

        :param config: The pipeline configuration containing the dataset name.
        :return: An instance of DatabaseDataset or YamlDataset.
        """
        dataset_name: str = getattr(config, 'dataset_name', 'unknown')
        if self._data_source_type == DataSourceType.YAML:
            return YamlDataset(name=dataset_name)
        return DatabaseDataset(name=dataset_name)

    @abstractmethod
    def _prepare_training_data(
        self, processed_data: Any, config: PipelineConfigType
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
        """
        Transforms processed data into final features and labels for training/validation.

        :param processed_data: The output from the data processor's training method.
        :param config: The pipeline configuration.
        :return: A tuple of (x_train, y_train, x_val, y_val).
        """
        pass

    @abstractmethod
    def _create_build_config(self, config: PipelineConfigType, num_features: int) -> Any:
        """
        Creates a model building configuration.
        """
        pass

    @abstractmethod
    def _create_fit_params(
        self,
        config: PipelineConfigType,
        validation_data: tuple[NDArray[Any], NDArray[Any]],
        callbacks: list[Callback],
        initial_epoch: int
    ) -> Any:
        """
        Creates the training parameters object.
        """
        pass

    @abstractmethod
    def _setup_f1_score_callback(
        self, config: PipelineConfigType, validation_data: tuple[NDArray[Any], NDArray[Any]]
    ) -> Optional[F1ScoreHistory]:
        """
        Sets up the F1 score monitoring callback if required.
        """
        pass

    @staticmethod
    def _setup_checkpoint_callback(
        config: PipelineConfigType, num_train_samples: int, artifacts_folder: Path
    ) -> Optional[ModelCheckpoint]:
        """
        Sets up the model checkpointing callback.

        :param config: The pipeline configuration.
        :param num_train_samples: Total number of training samples.
        :param artifacts_folder: Path where checkpoints will be saved.
        :return: A ModelCheckpoint instance or None.
        """
        interval: Optional[int] = getattr(config, 'checkpoint_interval', None)
        if not interval:
            return None

        batch_size: int = getattr(config, 'batch_size')
        steps_per_epoch: int = (num_train_samples + batch_size - 1) // batch_size
        save_freq: int = interval * steps_per_epoch

        model_id: str = getattr(config, 'model_id')
        checkpoint_filepath: Path = artifacts_folder / f"{model_id}_{{epoch:04d}}.keras"
        return ModelCheckpoint(filepath=checkpoint_filepath, save_freq=save_freq, verbose=1)

    @staticmethod
    def _setup_early_stopping_callback(config: PipelineConfigType) -> Optional[EarlyStopping]:
        """
        Sets up the early stopping callback.

        :param config: The pipeline configuration.
        :return: An EarlyStopping instance or None.
        """
        patience: Optional[int] = getattr(config, 'early_stopping_patience', None)
        if patience is None:
            return None

        monitor: str = getattr(config, 'early_stopping_monitor', 'val_loss')
        min_delta: float = getattr(config, 'early_stopping_min_delta', 0.0)
        mode: str = 'max' if 'f1' in monitor or 'accuracy' in monitor else 'auto'

        return EarlyStopping(
            monitor=monitor,
            patience=patience,
            min_delta=min_delta,
            mode=mode,
            restore_best_weights=True
        )

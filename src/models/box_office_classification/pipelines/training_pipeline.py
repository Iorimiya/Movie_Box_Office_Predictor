from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from numpy.typing import NDArray
from typing_extensions import override

from src.core.project_config import ProjectModelType
from src.data_handling.dataset import DatabaseDataset
from src.models.base.base_pipeline import Callback
from src.models.base.callbacks import F1ScoreHistory
from src.models.base.keras_setup import keras_base
from src.models.box_office_classification.components.data_processor import (
    BoxOfficeClassificationDataProcessor,
    BoxOfficeClassificationTrainingProcessedData,
)
from src.models.box_office_classification.components.model_core import (
    BoxOfficeClassificationBuildConfig,
    BoxOfficeClassificationModelCore,
    BoxOfficeClassificationTrainParams,
)
from src.models.box_office_common.box_office_data_processor import BoxOfficeDataConfig, BoxOfficeDataSource
from src.models.box_office_common.box_office_pipeline import BoxOfficeTrainingPipeline
from src.utilities.metrics import MultiClassClassificationMetricsCalculator

# noinspection PyUnresolvedReferences
to_categorical = keras_base.utils.to_categorical


@dataclass(frozen=True)
class BoxOfficeClassificationPipelineConfig:
    """
    Represents the master configuration for a Box Office Classification Model training run.
    """
    model_id: str
    dataset_name: str
    training_week_len: int
    split_ratios: tuple[int, int, int]
    lstm_units: int
    dense_units: int
    dropout_rate: float
    num_classes: int
    learning_rate: float
    clipnorm: float
    epochs: int
    batch_size: int
    random_state: int
    checkpoint_interval: int | None = None
    early_stopping_patience: Optional[int] = None
    early_stopping_monitor: str = 'val_loss'
    early_stopping_min_delta: float = 0.001
    f1_average_method: str = 'macro'


class BoxOfficeClassificationTrainingPipeline(
    BoxOfficeTrainingPipeline[
        BoxOfficeClassificationDataProcessor,
        BoxOfficeClassificationModelCore,
        BoxOfficeClassificationPipelineConfig
    ]
):
    """
    Orchestrates the end-to-end training process for the Box Office Classification Model.
    """

    @override
    def _get_model_type(self) -> ProjectModelType:
        return ProjectModelType.BOX_OFFICE_CLASSIFICATION

    @override
    def _create_data_config(self, config: BoxOfficeClassificationPipelineConfig) -> BoxOfficeDataConfig:
        return BoxOfficeDataConfig(
            training_week_len=config.training_week_len,
            split_ratios=config.split_ratios,
            random_state=config.random_state
        )

    @override
    def _create_data_source(self, config: BoxOfficeClassificationPipelineConfig) -> BoxOfficeDataSource:
        return DatabaseDataset(name=config.dataset_name)

    @override
    def _prepare_training_data(
        self,
        processed_data: BoxOfficeClassificationTrainingProcessedData,
        config: BoxOfficeClassificationPipelineConfig
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
        # Specific to Classification: Convert labels to One-hot
        y_train_cat = to_categorical(processed_data['y_train'], num_classes=config.num_classes)
        y_val_cat = to_categorical(processed_data['y_val'], num_classes=config.num_classes)
        return processed_data['x_train'], y_train_cat, processed_data['x_val'], y_val_cat

    @override
    def _create_build_config(self, config: BoxOfficeClassificationPipelineConfig,
                             num_features: int) -> BoxOfficeClassificationBuildConfig:
        return BoxOfficeClassificationBuildConfig(
            input_shape=(config.training_week_len, num_features),
            lstm_units=config.lstm_units,
            dense_units=config.dense_units,
            dropout_rate=config.dropout_rate,
            num_classes=config.num_classes,
            learning_rate=config.learning_rate,
            clipnorm=config.clipnorm
        )

    @override
    def _create_fit_params(
        self,
        config: BoxOfficeClassificationPipelineConfig,
        validation_data: tuple[NDArray[Any], NDArray[Any]],
        callbacks: list[Callback],
        initial_epoch: int
    ) -> BoxOfficeClassificationTrainParams:
        return BoxOfficeClassificationTrainParams(
            epochs=config.epochs,
            batch_size=config.batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            initial_epoch=initial_epoch
        )

    @override
    def _setup_f1_score_callback(
        self,
        config: BoxOfficeClassificationPipelineConfig,
        validation_data: tuple[NDArray[Any], NDArray[Any]]
    ) -> Optional[F1ScoreHistory]:
        if 'f1' not in config.early_stopping_monitor:
            return None

        self.logger.debug("F1 score monitoring is enabled. Setting up F1ScoreHistory callback.")
        metrics_calculator = MultiClassClassificationMetricsCalculator(f1_average_method=config.f1_average_method)
        return F1ScoreHistory(validation_data=validation_data, metrics_calculator=metrics_calculator)

    @override
    def _check_required_artifacts_for_continuation(self) -> None:
        if not self.data_processor.is_prepared:
            raise FileNotFoundError("DataProcessor is not prepared for continued training.")

    @override
    def _create_model_core(self, model_path: Path) -> BoxOfficeClassificationModelCore:
        return BoxOfficeClassificationModelCore(model_path=model_path)

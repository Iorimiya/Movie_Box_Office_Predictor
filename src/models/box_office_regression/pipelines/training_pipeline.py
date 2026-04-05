from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler
from typing_extensions import override

from src.core.project_config import ProjectModelType
from src.data_handling.dataset import DatabaseDataset
from src.models.base.base_pipeline import Callback
from src.models.base.callbacks import F1ScoreHistory
from src.models.box_office_common.box_office_data_processor import BoxOfficeDataConfig, BoxOfficeDataSource
from src.models.box_office_common.box_office_pipeline import BoxOfficeTrainingPipeline
from src.models.box_office_regression.components.data_processor import (
    BoxOfficeRegressionDataProcessor,
    BoxOfficeRegressionTrainingProcessedData
)
from src.models.box_office_regression.components.model_core import (
    BoxOfficeRegressionBuildConfig,
    BoxOfficeRegressionFitParams,
    BoxOfficeRegressionModelCore
)
from src.utilities.metrics import PointwiseClassificationMetricsCalculator


@dataclass(frozen=True)
class BoxOfficeRegressionPipelineConfig:
    """
    Represents the master configuration for a Box Office Regression Model training run.
    """
    model_id: str
    dataset_name: str
    training_week_len: int
    split_ratios: tuple[int, int, int]
    lstm_units: int
    dropout_rate: float
    epochs: int
    batch_size: int
    random_state: int
    checkpoint_interval: int | None = None
    early_stopping_patience: Optional[int] = None
    early_stopping_monitor: str = 'val_loss'
    early_stopping_min_delta: float = 0.001
    box_office_ranges: Optional[tuple[int, ...]] = None
    f1_average_method: str = 'macro'


class BoxOfficeRegressionTrainingPipeline(
    BoxOfficeTrainingPipeline[
        BoxOfficeRegressionDataProcessor,
        BoxOfficeRegressionModelCore,
        BoxOfficeRegressionPipelineConfig
    ]
):
    """
    Orchestrates the end-to-end training process for the Box Office Regression Model.
    """

    @override
    def _get_model_type(self) -> ProjectModelType:
        return ProjectModelType.BOX_OFFICE_REGRESSION

    @override
    def _create_data_config(self, config: BoxOfficeRegressionPipelineConfig) -> BoxOfficeDataConfig:
        return BoxOfficeDataConfig(
            training_week_len=config.training_week_len,
            split_ratios=config.split_ratios,
            random_state=config.random_state
        )

    @override
    def _create_data_source(self, config: BoxOfficeRegressionPipelineConfig) -> BoxOfficeDataSource:
        return DatabaseDataset(name=config.dataset_name)

    @override
    def _prepare_training_data(
        self,
        processed_data: BoxOfficeRegressionTrainingProcessedData,
        config: BoxOfficeRegressionPipelineConfig
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
        # For Regression, we use the processed data as-is.
        return processed_data['x_train'], processed_data['y_train'], processed_data['x_val'], processed_data['y_val']

    @override
    def _create_build_config(self, config: BoxOfficeRegressionPipelineConfig,
                             num_features: int) -> BoxOfficeRegressionBuildConfig:
        return BoxOfficeRegressionBuildConfig(
            input_shape=(config.training_week_len, num_features),
            lstm_units=config.lstm_units,
            dropout_rate=config.dropout_rate
        )

    @override
    def _create_fit_params(
        self,
        config: BoxOfficeRegressionPipelineConfig,
        validation_data: tuple[NDArray[Any], NDArray[Any]],
        callbacks: list[Callback],
        initial_epoch: int
    ) -> BoxOfficeRegressionFitParams:
        return BoxOfficeRegressionFitParams(
            epochs=config.epochs,
            batch_size=config.batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            initial_epoch=initial_epoch
        )

    @override
    def _setup_f1_score_callback(
        self,
        config: BoxOfficeRegressionPipelineConfig,
        validation_data: tuple[NDArray[Any], NDArray[Any]]
    ) -> Optional[F1ScoreHistory]:
        if 'f1' not in config.early_stopping_monitor:
            return None

        self.logger.debug("F1 score monitoring is enabled. Setting up F1ScoreHistory callback.")

        if not self.data_processor.scaler or config.box_office_ranges is None:
            self.logger.error("Cannot set up F1 score monitoring. Ensure scaler is loaded and ranges are configured.")
            return None

        scaler: MinMaxScaler = self.data_processor.scaler
        ranges: tuple[int, ...] = config.box_office_ranges

        def value_to_label_fn(value: float) -> int:
            unscaled_value: float = scaler.inverse_transform([[value]])[0][0]
            return BoxOfficeRegressionDataProcessor.get_range_index(value=unscaled_value, ranges=ranges)

        metrics_calculator = PointwiseClassificationMetricsCalculator(
            value_to_label_fn=value_to_label_fn,
            f1_average_method=config.f1_average_method
        )

        return F1ScoreHistory(validation_data=validation_data, metrics_calculator=metrics_calculator)

    @override
    def _check_required_artifacts_for_continuation(self) -> None:
        if not self.data_processor.is_prepared:
            raise FileNotFoundError("DataProcessor is not prepared for continued training.")

    @override
    def _create_model_core(self, model_path: Path) -> BoxOfficeRegressionModelCore:
        return BoxOfficeRegressionModelCore(model_path=model_path)

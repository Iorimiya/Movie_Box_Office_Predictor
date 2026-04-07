from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler
from typing_extensions import override

from src.core.project_config import ProjectModelType
from src.models.base.base_pipeline import Callback
from src.models.base.callbacks import F1ScoreHistory
from src.models.box_office_common.box_office_data_processor import BoxOfficeDataConfig
from src.models.box_office_common.box_office_evaluator import DataSourceType
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

    :ivar model_id: The unique identifier for the model series.
    :ivar dataset_name: The name of the source structured dataset.
    :ivar training_week_len: The number of past weeks used as input.
    :ivar split_ratios: Ratios for data splitting (train, val, test).
    :ivar lstm_units: The number of units in the LSTM layer.
    :ivar dropout_rate: The dropout rate to apply after the LSTM layer.
    :ivar epochs: The number of epochs for training.
    :ivar batch_size: The batch size for training.
    :ivar random_state: Seed for the random number generator.
    :ivar checkpoint_interval: The interval at which to save model checkpoints.
    :ivar early_stopping_patience: Number of epochs to wait for improvement.
    :ivar early_stopping_monitor: Metric to monitor for early stopping.
    :ivar early_stopping_min_delta: Minimum change to qualify as improvement.
    :ivar box_office_ranges: Upper boundaries for classification ranges.
    :ivar f1_average_method: Averaging method for F1 score calculation.
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

    :ivar _data_processor: The data processor for regression data.
    :ivar _model_core: The core model architecture.
    """
    _data_processor: BoxOfficeRegressionDataProcessor
    _model_core: BoxOfficeRegressionModelCore

    def __init__(
        self,
        data_processor: BoxOfficeRegressionDataProcessor,
        model_core: BoxOfficeRegressionModelCore,
        data_source_type: DataSourceType = DataSourceType.DATABASE
    ) -> None:
        """
        Initializes the BoxOfficeRegressionTrainingPipeline.

        :param data_processor: The data processor instance.
        :param model_core: The model core instance.
        :param data_source_type: The type of data source to use (Database or YAML).
        """
        super().__init__(data_source_type=data_source_type, data_processor=data_processor, model_core=model_core)

    @override
    def model_type(self) -> ProjectModelType:
        """
        Returns the specific model type for this pipeline.

        :return: ProjectModelType.BOX_OFFICE_REGRESSION.
        """
        return ProjectModelType.BOX_OFFICE_REGRESSION

    @override
    def _create_data_config(self, config: BoxOfficeRegressionPipelineConfig) -> BoxOfficeDataConfig:
        """
        Creates a data configuration for the regression model.

        :param config: The master pipeline configuration.
        :return: A BoxOfficeDataConfig instance.
        """
        return BoxOfficeDataConfig(
            training_week_len=config.training_week_len,
            split_ratios=config.split_ratios,
            random_state=config.random_state
        )

    @override
    def _prepare_training_data(
        self,
        processed_data: BoxOfficeRegressionTrainingProcessedData,
        config: BoxOfficeRegressionPipelineConfig
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
        """
        Prepares features and labels for regression training.

        :param processed_data: The scaled and split dataset from the processor.
        :param config: The master pipeline configuration.
        :return: A tuple of (x_train, y_train, x_val, y_val).
        """
        return processed_data['x_train'], processed_data['y_train'], processed_data['x_val'], processed_data['y_val']

    @override
    def _create_build_config(
        self, config: BoxOfficeRegressionPipelineConfig, num_features: int
    ) -> BoxOfficeRegressionBuildConfig:
        """
        Creates a configuration for building the Keras model.

        :param config: The master pipeline configuration.
        :param num_features: The number of features in the input sequence.
        :return: A BoxOfficeRegressionBuildConfig instance.
        """
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
        """
        Creates the parameters for the Keras fit method.

        :param config: The master pipeline configuration.
        :param validation_data: The data used for validation.
        :param callbacks: A list of Keras callbacks.
        :param initial_epoch: The starting epoch number.
        :return: A BoxOfficeRegressionFitParams instance.
        """
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
        """
        Configures an F1 score monitoring callback if required by early stopping.

        :param config: The master pipeline configuration.
        :param validation_data: The data used for computing the F1 score.
        :return: An F1ScoreHistory instance or None.
        """
        if 'f1' not in config.early_stopping_monitor:
            return None

        self._logger.debug("F1 score monitoring is enabled. Setting up F1ScoreHistory callback.")

        if not self._data_processor.scaler or config.box_office_ranges is None:
            self._logger.error("Cannot set up F1 score monitoring. Ensure scaler is loaded and ranges are configured.")
            return None

        scaler: MinMaxScaler = self._data_processor.scaler
        ranges: tuple[int, ...] = config.box_office_ranges

        def value_to_label_fn(value: float) -> int:
            unscaled_value: float = scaler.inverse_transform(X=[[value]])[0][0]
            return BoxOfficeRegressionDataProcessor.get_range_index(value=unscaled_value, ranges=ranges)

        metrics_calculator: PointwiseClassificationMetricsCalculator = PointwiseClassificationMetricsCalculator(
            value_to_label_fn=value_to_label_fn,
            f1_average_method=config.f1_average_method
        )

        return F1ScoreHistory(validation_data=validation_data, metrics_calculator=metrics_calculator)

    @override
    def _check_required_artifacts_for_continuation(self) -> None:
        """
        Ensures the DataProcessor has loaded necessary artifacts to resume training.

        :raises FileNotFoundError: If the DataProcessor is not prepared.
        """
        if not self._data_processor.is_prepared:
            raise FileNotFoundError("DataProcessor is not prepared for continued training.")

    @override
    def _create_model_core(self, model_path: Path) -> BoxOfficeRegressionModelCore:
        """
        Creates a ModelCore instance with a loaded model for resuming.

        :param model_path: Path to the .keras model file.
        :return: A BoxOfficeRegressionModelCore instance.
        """
        return BoxOfficeRegressionModelCore(model_path=model_path)

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler
from typing_extensions import override

from src.core.project_config import ProjectModelType, ProjectPaths
from src.data_handling.movie_collections import MovieSessionData
from src.models.base.base_pipeline import BaseTrainingPipeline
from src.models.base.callbacks import F1ScoreHistory
from src.models.base.keras_setup import keras_base
from src.models.box_office_regression.components.data_processor import (
    BoxOfficeRegressionDataConfig,
    BoxOfficeRegressionDataProcessor,
    BoxOfficeRegressionDataSource,
    BoxOfficeRegressionTrainingProcessedData
)
from src.models.box_office_regression.components.model_core import (
    BoxOfficeRegressionBuildConfig,
    BoxOfficeRegressionFitParams,
    BoxOfficeRegressionModelCore
)
from src.utilities.metrics import PointwiseClassificationMetricsCalculator

# noinspection PyUnresolvedReferences
History = keras_base.callbacks.History
# noinspection PyUnresolvedReferences
ModelCheckpoint = keras_base.callbacks.ModelCheckpoint
# noinspection PyUnresolvedReferences
EarlyStopping = keras_base.callbacks.EarlyStopping


@dataclass(frozen=True)
class BoxOfficeRegressionPipelineConfig:
    """
    Represents the master configuration for a Box Office Regression Model training run.

    This object is typically loaded from an external YAML file and contains all
    necessary parameters to orchestrate the entire training pipeline.

    :ivar model_id: The unique identifier for this model series.
    :ivar dataset_name: The name of the source structured dataset for training data.
    :ivar training_week_len: The number of past weeks to use as input for box_office_regression.
    :ivar split_ratios: A tuple representing the train, validation, and test split ratios.
    :ivar lstm_units: The number of units in the LSTM layer.
    :ivar dropout_rate: The dropout rate to apply after the LSTM layer.
    :ivar epochs: The number of epochs for training.
    :ivar batch_size: The batch size for training.
    :ivar random_state: The seed for the random number generator.
    :ivar checkpoint_interval: The interval in epochs at which to save model checkpoints.
                               If None, only the final model is saved.
    :ivar early_stopping_patience: Number of epochs with no improvement after which training will be stopped.
                                  If None, early stopping is disabled.
    :ivar early_stopping_monitor: Metric to be monitored by early stopping (e.g., 'val_loss', 'val_f1_score').
    :ivar early_stopping_min_delta: Minimum change in the monitored quantity to qualify as an improvement.
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
    BaseTrainingPipeline[
        BoxOfficeRegressionDataProcessor,
        BoxOfficeRegressionModelCore,
        BoxOfficeRegressionPipelineConfig
    ]
):
    """
    Orchestrates the end-to-end training process for the Box Office Regression Model.

    This pipeline coordinates the DataProcessor and ModelCore to execute a
    full training run based on a master configuration file. It handles data
    loading, processing, model building, training, and artifact saving.
    """

    @override
    def run(self, config: BoxOfficeRegressionPipelineConfig, continue_from_epoch: Optional[int] = None) -> None:
        """
        Executes the Box Office Regression Model training pipeline from a configuration file.

        :param config: The master configuration object for this run.
        :param continue_from_epoch: If provided, loads the model from this epoch and continues training.
        :raises FileNotFoundError: If the configuration or required artifacts are not found.
        :raises ValueError: If the configuration file is empty or invalid.
        """
        self.logger.debug(f"Starting BOX OFFICE REGRESSION training pipeline for model: {config.model_id}")
        self.logger.debug(f"Pipeline configured with: {config}")

        master_config: BoxOfficeRegressionPipelineConfig = config

        artifacts_folder: Path = ProjectPaths.get_model_root_path(
            model_id=master_config.model_id, model_type=ProjectModelType.BOX_OFFICE_REGRESSION
        )
        artifacts_folder.mkdir(parents=True, exist_ok=True)

        # Model Building or Loading
        if continue_from_epoch:
            self.logger.debug(f"Setting up for continued training...")
            self.model_core = self._setup_for_continuation(
                artifacts_folder=artifacts_folder,
                model_id=master_config.model_id,
                continue_from_epoch=continue_from_epoch
            )
        else:
            self.logger.debug("This is a new training run.")

        # Data Loading and Processing
        self.logger.debug("Loading and processing data...")

        processing_config: BoxOfficeRegressionDataConfig = BoxOfficeRegressionDataConfig(
            training_week_len=master_config.training_week_len,
            split_ratios=master_config.split_ratios,
            random_state=master_config.random_state
        )

        data_source: BoxOfficeRegressionDataSource = BoxOfficeRegressionDataSource(
            dataset_name=master_config.dataset_name
        )

        raw_data: list[MovieSessionData] = self.data_processor.load_raw_data(source=data_source,
                                                                             config=processing_config)

        processed_data: BoxOfficeRegressionTrainingProcessedData = self.data_processor.process_for_training(
            raw_data=raw_data, config=processing_config
        )
        self.logger.debug("Data processing complete.")

        # Build Model (if new run)
        if not continue_from_epoch:
            num_features: int = processed_data['x_train'].shape[2]
            build_config: BoxOfficeRegressionBuildConfig = BoxOfficeRegressionBuildConfig(
                input_shape=(master_config.training_week_len, num_features),
                lstm_units=master_config.lstm_units,
                dropout_rate=master_config.dropout_rate
            )
            self.model_core.build(config=build_config)
            self.logger.debug("Model building complete.")

        # Model Training
        self.logger.debug("Starting model training...")
        # noinspection PyUnresolvedReferences
        monitoring_callbacks: list[keras_base.callbacks.Callback] = []

        # Setup F1 History callback (if needed)
        f1_history_callback: Optional[F1ScoreHistory] = self._setup_f1_score_callback(
            config=config,
            validation_data=(processed_data['x_val'], processed_data['y_val'])
        )
        if f1_history_callback:
            monitoring_callbacks.append(f1_history_callback)

        # Setup Early Stopping callback
        early_stopping_callback: Optional[EarlyStopping] = self._setup_early_stopping_callback(config=config)
        if early_stopping_callback:
            monitoring_callbacks.append(early_stopping_callback)

        # Setup Checkpoint callback
        checkpoint_callback: Optional[ModelCheckpoint] = self._setup_checkpoint_callback(
            config=config,
            num_train_samples=len(processed_data['x_train']),
            artifacts_folder=artifacts_folder
        )
        if checkpoint_callback:
            monitoring_callbacks.append(checkpoint_callback)

        fit_params: BoxOfficeRegressionFitParams = BoxOfficeRegressionFitParams(
            epochs=master_config.epochs,
            batch_size=master_config.batch_size,
            validation_data=(processed_data['x_val'], processed_data['y_val']),
            callbacks=monitoring_callbacks,
            initial_epoch=continue_from_epoch or 0
        )
        history: History = self.model_core.train(
            x_train=processed_data['x_train'],
            y_train=processed_data['y_train'],
            params=fit_params
        )
        self.logger.debug("Model training complete.")

        self._save_run_artifacts(
            config=master_config,
            history=history,
            artifacts_folder=artifacts_folder,
            callbacks=monitoring_callbacks,
            continue_from_epoch=continue_from_epoch,
            f1_history_callback=f1_history_callback
        )
        self.logger.debug("BOX OFFICE REGRESSION training pipeline finished successfully.")

    @override
    def _check_required_artifacts_for_continuation(self) -> None:
        """
        Checks if the required artifacts for continuing training are available.

        For the Box Office Regression Model, this specifically verifies that the scaler
        has been loaded into the data processor.

        :raises FileNotFoundError: If the scaler artifact is not loaded.
        """
        if not self.data_processor.scaler:
            raise FileNotFoundError(
                f"Could not load scaler for continued training from {self.data_processor.model_artifacts_path}."
            )

    @override
    def _create_model_core(self, model_path: Path) -> BoxOfficeRegressionModelCore:
        """
        Creates a BoxOfficeRegressionModelCore instance from a saved model file.

        :param model_path: The path to the saved Keras model file.
        :returns: Returns the initialized model core instance.
        """
        return BoxOfficeRegressionModelCore(model_path=model_path)

    def _setup_f1_score_callback(
        self,
        config: BoxOfficeRegressionPipelineConfig,
        validation_data: tuple[NDArray[Any], NDArray[Any]]
    ) -> Optional[F1ScoreHistory]:
        """
        Sets up the F1ScoreHistory callback if F1 score is being monitored.

        :param config: The master configuration object for this run.
        :param validation_data: The validation data (x_val, y_val) required by F1ScoreHistory.
        :returns: The F1ScoreHistory instance if needed, otherwise None.
        """
        # If we monitor F1 score, we must add the F1ScoreHistory callback.
        if 'f1' not in config.early_stopping_monitor:
            return None

        self.logger.debug("F1 score monitoring is enabled. Setting up F1ScoreHistory callback.")

        if not self.data_processor.scaler or config.box_office_ranges is None:
            self.logger.error(
                "Cannot set up F1 score monitoring. "
                "Ensure scaler is loaded and 'box_office_ranges' are configured."
            )
            return None

        # Create the function to convert continuous values to labels.
        scaler: MinMaxScaler = self.data_processor.scaler
        ranges: tuple[int, ...] = config.box_office_ranges

        def value_to_label_fn(value: float) -> int:
            unscaled_value: float = scaler.inverse_transform([[value]])[0][0]
            return BoxOfficeRegressionDataProcessor.get_range_index(value=unscaled_value, ranges=ranges)

        # Create and configure the metrics calculator.
        metrics_calculator = PointwiseClassificationMetricsCalculator(
            value_to_label_fn=value_to_label_fn,
            f1_average_method=config.f1_average_method
        )

        # Inject the calculator into the F1ScoreHistory callback.
        f1_history_callback = F1ScoreHistory(
            validation_data=validation_data,
            metrics_calculator=metrics_calculator
        )
        return f1_history_callback

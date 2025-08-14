from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from typing_extensions import override

from src.core.project_config import ProjectPaths, ProjectModelType
from src.models.base.base_pipeline import BaseTrainingPipeline
from src.models.base.keras_setup import keras_base
from src.models.prediction.components.data_processor import (
    PredictionDataProcessor, PredictionDataSource, PredictionDataConfig, PredictionTrainingProcessedData
)
from src.models.prediction.components.model_core import (
    PredictionBuildConfig, PredictionModelCore, PredictionTrainConfig
)

History = keras_base.callbacks.History
ModelCheckpoint = keras_base.callbacks.ModelCheckpoint

@dataclass(frozen=True)
class PredictionPipelineConfig:
    """
    Represents the master configuration for a prediction model training run.

    This object is typically loaded from an external YAML file and contains all
    necessary parameters to orchestrate the entire training pipeline.

    :ivar model_id: The unique identifier for this model series.
    :ivar dataset_name: The name of the source structured dataset for training data.
    :ivar training_week_len: The number of past weeks to use as input for prediction.
    :ivar split_ratios: A tuple representing the train, validation, and test split ratios.
    :ivar lstm_units: The number of units in the LSTM layer.
    :ivar dropout_rate: The dropout rate to apply after the LSTM layer.
    :ivar epochs: The number of epochs for training.
    :ivar batch_size: The batch size for training.
    :ivar random_state: The seed for the random number generator. If None, a random seed will be generated.
    :ivar verbose: The verbosity mode for Keras training output.
    :ivar checkpoint_interval: The interval in epochs at which to save model checkpoints.
                               If None, only the final model is saved.
    """
    model_id: str
    dataset_name: str
    training_week_len: int
    split_ratios: tuple[int, int, int]
    lstm_units: int
    dropout_rate: float
    epochs: int
    batch_size: int
    random_state: Optional[int] = None
    verbose: int | str = 1
    checkpoint_interval: int | None = None


class PredictionTrainingPipeline(
    BaseTrainingPipeline[PredictionDataProcessor, PredictionModelCore, PredictionPipelineConfig]
):
    """
    Orchestrates the end-to-end training process for the box office prediction model.

    This pipeline coordinates the DataProcessor and ModelCore to execute a
    full training run based on a master configuration file. It handles data
    loading, processing, model building, training, and artifact saving.
    """

    HISTORY_FILE_NAME: str = "training_history.pkl"

    @override
    def run(self, config: PredictionPipelineConfig, continue_from_epoch: Optional[int] = None) -> None:
        """
        Executes the prediction model training pipeline from a configuration file.

        :param config: The master configuration object for this run.
        :param continue_from_epoch: If provided, loads the model from this epoch and continues training.
        :raises FileNotFoundError: If the configuration or required artifacts are not found.
        :raises ValueError: If the configuration file is empty or invalid.
        """
        self.logger.info(f"--- Starting PREDICTION training pipeline for model: {config.model_id} ---")
        self.logger.info(f"Pipeline configured with: {config}")

        master_config = config

        artifacts_folder: Path = ProjectPaths.get_model_root_path(
            model_id=master_config.model_id, model_type=ProjectModelType.PREDICTION
        )
        artifacts_folder.mkdir(parents=True, exist_ok=True)

        # 2. Model Building or Loading
        if continue_from_epoch:
            self.logger.info(f"Step 2 (Continue): Setting up for continued training...")
            self.model_core = self._setup_for_continuation(
                artifacts_folder=artifacts_folder,
                model_id=master_config.model_id,
                continue_from_epoch=continue_from_epoch
            )
        else:
            self.logger.info("Step 2 (New): This is a new training run.")



        # 3. Data Loading and Processing
        self.logger.info("Step 3: Loading and processing data...")
        data_source = PredictionDataSource(dataset_name=master_config.dataset_name)
        raw_data = self.data_processor.load_raw_data(source=data_source)

        processing_config = PredictionDataConfig(
            training_week_len=master_config.training_week_len,
            split_ratios=master_config.split_ratios,
            random_state=master_config.random_state
        )
        processed_data: PredictionTrainingProcessedData = self.data_processor.process_for_training(
            raw_data=raw_data, config=processing_config
        )
        self.logger.info("Data processing complete.")

        # 4. Build Model (if new run)
        if not continue_from_epoch:
            num_features = processed_data['x_train'].shape[2]
            build_config = PredictionBuildConfig(
                input_shape=(master_config.training_week_len, num_features),
                lstm_units=master_config.lstm_units,
                dropout_rate=master_config.dropout_rate
            )
            self.model_core.build(config=build_config)
            self.logger.info("Model building complete.")

        # 5. Model Training
        self.logger.info("Starting model training...")
        callbacks_to_use = []
        if master_config.checkpoint_interval:
            checkpoint_filepath = artifacts_folder / f"{master_config.model_id}_{{epoch:04d}}.keras"
            model_checkpoint_callback = ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=False,
                save_freq='epoch',
                period=master_config.checkpoint_interval,
                verbose=1
            )
            callbacks_to_use.append(model_checkpoint_callback)
            self.logger.info(
                f"Model checkpointing enabled. Saving every {master_config.checkpoint_interval} epochs."
            )

        train_config = PredictionTrainConfig(
            epochs=master_config.epochs,
            batch_size=master_config.batch_size,
            validation_data=(processed_data['x_val'], processed_data['y_val']),
            verbose=master_config.verbose,
            callbacks=callbacks_to_use,
            initial_epoch=continue_from_epoch or 0
        )
        history: History = self.model_core.train(
            x_train=processed_data['x_train'],
            y_train=processed_data['y_train'],
            config=train_config
        )
        self.logger.info("Model training complete.")

        self._save_run_artifacts(
            config=master_config,
            history=history,
            artifacts_folder=artifacts_folder,
            continue_from_epoch=continue_from_epoch
        )
        self.logger.info("--- PREDICTION training pipeline finished successfully. ---")

    @override
    def _check_required_artifacts_for_continuation(self) -> None:
        """
        Checks if the scaler is loaded for the prediction model.
        """
        if not self.data_processor.scaler:
            raise FileNotFoundError(
                f"Could not load scaler for continued training from {self.data_processor.model_artifacts_path}."
            )

    @override
    def _create_model_core(self, model_path: Path) -> PredictionModelCore:
        """
        Creates a PredictionModelCore instance from a saved model file.
        """
        return PredictionModelCore(model_path=model_path)

    @override
    def _get_history_filename(self) -> str:
        """
        Returns the history filename for the prediction model.
        """
        return self.HISTORY_FILE_NAME



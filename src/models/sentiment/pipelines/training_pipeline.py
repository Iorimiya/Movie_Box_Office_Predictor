from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from typing_extensions import override

from src.core.project_config import ProjectPaths, ProjectModelType
from src.models.base.base_pipeline import BaseTrainingPipeline
from src.models.base.callbacks import F1ScoreHistory
from src.models.base.keras_setup import keras_base
from src.models.sentiment.components.data_processor import (
    SentimentDataProcessor, SentimentDataSource, SentimentTrainingProcessedData, SentimentDataConfig
)
from src.models.sentiment.components.model_core import (
    SentimentBuildConfig, SentimentModelCore, SentimentTrainConfig
)

History = keras_base.callbacks.History
ModelCheckpoint = keras_base.callbacks.ModelCheckpoint

@dataclass(frozen=True)
class SentimentPipelineConfig:
    """
    Represents the master configuration for a sentiment model training run.

    This object is typically loaded from an external YAML file and contains all
    necessary parameters to orchestrate the entire training pipeline.

    :ivar model_id: The unique identifier for this model series.
    :ivar dataset_file_name: The name of the source CSV file for training data.
    :ivar vocabulary_size: The maximum number of words for the tokenizer.
    :ivar embedding_dim: The dimensionality of the word embedding vectors.
    :ivar lstm_units: The number of units in the LSTM layer.
    :ivar split_ratios: A tuple representing the train, validation, and test split ratios.
    :ivar random_state: The seed for reproducible data splitting.
    :ivar epochs: The number of epochs for training.
    :ivar batch_size: The batch size for training.
    :ivar verbose: The verbosity mode for Keras training output.
    :ivar checkpoint_interval: The interval in epochs at which to save model checkpoints.
                               If None, only the final model is saved.
    """
    model_id: str
    dataset_file_name: str
    vocabulary_size: int
    embedding_dim: int
    lstm_units: int
    split_ratios: tuple[int, int, int]
    random_state: int
    epochs: int
    batch_size: int
    verbose: int | str = 1
    checkpoint_interval: int | None = None


class SentimentTrainingPipeline(
    BaseTrainingPipeline[SentimentDataProcessor, SentimentModelCore, SentimentPipelineConfig]
):
    """
    Orchestrates the end-to-end training process for the sentiment analysis model.

    This pipeline coordinates the DataProcessor and ModelCore to execute a
    full training run based on a master configuration file. It handles data
    loading, processing, model building, training, and artifact saving.
    """

    HISTORY_FILE_NAME: str = "training_history.pkl"

    @override
    def run(self, config: SentimentPipelineConfig, continue_from_epoch: Optional[int] = None) -> None:
        """
        Executes the sentiment model training pipeline from a configuration file.

        This method orchestrates the entire process, including handling both
        new training runs and continuing from a saved checkpoint.

        :param config: The master configuration object for this run.
        :param continue_from_epoch: If provided, loads the model from this epoch and continues training.
                                     If None, starts a new training run.
        :raises FileNotFoundError: If the configuration or required artifacts are not found.
        :raises ValueError: If the configuration file is empty or invalid.
        """
        self.logger.info(f"--- Starting SENTIMENT training pipeline for model: {config.model_id} ---")
        self.logger.info(f"Pipeline configured with: {config}")
        master_config = config

        artifacts_folder: Path = ProjectPaths.get_model_root_path(
            model_id=master_config.model_id, model_type=ProjectModelType.SENTIMENT
        )
        artifacts_folder.mkdir(parents=True, exist_ok=True)

        # 2. Model Building or Loading
        if continue_from_epoch:
            self.logger.info(f"Step 2 (Continue): Loading existing model from epoch {continue_from_epoch}...")
            self.model_core = self._setup_for_continuation(
                artifacts_folder=artifacts_folder,
                model_id=master_config.model_id,
                continue_from_epoch=continue_from_epoch
            )
        else:
            # This is a new training run.
            self.logger.info("Step 2 (New): Building new model architecture...")

        # 3. Data Loading and Processing
        self.logger.info("Step 3: Loading and processing data...")
        data_source: SentimentDataSource = SentimentDataSource(file_name=master_config.dataset_file_name)
        raw_data = self.data_processor.load_raw_data(source=data_source)

        processing_config: SentimentDataConfig = SentimentDataConfig(
            vocabulary_size=master_config.vocabulary_size,
            split_ratios=master_config.split_ratios,
            random_state=master_config.random_state
        )
        # Note: In "continue training" mode, this re-processes the data, which is necessary
        # to get the validation set for Keras. The tokenizer is loaded, not re-fitted.
        processed_data: SentimentTrainingProcessedData = self.data_processor.process_for_training(
            raw_data=raw_data,
            config=processing_config
        )
        self.logger.info("Data processing complete.")

        # If it was a new run, we now build the model with the sequence length from the processor
        if not continue_from_epoch:
            if self.data_processor.max_sequence_length is None:
                raise ValueError("max_sequence_length was not set by the data processor.")

            build_config: SentimentBuildConfig = SentimentBuildConfig(
                vocabulary_size=master_config.vocabulary_size,
                embedding_dim=master_config.embedding_dim,
                lstm_units=master_config.lstm_units,
                max_sequence_length=self.data_processor.max_sequence_length
            )
            self.model_core.build(config=build_config)
            self.logger.info("Model building complete.")

        # 4. Model Training
        self.logger.info("Step 4: Starting model training...")
        callbacks_to_use = []
        f1_callback = F1ScoreHistory(validation_data=(processed_data['x_val'], processed_data['y_val']))
        callbacks_to_use.append(f1_callback)

        if master_config.checkpoint_interval:
            num_train_samples: int = len(processed_data['x_train'])
            save_frequency_in_batches: int | str

            if num_train_samples == 0:
                self.logger.warning("Training data is empty. Checkpoint callback will fall back to saving every epoch.")
                save_frequency_in_batches = 'epoch'
            else:
                # Calculate steps per epoch using ceiling division
                steps_per_epoch: int = (num_train_samples + master_config.batch_size - 1) // master_config.batch_size
                save_frequency_in_batches = master_config.checkpoint_interval * steps_per_epoch
                self.logger.info(
                    f"Checkpoint interval of {master_config.checkpoint_interval} epochs "
                    f"translates to a save frequency of {save_frequency_in_batches} batches "
                    f"(steps_per_epoch: {steps_per_epoch})."
                )
            checkpoint_filepath = artifacts_folder / f"{master_config.model_id}_{{epoch:04d}}.keras"
            model_checkpoint_callback = ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=False,
                save_freq=save_frequency_in_batches,
                verbose=1
            )
            callbacks_to_use.append(model_checkpoint_callback)
            self.logger.info(
                f"Model checkpointing enabled. Saving every {master_config.checkpoint_interval} epochs."
            )

        train_config: SentimentTrainConfig = SentimentTrainConfig(
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

        # 5. Saving Artifacts
        self._save_run_artifacts(
            config=master_config,
            history=history,
            artifacts_folder=artifacts_folder,
            continue_from_epoch=continue_from_epoch,
            f1_callback=f1_callback
        )

        self.logger.info("--- SENTIMENT training pipeline finished successfully. ---")

    @override
    def _check_required_artifacts_for_continuation(self) -> None:
        """
        Checks if the tokenizer is loaded for the sentiment model.
        """
        if not self.data_processor.tokenizer:
            raise FileNotFoundError(
                f"Could not load tokenizer for continued training from {self.data_processor.model_artifacts_path}."
            )

    @override
    def _create_model_core(self, model_path: Path) -> SentimentModelCore:
        """
        Creates a SentimentModelCore instance from a saved model file.
        """
        return SentimentModelCore(model_path=model_path)

    @override
    def _get_history_filename(self) -> str:
        """
        Returns the history filename for the prediction model.
        """
        return self.HISTORY_FILE_NAME

    @override
    def _merge_histories(
        self,
        new_history: History,
        history_save_path: Path,
        continue_from_epoch: Optional[int],
        f1_callback: Optional[F1ScoreHistory] = None
    ) -> History:
        """
        Extends the base history merging to also handle the 'val_f1_score' metric.

        :param new_history: The History object from the latest training run.
        :param history_save_path: The path to the history file.
        :param continue_from_epoch: The epoch number the training continued from.
        :param f1_callback: The F1ScoreHistory callback instance used during training.
        :returns: The final, potentially merged, History object to be saved.
        """
        merged_history: History = super()._merge_histories(
            new_history=new_history,
            history_save_path=history_save_path,
            continue_from_epoch=continue_from_epoch
        )

        if f1_callback:
            if continue_from_epoch and history_save_path.exists():
                merged_history.history.setdefault('val_f1_score', []).extend(f1_callback.f1_scores)
            else:
                merged_history.history['val_f1_score'] = f1_callback.f1_scores

        return merged_history

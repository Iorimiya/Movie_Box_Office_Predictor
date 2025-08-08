from dataclasses import dataclass
from pathlib import Path
from typing import override, Optional

from keras.src.callbacks import History, ModelCheckpoint

from src.core.project_config import ProjectPaths, ProjectModelType
from src.data_handling.file_io import YamlFile, PickleFile
from src.models.base.base_pipeline import BaseTrainingPipeline
from src.models.base.callbacks import F1ScoreHistory
from src.models.sentiment.components.data_processor import (
    ProcessedSentimentData, SentimentDataProcessor, SentimentDataSource, SentimentDataConfig
)
from src.models.sentiment.components.evaluator import SentimentEvaluator
from src.models.sentiment.components.model_core import (
    SentimentBuildConfig, SentimentModelCore, SentimentTrainConfig
)


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
    BaseTrainingPipeline[SentimentDataProcessor, SentimentModelCore, Path]
):
    """
    Orchestrates the end-to-end training process for the sentiment analysis model.

    This pipeline coordinates the DataProcessor and ModelCore to execute a
    full training run based on a master configuration file. It handles data
    loading, processing, model building, training, and artifact saving.
    """

    @override
    def run(self, config_path: Path, continue_from_epoch: Optional[int] = None) -> None:
        """
        Executes the sentiment model training pipeline from a configuration file.

        This method orchestrates the entire process, including handling both
        new training runs and continuing from a saved checkpoint.

        :param config_path: The path to the master YAML configuration file for this run.
        :param continue_from_epoch: If provided, loads the model from this epoch and continues training.
                                     If None, starts a new training run.
        :raises FileNotFoundError: If the configuration or required artifacts are not found.
        :raises ValueError: If the configuration file is empty or invalid.
        """
        self.logger.info(f"--- Starting SENTIMENT training pipeline from config: {config_path} ---")

        # 1. Load Master Configuration
        self.logger.info("Step 1: Loading master configuration file...")
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        yaml_loader: YamlFile = YamlFile(path=config_path)
        raw_config: dict[str, any] | None = yaml_loader.load_single_document()
        if not raw_config:
            raise ValueError(f"Configuration file is empty or invalid: {config_path}")

        master_config: SentimentPipelineConfig = SentimentPipelineConfig(**raw_config)
        self.logger.info(f"Pipeline configured with: {master_config}")

        artifacts_folder: Path = ProjectPaths.get_model_root_path(
            model_id=master_config.model_id, model_type=ProjectModelType.SENTIMENT
        )
        artifacts_folder.mkdir(parents=True, exist_ok=True)

        # 2. Model Building or Loading
        if continue_from_epoch:
            self.logger.info(f"Step 2 (Continue): Loading existing model from epoch {continue_from_epoch}...")
            # In continue mode, the DataProcessor must load its existing tokenizer.
            # The Handler already initialized it with the correct path.
            self.data_processor.load_artifacts()
            if not self.data_processor.tokenizer:
                raise FileNotFoundError(f"Could not load tokenizer for continued training from {artifacts_folder}.")

            model_to_load_path = artifacts_folder / f"{master_config.model_id}_{continue_from_epoch:04d}.keras"
            if not model_to_load_path.exists():
                raise FileNotFoundError(f"Checkpoint to continue from not found: {model_to_load_path}")

            # Re-initialize ModelCore with the loaded model
            self.model_core = SentimentModelCore(model_path=model_to_load_path)
            self.logger.info(f"Successfully loaded model from: {model_to_load_path}")
        else:
            # This is a new training run.
            self.logger.info("Step 2 (New): Building new model architecture...")
            # The DataProcessor will fit a new tokenizer in the next step.
            pass

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
        processed_data: ProcessedSentimentData = self.data_processor.process_for_training(
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
            checkpoint_filepath = artifacts_folder / f"{master_config.model_id}_{{epoch:04d}}.keras"
            model_checkpoint_callback = ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=False,
                # Keras's save_freq expects 'epoch' or an integer number of batches.
                # We interpret checkpoint_interval as number of epochs.
                save_freq='epoch',
                period=master_config.checkpoint_interval,
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
        self.logger.info("Step 5: Saving all artifacts...")
        history_save_path: Path = artifacts_folder / SentimentEvaluator.HISTORY_FILE_NAME

        # Merge histories if we are continuing a run
        if continue_from_epoch and history_save_path.exists():
            self.logger.info(f"Loading existing history from {history_save_path} to append new results.")
            old_history: History = PickleFile(path=history_save_path).load()
            for key, value in history.history.items():
                old_history.history.setdefault(key, []).extend(value)
            # Add the new F1 scores to the merged history
            old_history.history.setdefault('val_f1_score', []).extend(f1_callback.f1_scores)
            history_to_save = old_history
        else:
            # This is a new run, or the old history file is missing
            history.history['val_f1_score'] = f1_callback.f1_scores
            history_to_save = history

        # Save the (potentially merged) history
        PickleFile(path=history_save_path).save(data=history_to_save)
        self.logger.info(f"Training history saved to: {history_save_path}")

        # Save Data Processor Artifacts (Tokenizer) only on a new run
        if not continue_from_epoch:
            self.data_processor.save_artifacts()
            self.logger.info(f"Data processor artifacts saved in: {self.data_processor.model_artifacts_path}")

        # Always save the final model state if it wasn't already saved by a checkpoint
        final_model_save_path: Path = artifacts_folder / f"{master_config.model_id}_{master_config.epochs:04d}.keras"
        if not final_model_save_path.exists():
            self.model_core.save(file_path=final_model_save_path)
            self.logger.info(f"Final model state for epoch {master_config.epochs} saved to: {final_model_save_path}")

        self.logger.info("--- SENTIMENT training pipeline finished successfully. ---")

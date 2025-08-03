from pathlib import Path
from typing import override, Optional

from src.models.base.base_pipeline import BaseTrainingPipeline
from src.models.sentiment.components.data_processor import SentimentDataProcessor
from src.models.sentiment.components.model_core import SentimentModelCore


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

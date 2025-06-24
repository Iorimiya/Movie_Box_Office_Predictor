from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from logging import Logger
from pathlib import Path

from src.core.logging_manager import LoggingManager
from src.core.project_config import ProjectDatasetType, ProjectModelType, ProjectPaths
from src.data_handling.dataset import Dataset


class DatasetHandler:
    """
    Handles Command-Line Interface (CLI) commands related to dataset management.

    :ivar _logger: The logger instance for this class.
    :ivar _parser: The argument parser instance for handling CLI arguments.
    """
    _logger: Logger = LoggingManager().get_logger()
    _parser: ArgumentParser

    def __init__(self, parser: ArgumentParser) -> None:
        """
        Initializes the DatasetHandler.

        :param parser: The argument parser instance.
        """
        self._parser = parser

    def create_index(self, args: Namespace) -> None:
        """
        Creates an index file for a structured dataset from a source CSV file.

        :param args: The namespace object containing command-line arguments,
                     expected to have 'structured_dataset_name' and 'source_file'.
        :raises FileNotFoundError: If the specified source file does not exist.
        """
        self._logger.info(
            f"Executing: Create dataset index for '{args.structured_dataset_name}'"
        )
        source_path: Path = Path(args.source_file)
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found at: {args.source_file}")

        Dataset(name=args.structured_dataset_name).initialize_index_file(
            source_csv=args.source_file
        )

    @staticmethod
    def _validate_dataset_path(dataset_name: str) -> Path:
        """
        Validates the existence of a structured dataset path.

        :param dataset_name: The name of the structured dataset.
        :return: The validated path to the dataset.
        :raises FileNotFoundError: If the dataset path does not exist.
        """
        dataset_path: Path = ProjectPaths.get_dataset_path(
            dataset_name=dataset_name,
            dataset_type=ProjectDatasetType.STRUCTURED
        )
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset '{dataset_name}' not found at expected path: {dataset_path}")
        return dataset_path

    def collect_box_office(self, args: Namespace) -> None:
        """
        Collects box office data for an entire dataset or a single movie.

        :param args: The namespace object containing command-line arguments,
                     expected to have 'structured_dataset_name' or 'movie_name'.
        :raises FileNotFoundError: If the specified dataset path does not exist.
        """
        if args.structured_dataset_name:
            dataset_name: str = args.structured_dataset_name
            self._logger.info(f"Executing: Collect box office data for dataset '{dataset_name}'.")
            self._validate_dataset_path(dataset_name=dataset_name)
            Dataset(name=dataset_name).collect_box_office()
        elif args.movie_name:
            self._logger.info(f"Executing: Collect box office data for movie '{args.movie_name}'.")
            # TODO: Implement single movie box office collection logic
            pass

    def collect_ptt_review(self, args: Namespace) -> None:
        """
        Collects PTT reviews for an entire dataset or a single movie.

        :param args: The namespace object containing command-line arguments,
                     expected to have 'structured_dataset_name' or 'movie_name'.
        :raises FileNotFoundError: If the specified dataset path does not exist.
        """
        if args.structured_dataset_name:
            dataset_name: str = args.structured_dataset_name
            self._logger.info(f"Executing: Collect PTT reviews for dataset '{dataset_name}'.")
            self._validate_dataset_path(dataset_name=dataset_name)
            Dataset(name=dataset_name).collect_public_review(target_website='PTT')
        elif args.movie_name:
            self._logger.info(f"Executing: Collect PTT reviews for movie '{args.movie_name}'.")
            # TODO: Implement single movie PTT review collection logic
            pass

    def collect_dcard_review(self, args: Namespace) -> None:
        """
        Collects Dcard reviews for an entire dataset or a single movie.

        :param args: The namespace object containing command-line arguments,
                     expected to have 'structured_dataset_name' or 'movie_name'.
        :raises FileNotFoundError: If the specified dataset path does not exist.
        """
        if args.structured_dataset_name:
            dataset_name: str = args.structured_dataset_name
            self._logger.info(f"Executing: Collect Dcard reviews for dataset '{dataset_name}'.")
            self._validate_dataset_path(dataset_name=dataset_name)
            Dataset(name=dataset_name).collect_public_review(target_website='DCARD')
        elif args.movie_name:
            self._logger.info(f"Executing: Collect Dcard reviews for movie '{args.movie_name}'.")
            # TODO: Implement single movie Dcard review collection logic
            pass

    def compute_sentiment(self, args: Namespace) -> None:
        """
        Computes sentiment scores for a structured dataset using a specified model.

        :param args: The namespace object containing command-line arguments,
                     expected to have 'structured_dataset_name', 'model_id', and 'epoch'.
        :raises FileNotFoundError: If the dataset or model path does not exist.
        """
        self._logger.info(
            f"Executing: Compute sentiment for dataset '{args.structured_dataset_name}' "
            f"using model '{args.model_id}' (epoch: {args.epoch})."
        )
        dataset_name: str = args.structured_dataset_name
        self._validate_dataset_path(dataset_name=dataset_name)

        model_path: Path = ProjectPaths.get_model_root_path(
            model_id=args.model_id,
            model_type=ProjectModelType.SENTIMENT
        )
        if not model_path.exists():
            raise FileNotFoundError(f"Model '{args.model_id}' not found at expected path: {model_path}")

        Dataset(name=dataset_name).compute_sentiment(
            model_id=args.model_id,
            model_epoch=args.epoch
        )


class BaseModelHandler(ABC):
    """
    An abstract base class for model handlers to reduce code duplication.

    :ivar _logger: The shared logger instance for all model handlers.
    :ivar _parser: The argument parser instance for the specific command.
    :ivar _model_type_name: The display name of the model type (e.g., "Sentiment").
    """
    _logger: Logger = LoggingManager().get_logger()
    _parser: ArgumentParser
    _model_type_name: str

    def __init__(self, parser: ArgumentParser, model_type_name: str) -> None:
        """
        Initializes the BaseModelHandler.

        :param parser: The argument parser instance.
        :param model_type_name: The name of the model type for logging.
        """
        self._parser = parser
        self._model_type_name = model_type_name

    def train(self, args: Namespace) -> None:
        """
        Trains a model based on the provided arguments.

        This method handles the logic for initiating a model training process,
        logging the configuration, and selecting the data source.

        :param args: The namespace object containing command-line arguments
                     for model training, such as data source, model ID, and epochs.
        """
        source_type: str = ""
        if args.feature_dataset_name:
            source_type = f"feature dataset: {args.feature_dataset_name}"
        elif args.structured_dataset_name:
            source_type = f"structured dataset: {args.structured_dataset_name}"
        elif args.random_data:
            source_type = "randomly generated data"

        self._logger.info(f"Executing: Train {self._model_type_name} model '{args.model_id}'. Source: {source_type}.")
        self._logger.info(
            f"  Old Epoch: {args.old_epoch if args.old_epoch else 'None'}, "
            f"Target Epoch: {args.target_epoch}, "
            f"Checkpoint Interval: {args.checkpoint_interval if args.checkpoint_interval else 'None'}"
        )
        # TODO: Add actual training logic here

    @abstractmethod
    def test(self, args: Namespace) -> None:
        """
        An abstract method for testing a model. Must be implemented by subclasses.

        :param args: The namespace object containing command-line arguments.
        """
        pass

    def plot_graph(self, args: Namespace) -> None:
        """
        Plots evaluation graphs (e.g., loss, F1-score) for a model.

        :param args: The namespace object containing command-line arguments,
                     including 'model_id' and flags for which graphs to plot.
        """
        if not (args.training_loss or args.validation_loss or args.f1_score):
            self._parser.error(
                "At least one flag from --training-loss, --validation-loss, or --f1-score must be selected for plotting."
            )

        self._logger.info(f"Executing: Plot evaluation graphs for {self._model_type_name} model '{args.model_id}'.")
        if args.training_loss:
            self._logger.info("  - Plotting training loss curve.")
        if args.validation_loss:
            self._logger.info("  - Plotting validation loss curve.")
        if args.f1_score:
            self._logger.info("  - Plotting F1-score curve.")
        # TODO: Add actual plotting logic here

    def get_metrics(self, args: Namespace) -> None:
        """
        Retrieves and displays evaluation metrics for a model at a specific epoch.

        :param args: The namespace object containing command-line arguments,
                     including 'model_id', 'epoch', and flags for which metrics to retrieve.
        """
        if not (args.training_loss or args.validation_loss or args.f1_score):
            self._parser.error(
                "At least one flag from --training-loss, --validation-loss, or --f1-score must be selected to get metrics."
            )

        self._logger.info(
            f"Executing: Get evaluation metrics for {self._model_type_name} model "
            f"'{args.model_id}' (epoch: {args.epoch})."
        )
        if args.training_loss:
            self._logger.info("  - Retrieving training loss.")
        if args.validation_loss:
            self._logger.info("  - Retrieving validation loss.")
        if args.f1_score:
            self._logger.info("  - Retrieving F1-score.")
        # TODO: Add actual metric retrieval logic here


class SentimentModelHandler(BaseModelHandler):
    """
    Handles CLI commands related to sentiment analysis model management.
    """

    def __init__(self, parser: ArgumentParser) -> None:
        """
        Initializes the SentimentModelHandler.

        :param parser: The argument parser instance.
        """
        super().__init__(parser=parser, model_type_name="Sentiment")

    def test(self, args: Namespace) -> None:
        """
        Tests a sentiment analysis model with a given input sentence.

        :param args: The namespace object containing command-line arguments,
                     expected to have 'model_id', 'epoch', and 'input_sentence'.
        """
        self._logger.info(
            f"Executing: Test {self._model_type_name} model '{args.model_id}' (epoch: {args.epoch}) "
            f"with input: '{args.input_sentence}'"
        )
        # TODO: Add actual testing logic here


class PredictionModelHandler(BaseModelHandler):
    """
    Handles CLI commands related to the box office prediction model.
    """

    def __init__(self, parser: ArgumentParser) -> None:
        """
        Initializes the PredictionModelHandler.

        :param parser: The argument parser instance.
        """
        super().__init__(parser=parser, model_type_name="Prediction")

    def test(self, args: Namespace) -> None:
        """
        Tests the prediction model on a specific movie or with random data.

        :param args: The namespace object containing command-line arguments,
                     expected to have 'model_id', 'epoch', and either 'movie_name' or 'random'.
        """
        if args.movie_name:
            self._logger.info(
                f"Executing: Test {self._model_type_name} model '{args.model_id}' (epoch: {args.epoch}) "
                f"on movie: '{args.movie_name}'."
            )
        elif args.random:
            self._logger.info(
                f"Executing: Test {self._model_type_name} model '{args.model_id}' (epoch: {args.epoch}) "
                f"with random data."
            )
        # TODO: Add actual testing logic here

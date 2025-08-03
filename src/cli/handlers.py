from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Final

from numpy.typing import NDArray
from src.utilities.plot import plot_multi_line_graph, PlotDataset
from typing_extensions import override

from src.core.logging_manager import LoggingManager
from src.core.project_config import ProjectDatasetType, ProjectModelType, ProjectPaths
from src.data_collection.box_office_collector import BoxOfficeCollector
from src.data_collection.review_collector import ReviewCollector, TargetWebsite
from src.data_handling.box_office import BoxOffice
from src.data_handling.dataset import Dataset
from src.data_handling.file_io import YamlFile
from src.data_handling.reviews import PublicReview
from src.models.sentiment.components.data_processor import (
    SentimentDataProcessor, SentimentTrainingConfig
)
from src.models.sentiment.components.evaluator import (
    SentimentEvaluator, SentimentEvaluationResult, SentimentEvaluationConfig
)
from src.models.sentiment.components.model_core import SentimentModelCore, SentimentPredictConfig
from src.models.sentiment.pipelines.training_pipeline import SentimentTrainingPipeline


@dataclass(frozen=True)
class SentimentMultiEpochEvaluationResult:
    """
    Holds the aggregated results from evaluating multiple epochs of a single model series.

    This class is used to gather all necessary data for plotting comparative graphs.

    :ivar model_id: The unique identifier for the model series.
    :ivar evaluated_epochs: A list of the epoch numbers that were actually evaluated.
    :ivar test_f1_scores: A list of F1-scores from the test set, corresponding to each evaluated epoch.
    :ivar test_losses: A list of loss values from the test set, corresponding to each evaluated epoch.
    :ivar full_training_loss_history: The complete training loss history from the original run.
    :ivar full_validation_loss_history: The complete validation loss history from the original run.
    """
    model_id: str
    evaluated_epochs: list[int]
    test_f1_scores: list[float]
    test_losses: list[float]
    full_training_loss_history: list[float]
    full_validation_loss_history: list[float]



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
        :returns: The validated path to the dataset.
        :raises FileNotFoundError: If the dataset path does not exist.
        """
        dataset_path: Path = ProjectPaths.get_dataset_path(
            dataset_name=dataset_name,
            dataset_type=ProjectDatasetType.STRUCTURED
        )
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset '{dataset_name}' not found at expected path: {dataset_path}")
        return dataset_path

    @staticmethod
    def _log_box_office_data(logger: Logger, movie_name: str, box_office_data: list[BoxOffice]) -> None:
        """
        Logs formatted box office data for a movie.

        :param logger: The logger instance to use for logging.
        :param movie_name: The name of the movie.
        :param box_office_data: A list of BoxOffice objects.
        """
        logger.info(f"--- Box Office Data for '{movie_name}' ---")
        for entry in box_office_data:
            logger.info(str(entry))
        logger.info(f"---------------------------------------")

    @staticmethod
    def _log_reviews_data(logger: Logger, movie_name: str, reviews: list[PublicReview], review_type: str) -> None:
        """
        Logs formatted review data for a movie.

        :param logger: The logger instance to use for logging.
        :param movie_name: The name of the movie.
        :param reviews: A list of PublicReview objects.
        :param review_type: The type of review (e.g., "PTT", "Dcard").
        """
        logger.info(f"--- {review_type} Reviews for '{movie_name}' ---")
        for i, review in enumerate(reviews):
            logger.info(f"Review {i + 1}:")
            logger.info(str(review))
            if i < len(reviews) - 1:
                logger.info("-" * 40)
        logger.info(f"---------------------------------------")

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
            movie_name: str = args.movie_name
            self._logger.info(f"Executing: Collect box office data for movie '{movie_name}'.")

            try:
                with BoxOfficeCollector(download_mode='WEEK') as collector:
                    box_office_data: list[BoxOffice] = collector.download_box_office_data_for_movie(
                        movie_name=movie_name
                    )

                    if box_office_data:
                        self._logger.info(f"Successfully retrieved box office data for '{movie_name}'.")
                        self._log_box_office_data(
                            logger=self._logger, movie_name=movie_name, box_office_data=box_office_data
                        )
                    else:
                        self._logger.warning(f"No box office data found for '{movie_name}'.")
            except RuntimeError as e:
                self._logger.error(f"Failed to retrieve box office data for '{movie_name}': {e}")
            except Exception as e:
                self._logger.error(
                    f"An unexpected error occurred while collecting box office data for '{movie_name}': {e}",
                    exc_info=True
                )

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
            movie_name: str = args.movie_name
            self._logger.info(f"Executing: Collect PTT reviews for movie '{movie_name}'.")

            try:
                with ReviewCollector(target_website=TargetWebsite.PTT) as collector:
                    reviews: list[PublicReview] = collector.collect_reviews_for_movie(movie_name=movie_name)
                    if reviews:
                        self._logger.info(f"Successfully retrieved PTT reviews for '{movie_name}'.")
                        self._log_reviews_data(logger=self._logger, movie_name=movie_name, reviews=reviews,
                                               review_type="PTT")
                    else:
                        self._logger.warning(f"No PTT reviews found for '{movie_name}'.")

            except Exception as e:
                self._logger.error(
                    f"An error occurred while collecting PTT reviews for '{movie_name}': {e}",
                    exc_info=True
                )
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
            movie_name: str = args.movie_name
            self._logger.info(f"Executing: Collect Dcard reviews for movie '{movie_name}'.")

            try:
                with ReviewCollector(target_website=TargetWebsite.DCARD) as collector:
                    reviews: list[PublicReview] = collector.collect_reviews_for_movie(movie_name=movie_name)
                    if reviews:
                        self._logger.info(f"Successfully retrieved Dcard reviews for '{movie_name}'.")
                        self._log_reviews_data(
                            logger=self._logger, movie_name=movie_name, reviews=reviews, review_type="Dcard"
                        )
                    else:
                        self._logger.warning(f"No Dcard reviews found for '{movie_name}'.")
            except Exception as e:
                self._logger.error(
                    f"An error occurred while collecting Dcard reviews for '{movie_name}': {e}",
                    exc_info=True
                )
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

    @staticmethod
    def _find_available_epochs(model_id: str, model_type: ProjectModelType) -> list[int]:
        """
        Finds all available model checkpoint epochs for a given model ID and type.

        This is a static utility method that scans the appropriate model artifact
        directory for files matching the pattern '<model_id>_*.keras' and extracts
        the epoch numbers.

        :param model_id: The unique identifier for the model series.
        :param model_type: The type of the model (e.g., SENTIMENT, PREDICTION).
        :returns: A sorted list of available epoch numbers.
        """
        model_artifacts_path: Path = ProjectPaths.get_model_root_path(
            model_id=model_id, model_type=model_type
        )
        if not model_artifacts_path.exists():
            return []

        epochs: list[int] = []
        for f in model_artifacts_path.glob(f"{model_id}_*.keras"):
            try:
                # Extracts the number from 'model_id_0080.keras'
                epoch_str = f.stem.split('_')[-1]
                epochs.append(int(epoch_str))
            except (ValueError, IndexError):
                continue
        return sorted(epochs)

    @abstractmethod
    def train(self, args: Namespace) -> None:
        """
        An abstract method for training a model. Must be implemented by subclasses.

        :param args: The namespace object containing command-line arguments.
        """
        pass

    @abstractmethod
    def predict(self, args: Namespace) -> None:
        """
        An abstract method for testing a model. Must be implemented by subclasses.

        :param args: The namespace object containing command-line arguments.
        """
        pass

    @abstractmethod
    def get_metrics(self, args: Namespace) -> None:
        """
        An abstract method for retrieving evaluation metrics. Must be implemented by subclasses.

        :param args: The namespace object containing command-line arguments.
        """
        pass

    @abstractmethod
    def plot_graph(self, args: Namespace) -> None:
        """
        An abstract method for plotting evaluation graphs. Must be implemented by subclasses.

        :param args: The namespace object containing command-line arguments.
        """
        pass


# noinspection PyUnreachableCode
class SentimentModelHandler(BaseModelHandler):
    """
    Handles CLI commands related to sentiment analysis model management.
    """

    _EVALUATION_CACHE_FILE_NAME: Final[str] = "evaluation_cache.yaml"

    @override
    def __init__(self, parser: ArgumentParser) -> None:
        """
        Initializes the SentimentModelHandler.

        :param parser: The argument parser instance.
        """
        super().__init__(parser=parser, model_type_name="Sentiment")
        self._evaluator: SentimentEvaluator = SentimentEvaluator()

    @override
    def train(self, args: Namespace) -> None:
        """
        Orchestrates the sentiment model training process, handling both new
        and continued training runs.

        If continuing a training run (using `--continue-from-epoch`), this method
        enforces the use of the original model's configuration and disallows any
        new overrides.

        For a new training run, it implements a 'default + override' configuration
        logic, creating and saving a new master configuration file before
        launching the training pipeline.

        :param args: The namespace object from argparse, containing `model_id` and
                     other training-related parameters.
        """
        model_id: str = args.model_id
        artifacts_folder: Path = ProjectPaths.get_model_root_path(
            model_id=model_id, model_type=ProjectModelType.SENTIMENT
        )
        final_config_path: Path = artifacts_folder / "config.yaml"

        # --- Cache Invalidation ---
        # A new training run will invalidate any previous evaluation results.
        cache_path: Path = artifacts_folder / self._EVALUATION_CACHE_FILE_NAME
        if cache_path.exists():
            self._logger.info(f"Invalidating evaluation cache at '{cache_path}' due to new training run.")
            cache_path.unlink()

        # --- Main logic branch: Determine if it's a "New" or "Continue" training run ---
        if args.continue_from_epoch:
            # --- Mode: Continue Training ---
            self._logger.info(
                f"Executing: Continue training {self._model_type_name} model '{model_id}' "
                f"from epoch {args.continue_from_epoch}."
            )

            # Rule: No new overrides are allowed when continuing training
            if args.config_override:
                self._parser.error("Argument --config-override cannot be used with --continue-from-epoch.")

            individual_overrides: dict[str, any] = {
                key: value for key, value in vars(args).items()
                if key not in ['command_group', 'sentiment_subcommand', 'func', 'model_id',
                               'continue_from_epoch'] and value is not None
            }
            if individual_overrides:
                self._parser.error(
                    f"Individual overrides like --{next(iter(individual_overrides))} cannot be used with --continue-from-epoch.")

            # Rule: The original config.yaml must be found
            if not final_config_path.exists():
                self._parser.error(
                    f"Cannot continue training: Original config.yaml not found for model '{model_id}' at '{final_config_path}'."
                )

            self._logger.info(f"Using existing configuration file: {final_config_path}")

        else:
            # --- 1. Check for mutually exclusive arguments ---
            individual_overrides: dict[str, any] = {
                key: value for key, value in vars(args).items()
                if key not in ['command_group', 'sentiment_subcommand', 'func', 'model_id',
                               'config_override'] and value is not None
            }
            if args.config_override and individual_overrides:
                self._parser.error("Argument --config-override cannot be used with individual parameter overrides.")

            # --- 2. Load default configuration ---
            default_config_path: Path = ProjectPaths.project_root / "configs" / "sentiment_defaults.yaml"
            try:
                default_config: dict[str, any] = YamlFile(path=default_config_path).load_single_document()
                self._logger.info(f"Loaded default configuration from: {default_config_path}")
            except FileNotFoundError:
                self._parser.error(f"Default configuration file not found at: {default_config_path}")
                return

            # --- 3. Apply overrides ---
            effective_config: dict[str, any] = default_config.copy()
            if args.config_override:
                try:
                    self._logger.info(f"Applying overrides from file: {args.config_override}")
                    override_config: dict[str, any] = YamlFile(path=args.config_override).load_single_document()
                    effective_config.update(override_config)
                except FileNotFoundError:
                    self._parser.error(f"Override configuration file not found: {args.config_override}")
                    return
            elif individual_overrides:
                self._logger.info(f"Applying individual overrides: {individual_overrides}")
                effective_config.update(individual_overrides)

            # --- 4. Create artifact directory and save the final configuration ---
            model_id: str = args.model_id
            artifacts_folder: Path = ProjectPaths.get_model_root_path(
                model_id=model_id, model_type=ProjectModelType.SENTIMENT
            )
            artifacts_folder.mkdir(parents=True, exist_ok=True)

            effective_config['model_id'] = model_id
            final_config_path: Path = artifacts_folder / "config.yaml"
            try:
                YamlFile(path=final_config_path).save_single_document(data=effective_config)
                self._logger.info(f"Effective configuration saved to: {final_config_path}")
            except Exception as e:
                self._parser.error(f"Failed to save final configuration file: {e}")
                return

        # --- 5. Execute the pipeline ---
        try:
            data_processor: SentimentDataProcessor = SentimentDataProcessor(model_artifacts_path=artifacts_folder)
            model_core: SentimentModelCore = SentimentModelCore()
            pipeline: SentimentTrainingPipeline = SentimentTrainingPipeline(
                data_processor=data_processor, model_core=model_core
            )

            pipeline.run(
                config_path=final_config_path,
                continue_from_epoch=args.continue_from_epoch
            )
        except Exception as e:
            self._logger.error(f"An error occurred during the training pipeline execution: {e}", exc_info=True)
            self._parser.error(f"Pipeline execution failed. Check logs for details.")

    @override
    def predict(self, args: Namespace) -> None:
        """
        Tests a sentiment analysis model with a given input sentence.

        This method loads the specified model and its corresponding tokenizer,
        processes the input sentence, and prints the predicted sentiment score.
        It terminates with an error if the required model or tokenizer files
        are not found.

        :param args: The namespace object containing command-line arguments,
                     expected to have 'model_id', 'epoch', and 'input_sentence'.
        """
        self._logger.info(
            f"Executing: Test {self._model_type_name} model '{args.model_id}' (epoch: {args.epoch}) "
            f"with input: '{args.input_sentence}'"
        )

        # --- 1. Locate artifact paths ---
        model_artifacts_path: Path = ProjectPaths.get_model_root_path(
            model_id=args.model_id, model_type=ProjectModelType.SENTIMENT
        )
        model_file_path: Path = model_artifacts_path / f"{args.model_id}_{args.epoch:04d}.keras"

        if not model_file_path.exists():
            self._parser.error(f"Model file not found at: {model_file_path}")
            return

        # --- 2. Instantiate and load components ---
        try:
            # DataProcessor automatically calls load_artifacts in its __init__
            data_processor: SentimentDataProcessor = SentimentDataProcessor(model_artifacts_path=model_artifacts_path)
            model_core: SentimentModelCore = SentimentModelCore(model_path=model_file_path)

            if not data_processor.tokenizer or not data_processor.max_sequence_length:
                self._parser.error(
                    f"Tokenizer or max_sequence_length could not be loaded from artifacts at: {model_artifacts_path}"
                )
                return

        except FileNotFoundError as e:
            self._parser.error(f"Failed to load model artifacts: {e}")
            return
        except Exception as e:
            self._parser.error(f"An unexpected error occurred while loading components: {e}")
            return

        # --- 3. Process input and make a prediction ---
        try:
            # Process the single input sentence
            processed_input: NDArray[any] = data_processor.process_for_prediction(
                single_input=args.input_sentence
            )

            # Create the required configuration for prediction
            pred_config: SentimentPredictConfig = SentimentPredictConfig(verbose=0)

            prediction: NDArray[any] = model_core.predict(data=processed_input, config=pred_config)
            sentiment_score: float = float(prediction[0][0])

            # --- 4. Present the result ---
            sentiment_label: str = "Positive" if sentiment_score > 0.5 else "Negative"
            self._logger.info("--- Prediction Result ---")
            self._logger.info(f"  Input Sentence: '{args.input_sentence}'")
            self._logger.info(f"  Predicted Score: {sentiment_score:.4f}")
            self._logger.info(f"  Predicted Label: {sentiment_label}")
            self._logger.info("-------------------------")

        except Exception as e:
            self._logger.error(f"An error occurred during prediction: {e}", exc_info=True)
            self._parser.error("Prediction failed. Check logs for details.")

    @override
    def get_metrics(self, args: Namespace) -> None:
        """
        Retrieves and displays evaluation metrics for a model at a specific epoch.

        This method runs a full evaluation for the specified model and epoch,
        then prints the metrics requested by the user via CLI flags.

        :param args: The namespace object from argparse, containing `model_id`,
                     `epoch`, and flags for which metrics to retrieve.
        """
        if not (args.training_loss or args.validation_loss or args.f1_score):
            self._parser.error(
                "At least one flag from --training-loss, --validation-loss, or --f1-score must be selected to get metrics."
            )

        self._logger.info(
            f"Executing: Get evaluation metrics for {self._model_type_name} model "
            f"'{args.model_id}' (epoch: {args.epoch})."
        )

        # --- 1. Load original training configuration ---
        config_path: Path = ProjectPaths.get_model_root_path(
            model_id=args.model_id, model_type=ProjectModelType.SENTIMENT
        ) / "config.yaml"
        if not config_path.exists():
            self._parser.error(f"Master config file 'config.yaml' not found for model_id '{args.model_id}'.")
            return
        try:
            original_config_data: dict[str, any] = YamlFile(path=config_path).load_single_document()
        except Exception as e:
            self._parser.error(f"Failed to load or parse config file at '{config_path}': {e}")
            return

        # --- 2. Run evaluation ---
        try:
            # noinspection PyTypeChecker
            data_proc_config = SentimentTrainingConfig(
                vocabulary_size=original_config_data['vocabulary_size'],
                split_ratios=tuple(original_config_data['split_ratios']),
                random_state=original_config_data['random_state']
            )
            eval_config = SentimentEvaluationConfig(
                model_id=args.model_id,
                model_epoch=args.epoch,
                evaluation_dataset_file_name=original_config_data['dataset_file_name'],
                data_processing_config=data_proc_config
            )
            result: SentimentEvaluationResult = self._evaluator.run(config=eval_config)
        except FileNotFoundError as e:
            self._parser.error(f"Evaluation failed: A required file was not found. Details: {e}")
            return
        except Exception as e:
            self._logger.error(f"An unexpected error occurred during evaluation: {e}", exc_info=True)
            self._parser.error("Evaluation failed. Check logs for details.")
            return

        # --- 3. Display results ---
        self._logger.info(f"--- Metrics for {args.model_id} @ Epoch {args.epoch} ---")
        if args.training_loss:
            try:
                # Epochs are 1-based, list indices are 0-based
                train_loss = result.training_loss_history[args.epoch - 1]
                self._logger.info(f"  - Training Loss:   {train_loss:.6f}")
            except IndexError:
                self._logger.warning(
                    f"  - Training Loss:   Not available for epoch {args.epoch} (history length: {len(result.training_loss_history)}).")

        if args.validation_loss:
            try:
                val_loss = result.validation_loss_history[args.epoch - 1]
                self._logger.info(f"  - Validation Loss: {val_loss:.6f}")
                self._logger.info(f"  - Test Loss:       {result.test_loss:.6f}")
            except IndexError:
                self._logger.warning(
                    f"  - Validation Loss: Not available for epoch {args.epoch} (history length: {len(result.validation_loss_history)}).")
                self._logger.info(f"  - Test Loss:       {result.test_loss:.6f}")

        if args.f1_score:
            self._logger.info(f"  - F1-Score (Test): {result.f1_score:.4f} ({result.f1_score:.2%})")
        self._logger.info("-------------------------------------------------")

    @override
    def plot_graph(self, args: Namespace) -> None:
        """
        Generates and saves evaluation graphs for a sentiment model series.

        This method orchestrates the evaluation of all model checkpoints and
        then plots the results based on user-selected metrics.

        :param args: The namespace object from argparse, containing `model_id`
                     and flags for which graphs to plot.
        """
        if not (args.training_loss or args.validation_loss or args.f1_score):
            self._parser.error(
                "At least one flag from --training-loss, --validation-loss, or --f1-score must be selected for plotting."
            )

        self._logger.info(f"Executing: Plot evaluation graphs for {self._model_type_name} model '{args.model_id}'.")

        try:
            # --- 1. Collect all evaluation data ---
            eval_results: SentimentMultiEpochEvaluationResult = self._evaluate_all_epochs(model_id=args.model_id)
        except (FileNotFoundError, ValueError) as e:
            self._parser.error(str(e))
            return

        # --- 2. Prepare and plot graphs based on user selection ---
        output_dir: Path = ProjectPaths.project_root / "outputs" / "graphs" / args.model_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Plot Loss graph
        if args.training_loss or args.validation_loss:
            datasets_to_plot: list[PlotDataset] = []
            # The x-axis for history is 1-based epoch numbers
            history_epochs = list(range(1, len(eval_results.full_training_loss_history) + 1))

            if args.training_loss:
                datasets_to_plot.append({
                    "label": "Training Loss",
                    "data": eval_results.full_training_loss_history
                })
            if args.validation_loss:
                datasets_to_plot.append({
                    "label": "Validation Loss",
                    "data": eval_results.full_validation_loss_history
                })
                # Align test loss data with the full epoch history for plotting
                test_loss_aligned_data = [float('nan')] * len(history_epochs)
                for i, epoch in enumerate(eval_results.evaluated_epochs):
                    if 1 <= epoch <= len(history_epochs):
                        test_loss_aligned_data[epoch - 1] = eval_results.test_losses[i]

                datasets_to_plot.append({
                    "label": "Test Loss",
                    "data": test_loss_aligned_data
                })

            plot_multi_line_graph(
                title=f"Loss Curves for {args.model_id}",
                save_path=output_dir / "loss_curves.png",
                x_data=history_epochs,
                y_datasets=datasets_to_plot,
                x_label="Epoch",
                y_label="Loss",
                y_formatter='sci-notation'
            )

        # Plot F1-Score graph
        if args.f1_score:
            plot_multi_line_graph(
                title=f"F1-Score on Test Set for {args.model_id}",
                save_path=output_dir / "f1_score_curve.png",
                x_data=eval_results.evaluated_epochs,
                y_datasets=[{"label": "Test F1-Score", "data": eval_results.test_f1_scores}],
                x_label="Epoch",
                y_label="F1-Score",
                y_formatter='percent'
            )


    def _evaluate_all_epochs(self, model_id: str) -> SentimentMultiEpochEvaluationResult:
        """
        Finds and evaluates all available checkpoints for a model, using a cache
        to avoid re-computation, and returns aggregated results.

        :param model_id: The unique identifier for the model series to evaluate.
        :returns: A SentimentMultiEpochEvaluationResult object containing all collected metrics.
        :raises FileNotFoundError: If the master config or history file for the model is not found.
        :raises ValueError: If evaluation fails for all available epochs.
        """
        # --- 1. Setup paths and find available epochs ---
        artifacts_folder: Path = ProjectPaths.get_model_root_path(
            model_id=model_id, model_type=ProjectModelType.SENTIMENT
        )
        available_epochs: list[int] = self._find_available_epochs(
            model_id=model_id,
            model_type=ProjectModelType.SENTIMENT
        )
        if not available_epochs:
            self._parser.error(f"No model checkpoints found for model_id '{model_id}'. Cannot generate plots.")
            raise ValueError(f"No checkpoints found for {model_id}")

        self._logger.info(f"Found {len(available_epochs)} checkpoints to evaluate: {available_epochs}")

        # --- 2. Load master config ---
        config_path: Path = artifacts_folder / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Master config file 'config.yaml' not found for model_id '{model_id}'.")
        original_config_data: dict[str, any] = YamlFile(path=config_path).load_single_document()

        # --- 3. Load or initialize cache ---
        cache_path: Path = artifacts_folder / self._EVALUATION_CACHE_FILE_NAME
        cache_handler = YamlFile(path=cache_path)
        cached_results: dict[int, dict[str, float]] = {}
        if cache_path.exists():
            try:
                loaded_cache = cache_handler.load_single_document()
                if isinstance(loaded_cache, dict):
                    # YAML keys might be loaded as strings, convert them to int
                    cached_results = {int(k): v for k, v in loaded_cache.items() if isinstance(v, dict)}
                    self._logger.info(f"Loaded {len(cached_results)} results from cache: {cache_path}")
            except Exception as e:
                self._logger.warning(f"Could not load or parse cache file at {cache_path}. Re-evaluating. Error: {e}")

        # --- 4. Iterate, evaluate, and collect data ---
        test_f1_scores: list[float] = []
        test_losses: list[float] = []
        first_eval_result: SentimentEvaluationResult | None = None

        for epoch in available_epochs:
            if epoch in cached_results:
                self._logger.info(f"--- Using cached evaluation for epoch {epoch} ---")
                cached_metric = cached_results[epoch]
                test_f1_scores.append(cached_metric.get('f1_score', float('nan')))
                test_losses.append(cached_metric.get('loss', float('nan')))
                continue

            self._logger.info(f"--- Evaluating epoch {epoch} for plotting (not cached) ---")
            # noinspection PyTypeChecker
            data_proc_config = SentimentTrainingConfig(
                vocabulary_size=original_config_data['vocabulary_size'],
                split_ratios=tuple(original_config_data['split_ratios']),
                random_state=original_config_data['random_state']
            )
            eval_config = SentimentEvaluationConfig(
                model_id=model_id,
                model_epoch=epoch,
                evaluation_dataset_file_name=original_config_data['dataset_file_name'],
                data_processing_config=data_proc_config
            )
            try:
                result: SentimentEvaluationResult = self._evaluator.run(config=eval_config)
                test_f1_scores.append(result.f1_score)
                test_losses.append(result.test_loss)
                cached_results[epoch] = {'f1_score': result.f1_score, 'loss': result.test_loss}

                if first_eval_result is None:
                    first_eval_result = result
            except Exception as e:
                self._logger.error(f"Failed to evaluate epoch {epoch}: {e}. Skipping this epoch for plot.")
                test_f1_scores.append(float('nan'))
                test_losses.append(float('nan'))

        # --- 5. Save updated cache ---
        try:
            cache_handler.save_single_document(data=cached_results)
            self._logger.info(f"Updated evaluation cache file at: {cache_path}")
        except Exception as e:
            self._logger.error(f"Failed to save evaluation cache to {cache_path}: {e}")

        # --- 6. Consolidate results ---
        if not any(not (isinstance(v, float) and v != v) for v in test_f1_scores):
            raise ValueError("Evaluation failed for all epochs. Cannot generate plots.")

        # Load history. If an evaluation ran, use its history. Otherwise, load manually.
        full_training_loss_history: list[float] = []
        full_validation_loss_history: list[float] = []
        if first_eval_result:
            full_training_loss_history = first_eval_result.training_loss_history
            full_validation_loss_history = first_eval_result.validation_loss_history
        elif cached_results:
            self._logger.info("All results were from cache. Loading training history manually.")
            history_path = artifacts_folder / self._evaluator.HISTORY_FILE_NAME
            if history_path.exists():
                try:
                    full_training_loss_history, full_validation_loss_history = self._evaluator.load_training_history(
                        history_file_path=history_path
                    )
                except Exception as e:
                    self._logger.error(f"Failed to load history file {history_path} even though cache exists: {e}")
            else:
                self._logger.warning(f"Evaluation cache exists, but history file {history_path} is missing.")

        return SentimentMultiEpochEvaluationResult(
            model_id=model_id,
            evaluated_epochs=available_epochs,
            test_f1_scores=test_f1_scores,
            test_losses=test_losses,
            full_training_loss_history=full_training_loss_history,
            full_validation_loss_history=full_validation_loss_history
        )

class PredictionModelHandler(BaseModelHandler):
    """
    Handles CLI commands related to the box office prediction model.
    """

    @override
    def __init__(self, parser: ArgumentParser) -> None:
        """
        Initializes the PredictionModelHandler.

        :param parser: The argument parser instance.
        """
        super().__init__(parser=parser, model_type_name="Prediction")

    @override
    def train(self, args: Namespace) -> None:
        """
        Orchestrates the box office prediction model training process.

        Note: This method is a placeholder and its full implementation is pending.

        :param args: The namespace object from argparse, containing training parameters.
        """
        # TODO: Add actual testing logic here
        pass

    @override
    def predict(self, args: Namespace) -> None:
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

    @override
    def get_metrics(self, args: Namespace) -> None:
        """
        Retrieves and displays evaluation metrics for a prediction model.

        Note: This method is a placeholder and its full implementation is pending.

        :param args: The namespace object from argparse, containing evaluation parameters.
        """
        # TODO: Add actual testing logic here
        pass

    @override
    def plot_graph(self, args: Namespace) -> None:
        """
        Generates and saves evaluation graphs for a prediction model series.

        Note: This method is a placeholder and its full implementation is pending.

        :param args: The namespace object from argparse, containing plotting parameters.
        """
        # TODO: Add actual testing logic here
        pass

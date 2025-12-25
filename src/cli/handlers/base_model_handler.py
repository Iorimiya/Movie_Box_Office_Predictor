from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from logging import Logger
from pathlib import Path
from random import randint

from src.core.logging_manager import LoggingManager
from src.core.project_config import ProjectModelType, ProjectPaths
from src.data_handling.file_io import PickleFile, YamlFile
from src.models.base.evaluation import BaseEvaluationConfig, BaseEvaluationResult, BaseEvaluator


class BaseModelHandler(ABC):
    """
    An abstract base class for model handlers to reduce code duplication.

    This class provides a common structure and reusable logic for handling
    model-related CLI commands like training, prediction, and evaluation.
    Subclasses must implement the abstract methods to provide model-specific behavior.

    :ivar _logger: The shared logger instance for all model handlers.
    :ivar _parser: The argument parser instance for the specific command.
    :ivar _model_type_name: The display name of the model type (e.g., "Sentiment").
    :ivar _model_type: The enum member for the model type.
    :ivar _evaluator: An instance of a class that inherits from BaseEvaluator.
    """
    _logger: Logger
    _parser: ArgumentParser
    _model_type_name: str
    _model_type: ProjectModelType
    _evaluator: BaseEvaluator

    def __init__(
        self, parser: ArgumentParser, model_type_name: str, model_type: ProjectModelType, evaluator: BaseEvaluator
    ) -> None:
        """
        Initializes the BaseModelHandler.

        :param parser: The argument parser instance.
        :param model_type_name: The name of the model type for logging.
        :param model_type: The enum member for the model type.
        :param evaluator: An instance of a class that inherits from BaseEvaluator.
        """
        self._parser = parser
        self._model_type_name = model_type_name
        self._model_type = model_type
        self._evaluator = evaluator
        self._logger = LoggingManager().get_logger()

    @abstractmethod
    def train(self, args: Namespace) -> None:
        """
        Handles the model training process based on provided arguments.

        Subclasses must implement this method to define the specific training
        pipeline for their model type.

        :param args: The namespace object containing command-line arguments.
        """
        pass

    @abstractmethod
    def predict(self, args: Namespace) -> None:
        """
        Handles making a prediction with a trained model.

        Subclasses must implement this method to define how to load a model
        and process input for prediction.

        :param args: The namespace object containing command-line arguments.
        """
        pass

    def get_metrics(self, args: Namespace) -> None:
        """
        Retrieves and displays evaluation metrics for a model at a specific epoch.

        This template method handles the common workflow of preparing the context,
        running the evaluation, and delegating the final display of metrics
        to a subclass-specific implementation.

        :param args: The namespace object from argparse, containing evaluation parameters.
        """
        metric_flags: list[str] = [
            'training_loss', 'validation_loss', 'test_loss',
            'classification_report', 'show_f1_score', 'show_confusion_matrix'
        ]
        original_config_data: dict[str, any] = self._prepare_evaluation_context(args=args, required_flags=metric_flags)

        try:
            eval_config: any = self._build_evaluation_config(
                args=args,
                original_config_data=original_config_data,
                epoch_to_evaluate=args.epoch
            )
            result: BaseEvaluationResult = self._run_evaluation_for_epoch(eval_config=eval_config)
        except (FileNotFoundError, ValueError) as e:
            self._parser.error(f"Evaluation failed: {e}")

        except Exception as e:
            self._logger.error(f"An unexpected error occurred during evaluation: {e}", exc_info=True)
            self._parser.error("Evaluation failed. Check logs for details.")

        self._display_metrics(result=result, args=args)

    @abstractmethod
    def plot_graph(self, args: Namespace) -> None:
        """
        Generates and saves evaluation graphs for a model series.

        Since all iterative models have metrics that change over epochs (Loss,
        Accuracy, Reward, etc.), this method is mandatory. Subclasses should
        use `_evaluate_all_epochs` to retrieve data and then implement their
        specific plotting logic.

        :param args: The namespace object from argparse.
        """
        pass

    @abstractmethod
    def _get_default_config_filename(self) -> str:
        """
        Gets the filename for the model's default configuration.

        Subclasses must implement this to return their specific default
        config file name (e.g., "sentiment_defaults.yaml").

        :returns: The name of the default configuration file.
        """
        pass

    def _prepare_training_config(self, args: Namespace) -> dict[str, any]:
        """
        A template method to prepare the final configuration for a training run.

        This method encapsulates the entire logic for handling new vs. continued
        training, loading default configurations, applying overrides from files
        or individual CLI arguments, and saving the final effective configuration.

        :param args: The namespace object from argparse.
        :returns: A dictionary containing the final, effective configuration.
        :raises SystemExit: If configuration rules are violated (e.g., using
                            overrides with --continue-from-epoch), or if files
                            are not found.
        """
        model_id: str = args.model_id
        artifacts_folder: Path = ProjectPaths.get_model_root_path(
            model_id=model_id, model_type=self._model_type
        )
        final_config_path: Path = artifacts_folder / "config.yaml"

        # Cache Invalidation
        # A new training run will invalidate any previous evaluation results.
        # This requires an abstract method to get the specific cache file name.
        cache_filename: str = self._get_evaluation_cache_filename()
        cache_path: Path = artifacts_folder / cache_filename
        if cache_path.exists():
            self._logger.info(f"Invalidating evaluation cache at '{cache_path}' due to new training run.")
            cache_path.unlink()

        # Main logic branch: New vs. Continue
        if args.continue_from_epoch:
            # Mode: Continue Training
            self._logger.info(
                f"Executing: Continue training {self._model_type_name} model '{model_id}' "
                f"from epoch {args.continue_from_epoch}."
            )

            # Rule: No new overrides are allowed when continuing training
            if args.config_override:
                self._parser.error("Argument --config-override cannot be used with --continue-from-epoch.")

            individual_overrides: dict[str, any] = self._get_individual_overrides(args=args, is_continue_mode=True)
            if individual_overrides:
                self._parser.error(
                    f"Individual overrides like --{next(iter(individual_overrides))} cannot be used with --continue-from-epoch.")

            # Rule: The original config.yaml must be found
            if not final_config_path.exists():
                self._parser.error(
                    f"Cannot continue training: Original config.yaml not found for model '{model_id}' at '{final_config_path}'."
                )

            self._logger.info(f"Using existing configuration file: {final_config_path}")
            return YamlFile(path=final_config_path).load_single_document()

        else:
            # Mode: New Training
            self._logger.info(f"Executing: Start new training for {self._model_type_name} model '{model_id}'.")

            # Check for mutually exclusive arguments
            individual_overrides: dict[str, any] = self._get_individual_overrides(args=args, is_continue_mode=False)
            if args.config_override and individual_overrides:
                self._parser.error("Argument --config-override cannot be used with individual parameter overrides.")

            # Load default configuration using the abstract method
            default_config_filename: str = self._get_default_config_filename()
            default_config_path: Path = ProjectPaths.get_config_path(config_name=default_config_filename)
            try:
                default_config: dict[str, any] = YamlFile(path=default_config_path).load_single_document()
                self._logger.info(f"Loaded default configuration from: {default_config_path}")
            except FileNotFoundError:
                self._parser.error(
                    f"Default configuration file '{default_config_filename}' not found at: {default_config_path}")

            # Apply overrides
            effective_config: dict[str, any] = default_config.copy()
            if args.config_override:
                try:
                    self._logger.info(f"Applying overrides from file: {args.config_override}")
                    override_config: dict[str, any] = YamlFile(path=args.config_override).load_single_document()
                    effective_config.update(override_config)
                except FileNotFoundError:
                    self._parser.error(f"Override configuration file not found: {args.config_override}")
            elif individual_overrides:
                self._logger.info(f"Applying individual overrides: {individual_overrides}")
                effective_config.update(individual_overrides)

            # Handle Random State
            if effective_config.get('random_state') is None:
                new_random_state: int = randint(0, 2 ** 32 - 1)
                effective_config['random_state'] = new_random_state
                self._logger.warning(
                    "The 'random_state' was not provided in the configuration. "
                    f"A new random state has been generated: {new_random_state}"
                )
                self._logger.warning(
                    "For full reproducibility, please add this 'random_state' to your configuration file for future runs."
                )

            # Create artifact directory and save the final configuration
            artifacts_folder.mkdir(parents=True, exist_ok=True)
            effective_config['model_id'] = model_id
            try:
                YamlFile(path=final_config_path).save_single_document(data=effective_config)
                self._logger.info(f"Effective configuration saved to: {final_config_path}")
            except Exception as e:
                self._parser.error(f"Failed to save final configuration file: {e}")

            return effective_config

    @abstractmethod
    def _get_evaluation_cache_filename(self) -> str:
        """
        Gets the filename for the model's evaluation cache.

        Subclasses must implement this to return their specific evaluation
        cache file name (e.g., "evaluation_cache.yaml").

        :returns: The name of the evaluation cache file.
        """
        pass

    @abstractmethod
    def _build_evaluation_config(
        self, args: Namespace, original_config_data: dict[str, any], epoch_to_evaluate: int
    ) -> BaseEvaluationConfig:
        """
        Builds the appropriate evaluation configuration object for a single epoch.

        Subclasses must implement this to construct a model-specific evaluation
        config object from the command-line arguments and the original training config.

        :param args: The namespace object from argparse.
        :param original_config_data: The loaded dictionary from the model's config.yaml.
        :param epoch_to_evaluate: The specific epoch to be evaluated.
        :returns: A model-specific evaluation configuration object.
        """
        pass

    @abstractmethod
    def _run_evaluation_for_epoch(self, eval_config: BaseEvaluationConfig) -> BaseEvaluationResult:
        """
        Runs the evaluation for a single epoch using the model-specific evaluator.

        Subclasses must implement this to call their specific evaluator instance
        and return the results.

        :param eval_config: The configuration object for the evaluation.
        :returns: The result object from the evaluation.
        """
        pass

    def _evaluate_all_epochs(self, model_id: str, args: Namespace) -> list[BaseEvaluationResult]:
        """
        Finds and evaluates all available checkpoints for a model.

        This template method uses a cache to avoid re-computation and returns
        a list of full evaluation result objects.

        :param model_id: The unique identifier for the model series to evaluate.
        :param args: The namespace object from argparse, used to determine evaluation mode.
        :returns: A list of `BaseEvaluationResult` objects, one for each successfully evaluated epoch.
        :raises FileNotFoundError: If the master config or history file for the model is not found.
        :raises ValueError: If evaluation fails for all available epochs.
        """
        metric_flags: list[str] = [
            'training_loss', 'validation_loss', 'test_loss',
            'classification_report', 'show_f1_score', 'show_confusion_matrix'
        ]
        original_config_data: dict[str, any] = self._prepare_evaluation_context(args=args, required_flags=metric_flags)

        artifacts_folder: Path = ProjectPaths.get_model_root_path(model_id=model_id, model_type=self._model_type)
        available_epochs: list[int] = self._find_available_epochs(model_id=model_id, model_type=self._model_type)
        if not available_epochs:
            self._parser.error(f"No model checkpoints found for model_id '{model_id}'. Cannot generate plots.")

        self._logger.info(f"Found {len(available_epochs)} checkpoints to evaluate: {available_epochs}")

        cache_filename: str = self._get_evaluation_cache_filename()
        cache_path: Path = artifacts_folder / cache_filename
        cache_handler: PickleFile = PickleFile(path=cache_path)
        cached_results: list[BaseEvaluationResult] = []
        if cache_path.exists() and not args.dataset_name:
            try:
                loaded_cache: list[BaseEvaluationResult] = cache_handler.load()
                if isinstance(loaded_cache, list) and all(isinstance(e, BaseEvaluationResult) for e in loaded_cache):
                    cached_results = loaded_cache
                    self._logger.info(f"Loaded {len(cached_results)} results from cache: {cache_path}")
            except Exception as e:
                self._logger.warning(f"Could not load or parse cache file at {cache_path}. Re-evaluating. Error: {e}")
        elif args.dataset_name:
            self._logger.info(f"Ignoring cache because a new dataset '{args.dataset_name}' was specified.")

        cached_results_map: dict[int, BaseEvaluationResult] = {
            res.model_epoch: res for res in cached_results
        }

        collected_results: list[BaseEvaluationResult] = []

        for epoch in available_epochs:
            if epoch in cached_results_map and not args.dataset_name:
                self._logger.info(f"--- Using cached evaluation for epoch {epoch} ---")
                collected_results.append(cached_results_map[epoch])
                continue
            self._logger.info(f"--- Evaluating epoch {epoch} for plotting ---")
            eval_config: BaseEvaluationConfig = self._build_evaluation_config(
                args=args, original_config_data=original_config_data, epoch_to_evaluate=epoch
            )

            try:
                result: BaseEvaluationResult = self._run_evaluation_for_epoch(eval_config=eval_config)
                collected_results.append(result)

                if not args.dataset_name:
                    cached_results_map[epoch] = result

            except Exception as e:
                self._logger.error(f"Failed to evaluate epoch {epoch}: {e}. Skipping this epoch for plot.")

        if not args.dataset_name and collected_results:
            try:
                valid_results: list[BaseEvaluationResult] = [
                    r for r in collected_results if isinstance(r, BaseEvaluationResult)
                ]
                cache_handler.save(data=valid_results)
                self._logger.info(f"Updated evaluation cache file at: {cache_path}")
            except Exception as e:
                self._logger.error(f"Failed to save evaluation cache to {cache_path}: {e}")

        return collected_results

    @staticmethod
    def _find_available_epochs(model_id: str, model_type: ProjectModelType) -> list[int]:
        """
        Finds all available model checkpoint epochs for a given model ID and type.

        This static utility method scans the appropriate model artifact
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
                epoch_str: str = f.stem.split('_')[-1]
                epochs.append(int(epoch_str))
            except (ValueError, IndexError):
                continue
        return sorted(epochs)

    @staticmethod
    def _get_individual_overrides(args: Namespace, is_continue_mode: bool = False) -> dict[str, any]:
        """
        Extracts individual parameter overrides from the argparse Namespace.

        This helper method filters out arguments that are not considered
        overridable parameters.

        :param args: The namespace object from argparse.
        :param is_continue_mode: A flag to adjust the keys to exclude for continuation mode.
        :returns: A dictionary of individual override parameters.
        """
        # Keys that are part of the CLI mechanism, not overridable config values
        # The subcommand key can vary, so find it dynamically.
        subcommand_key: str | None = next((key for key in vars(args) if key.endswith('_subcommand')), None)

        base_exclude_keys: set[str] = {'command_group', 'func', 'model_id', 'config_override'}
        if subcommand_key:
            base_exclude_keys.add(subcommand_key)

        if is_continue_mode:
            base_exclude_keys.add('continue_from_epoch')

        return {
            key: value for key, value in vars(args).items()
            if key not in base_exclude_keys and value is not None
        }

    def _prepare_evaluation_context(self, args: Namespace, required_flags: list[str]) -> dict[str, any]:
        """
        Performs common setup tasks for evaluation commands.

        This includes validating metric flags, logging initial messages, and loading
        the master configuration file for the specified model.

        :param args: The namespace object from argparse.
        :param required_flags: A list of attribute names on `args` to check for truthiness.
        :returns: The loaded dictionary from the model's config.yaml.
        :raises SystemExit: If validation fails or the config file cannot be loaded.
        """
        # Validate that at least one metric flag is present
        if not any(getattr(args, flag, False) for flag in required_flags):
            self._parser.error(
                f"At least one flag from --{', --'.join(flag.replace('_', '-') for flag in required_flags)} must be selected."
            )

        # Log initial messages
        command_name: str = "Get evaluation metrics" if 'epoch' in args else "Plot evaluation graphs"
        self._logger.info(
            f"Executing: {command_name} for {self._model_type_name} model '{args.model_id}'."
        )
        if args.dataset_name:
            self._logger.info(f"Evaluation will be performed on new dataset: '{args.dataset_name}'")

        # Load original training configuration
        config_path: Path = ProjectPaths.get_model_root_path(
            model_id=args.model_id, model_type=self._model_type
        ) / "config.yaml"
        if not config_path.exists():
            self._parser.error(f"Master config file 'config.yaml' not found for model_id '{args.model_id}'.")

        try:
            return YamlFile(path=config_path).load_single_document()
        except Exception as e:
            self._parser.error(f"Failed to load or parse config file at '{config_path}': {e}")

    @abstractmethod
    def _display_specific_metrics(self, result: BaseEvaluationResult, args: Namespace, result_logger: Logger) -> None:
        """
        Displays model-specific metrics using a format-less logger.

        Subclasses must implement this method to print any metrics unique to
        their model type using the provided `result_logger`. This logger is
        configured to output messages without any standard log formatting
        (like timestamps or log levels), making it suitable for clean final output.

        :param result: The evaluation result object containing the metrics.
        :param args: The command-line arguments, used to determine which metrics to display.
        :param result_logger: The logger instance configured for clean, format-less output.
        """
        pass

    def _display_metrics(self, result: BaseEvaluationResult, args: Namespace) -> None:
        """
        Displays evaluation metrics in a structured format.

        This template method displays common metrics (e.g., loss, F1-score) and
        then delegates to a subclass to display any model-specific metrics.

        :param result: The evaluation result object.
        :param args: The command-line arguments to check which metrics to display.
        """
        pass

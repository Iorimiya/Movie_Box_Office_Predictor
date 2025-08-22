from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from random import randint

from src.core.logging_manager import LoggingManager
from src.core.project_config import ProjectModelType, ProjectPaths
from src.data_handling.file_io import YamlFile
from src.models.base.base_evaluator import BaseEvaluationResult, BaseEvaluationConfig, BaseEvaluator
from src.models.base.base_pipeline import BaseTrainingPipeline
from src.utilities.plot import PlotDataset, plot_multi_line_graph


@dataclass(frozen=True)
class MultiEpochEvaluationResult:
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

        metric_flags: list[str] = ['training_loss', 'validation_loss', 'f1_score', 'test_loss']
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

    def plot_graph(self, args: Namespace) -> None:
        """
        Generates and saves evaluation graphs for a model series.

        This template method orchestrates the plotting process by first fetching
        all necessary evaluation data and then delegating the actual plotting
        of specific graphs to subclass implementations.

        :param args: The namespace object from argparse, containing `model_id`
                     and flags for which graphs to plot.
        """
        try:
            eval_results: MultiEpochEvaluationResult = self._evaluate_all_epochs(
                model_id=args.model_id, args=args
            )
        except (FileNotFoundError, ValueError) as e:
            self._parser.error(str(e))

        output_dir: Path = ProjectPaths.get_model_plots_path(
            model_id=args.model_id, model_type=self._model_type
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.training_loss or args.validation_loss or args.test_loss:
            self._plot_loss_graph(eval_results=eval_results, output_dir=output_dir, args=args)

        if args.f1_score:
            self._plot_f1_score_graph(eval_results=eval_results, output_dir=output_dir)

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

        # --- Cache Invalidation ---
        # A new training run will invalidate any previous evaluation results.
        # This requires an abstract method to get the specific cache file name.
        cache_filename: str = self._get_evaluation_cache_filename()
        cache_path: Path = artifacts_folder / cache_filename
        if cache_path.exists():
            self._logger.info(f"Invalidating evaluation cache at '{cache_path}' due to new training run.")
            cache_path.unlink()

        # --- Main logic branch: New vs. Continue ---
        if args.continue_from_epoch:
            # --- Mode: Continue Training ---
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
            # --- Mode: New Training ---
            self._logger.info(f"Executing: Start new training for {self._model_type_name} model '{model_id}'.")

            # --- Check for mutually exclusive arguments ---
            individual_overrides: dict[str, any] = self._get_individual_overrides(args=args, is_continue_mode=False)
            if args.config_override and individual_overrides:
                self._parser.error("Argument --config-override cannot be used with individual parameter overrides.")

            # --- Load default configuration using the abstract method ---
            default_config_filename: str = self._get_default_config_filename()
            default_config_path: Path = ProjectPaths.get_config_path(config_name=default_config_filename)
            try:
                default_config: dict[str, any] = YamlFile(path=default_config_path).load_single_document()
                self._logger.info(f"Loaded default configuration from: {default_config_path}")
            except FileNotFoundError:
                self._parser.error(
                    f"Default configuration file '{default_config_filename}' not found at: {default_config_path}")

            # --- Apply overrides ---
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

            # --- Handle Random State ---
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

            # --- Create artifact directory and save the final configuration ---
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

    @staticmethod
    def _create_multi_epoch_evaluation_result(
        model_id: str,
        available_epochs: list[int],
        collected_metrics: dict[str, list[float]],
        training_history: list[float],
        validation_history: list[float]
    ) -> MultiEpochEvaluationResult:
        """
        Constructs the final aggregated multi-epoch result object.

        :param model_id: The unique identifier for the model series.
        :param available_epochs: A list of all epoch numbers that were evaluated.
        :param collected_metrics: A dictionary where keys are metric names (e.g., 'loss', 'f1_score')
                                  and values are lists of metric values for each epoch.
        :param training_history: The full training loss history.
        :param validation_history: The full validation loss history.
        :returns: An instance of the `MultiEpochEvaluationResult` dataclass.
        """
        return MultiEpochEvaluationResult(
            model_id=model_id,
            evaluated_epochs=available_epochs,
            test_losses=collected_metrics.get('loss', []),
            test_f1_scores=collected_metrics.get('f1_score', []),
            full_training_loss_history=training_history,
            full_validation_loss_history=validation_history
        )

    def _evaluate_all_epochs(self, model_id: str, args: Namespace) -> MultiEpochEvaluationResult:
        """
        Finds and evaluates all available checkpoints for a model.

        This template method uses a cache to avoid re-computation and returns
        aggregated results. It provides a generic workflow that relies on abstract
        methods implemented by subclasses for model-specific logic.

        :param model_id: The unique identifier for the model series to evaluate.
        :param args: The namespace object from argparse, used to determine evaluation mode.
        :returns: A `MultiEpochEvaluationResult` object containing aggregated results.
        :raises FileNotFoundError: If the master config or history file for the model is not found.
        :raises ValueError: If evaluation fails for all available epochs.
        """
        metric_flags = ['training_loss', 'validation_loss', 'f1_score', 'test_loss']
        original_config_data = self._prepare_evaluation_context(args=args, required_flags=metric_flags)

        artifacts_folder: Path = ProjectPaths.get_model_root_path(model_id=model_id, model_type=self._model_type)
        available_epochs: list[int] = self._find_available_epochs(model_id=model_id, model_type=self._model_type)
        if not available_epochs:
            self._parser.error(f"No model checkpoints found for model_id '{model_id}'. Cannot generate plots.")

        self._logger.info(f"Found {len(available_epochs)} checkpoints to evaluate: {available_epochs}")

        cache_filename = self._get_evaluation_cache_filename()
        cache_path: Path = artifacts_folder / cache_filename
        cache_handler = YamlFile(path=cache_path)
        cached_results: dict[int, dict[str, float]] = {}
        if cache_path.exists() and not args.dataset_name:
            try:
                loaded_cache = cache_handler.load_single_document()
                if isinstance(loaded_cache, dict):
                    cached_results = {int(k): v for k, v in loaded_cache.items() if isinstance(v, dict)}
                    self._logger.info(f"Loaded {len(cached_results)} results from cache: {cache_path}")
            except Exception as e:
                self._logger.warning(f"Could not load or parse cache file at {cache_path}. Re-evaluating. Error: {e}")
        elif args.dataset_name:
            self._logger.info(f"Ignoring cache because a new dataset '{args.dataset_name}' was specified.")

        collected_metrics: dict[str, list[float]] = {}
        first_eval_result: any = None

        for epoch in available_epochs:
            if epoch in cached_results and not args.dataset_name:
                self._logger.info(f"--- Using cached evaluation for epoch {epoch} ---")
                for metric_name, value in cached_results[epoch].items():
                    collected_metrics.setdefault(metric_name, []).append(value)
                continue

            self._logger.info(f"--- Evaluating epoch {epoch} for plotting ---")
            eval_config = self._build_evaluation_config(
                args=args, original_config_data=original_config_data, epoch_to_evaluate=epoch
            )

            try:
                result: BaseEvaluationResult = self._run_evaluation_for_epoch(eval_config=eval_config)
                epoch_metrics = {'loss': result.test_loss, 'f1_score': result.f1_score}

                for metric_name, value in epoch_metrics.items():
                    collected_metrics.setdefault(metric_name, []).append(value)

                if not args.dataset_name:
                    cached_results[epoch] = epoch_metrics

                if first_eval_result is None:
                    first_eval_result = result
            except Exception as e:
                self._logger.error(f"Failed to evaluate epoch {epoch}: {e}. Skipping this epoch for plot.")
                # Append NaN to all collected metric lists to maintain alignment
                # Initialize keys if they don't exist yet
                if not collected_metrics:
                    collected_metrics['loss'] = []
                    collected_metrics['f1_score'] = []
                for metric_list in collected_metrics.values():
                    metric_list.append(float('nan'))

        if not args.dataset_name and cached_results:
            try:
                cache_handler.save_single_document(data=cached_results)
                self._logger.info(f"Updated evaluation cache file at: {cache_path}")
            except Exception as e:
                self._logger.error(f"Failed to save evaluation cache to {cache_path}: {e}")

        # Check if any valid results were obtained
        all_metric_values = [val for sublist in collected_metrics.values() for val in sublist]
        if not any(v is not None and not (isinstance(v, float) and v != v) for v in all_metric_values):
            raise ValueError("Evaluation failed for all epochs. Cannot generate plots.")

        full_training_loss_history: list[float] = []
        full_validation_loss_history: list[float] = []
        if first_eval_result:
            full_training_loss_history = first_eval_result.training_loss_history
            full_validation_loss_history = first_eval_result.validation_loss_history
        elif cached_results:
            self._logger.info("All results were from cache. Loading training history manually.")
            history_path: Path = artifacts_folder / BaseTrainingPipeline.HISTORY_FILE_NAME
            if history_path.exists():
                try:
                    full_training_loss_history, full_validation_loss_history = self._evaluator.load_training_history(
                        history_file_path=history_path
                    )
                except Exception as e:
                    self._logger.error(f"Failed to load history file {history_path} even though cache exists: {e}")
            else:
                self._logger.warning(f"Evaluation cache exists, but history file {history_path} is missing.")

        return self._create_multi_epoch_evaluation_result(
            model_id=model_id,
            available_epochs=available_epochs,
            collected_metrics=collected_metrics,
            training_history=full_training_loss_history,
            validation_history=full_validation_loss_history
        )

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
                epoch_str = f.stem.split('_')[-1]
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

        base_exclude_keys = {'command_group', 'func', 'model_id', 'config_override'}
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
        command_name = "Get evaluation metrics" if 'epoch' in args else "Plot evaluation graphs"
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
    def _display_specific_metrics(self, result: BaseEvaluationResult, args: Namespace) -> None:
        """
        Displays model-specific metrics that are not common to all models.

        Subclasses must implement this method to print any metrics unique to
        their model type.

        :param result: The evaluation result object containing the metrics.
        :param args: The command-line arguments.
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
        self._logger.info(f"--- Metrics for {args.model_id} @ Epoch {args.epoch} ---")
        if args.training_loss:
            try:
                train_loss: float = result.training_loss_history[args.epoch - 1]
                self._logger.info(f"  - Training Loss:   {train_loss:.6f}")
            except IndexError:
                self._logger.warning(
                    f"  - Training Loss:   Not available for epoch {args.epoch} (history length: {len(result.training_loss_history)}).")

        if args.validation_loss:
            try:
                val_loss: float = result.validation_loss_history[args.epoch - 1]
                self._logger.info(f"  - Validation Loss: {val_loss:.6f}")
            except IndexError:
                self._logger.warning(
                    f"  - Validation Loss: Not available for epoch {args.epoch} (history length: {len(result.validation_loss_history)}).")

        # Test loss is a common metric
        if args.test_loss:
            self._logger.info(f"  - Test Loss:       {result.test_loss:.6f}")

        # F1-score is a common metric
        if args.f1_score:
            self._logger.info(f"  - F1-Score (Test): {result.f1_score:.4f} ({result.f1_score:.2%})")

        self._display_specific_metrics(result=result, args=args)

        self._logger.info("-------------------------------------------------")

    @staticmethod
    def _plot_loss_graph(
        eval_results: MultiEpochEvaluationResult, output_dir: Path, args: Namespace
    ) -> None:
        """
        Plots the loss curves for a model.

        Includes training, validation, and test loss based on user flags.

        :param eval_results: The aggregated evaluation results for the model.
        :param output_dir: The directory to save the plot image.
        :param args: The command-line arguments to check which losses to plot.
        """
        datasets_to_plot: list[PlotDataset] = []
        history_epochs: list[int] = list(range(1, len(eval_results.full_training_loss_history) + 1))

        if args.training_loss:
            datasets_to_plot.append({"label": "Training Loss", "data": eval_results.full_training_loss_history})
        if args.validation_loss:
            datasets_to_plot.append({"label": "Validation Loss", "data": eval_results.full_validation_loss_history})
        if args.test_loss:
            # Align test loss data with the full epoch history
            test_loss_aligned_data = [float('nan')] * len(history_epochs)
            for i, epoch in enumerate(eval_results.evaluated_epochs):
                if 1 <= epoch <= len(history_epochs):
                    test_loss_aligned_data[epoch - 1] = eval_results.test_losses[i]
            datasets_to_plot.append({"label": "Test Loss", "data": test_loss_aligned_data})

        plot_multi_line_graph(
            title=f"Loss Curves for {eval_results.model_id}",
            save_path=output_dir / "loss_curves.png",
            x_data=history_epochs,
            y_datasets=datasets_to_plot,
            x_label="Epoch",
            y_label="Loss",
            y_formatter='sci-notation'
        )

    @staticmethod
    def _plot_f1_score_graph(eval_results: MultiEpochEvaluationResult, output_dir: Path) -> None:
        """
        Plots the F1-score curve for the sentiment model.

        :param eval_results: The aggregated evaluation results for the sentiment model.
        :param output_dir: The directory to save the plot image.
        """
        plot_multi_line_graph(
            title=f"F1-Score on Test Set for {eval_results.model_id}",
            save_path=output_dir / "f1_score_curve.png",
            x_data=eval_results.evaluated_epochs,
            y_datasets=[{"label": "Test F1-Score", "data": eval_results.test_f1_scores}],
            x_label="Epoch",
            y_label="F1-Score",
            y_formatter='percent'
        )

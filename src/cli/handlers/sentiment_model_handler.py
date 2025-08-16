from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from numpy.typing import NDArray
from typing_extensions import override

from src.cli.handlers.base_model_handler import BaseModelHandler
from src.core.project_config import ProjectModelType, ProjectPaths
from src.models.sentiment.components.data_processor import SentimentDataProcessor
from src.models.sentiment.components.evaluator import (
    SentimentEvaluator, SentimentEvaluationResult, SentimentEvaluationConfig
)
from src.models.sentiment.components.model_core import SentimentModelCore, SentimentPredictConfig
from src.models.sentiment.pipelines.training_pipeline import SentimentTrainingPipeline, SentimentPipelineConfig


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
        super().__init__(parser=parser, model_type_name="Sentiment", model_type=ProjectModelType.SENTIMENT)
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
        effective_config: dict[str, any] = self._prepare_training_config(args=args)

        try:
            artifacts_folder: Path = ProjectPaths.get_model_root_path(
                model_id=args.model_id, model_type=self._model_type
            )

            pipeline_config_process = lambda original: {'dataset_file_name' if k == 'dataset_name' else k: v for k, v in
                                                        original.items()}
            pipeline_config: SentimentPipelineConfig = SentimentPipelineConfig(
                **pipeline_config_process(effective_config))

            data_processor: SentimentDataProcessor = SentimentDataProcessor(model_artifacts_path=artifacts_folder)
            model_core: SentimentModelCore = SentimentModelCore()
            pipeline: SentimentTrainingPipeline = SentimentTrainingPipeline(
                data_processor=data_processor, model_core=model_core
            )

            pipeline.run(
                config=pipeline_config,
                continue_from_epoch=args.continue_from_epoch
            )
        except TypeError as e:
            self._logger.error(
                f"Configuration error: Mismatch between config data and pipeline requirements. Details: {e}",
                exc_info=True)
            self._parser.error("Pipeline execution failed due to a configuration mismatch. Check logs.")
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

        # --- Locate artifact paths ---
        model_artifacts_path: Path = ProjectPaths.get_model_root_path(
            model_id=args.model_id, model_type=ProjectModelType.SENTIMENT
        )
        model_file_path: Path = model_artifacts_path / f"{args.model_id}_{args.epoch:04d}.keras"

        if not model_file_path.exists():
            self._parser.error(f"Model file not found at: {model_file_path}")

        # --- Instantiate and load components ---
        try:
            # DataProcessor automatically calls load_artifacts in its __init__
            data_processor: SentimentDataProcessor = SentimentDataProcessor(model_artifacts_path=model_artifacts_path)
            model_core: SentimentModelCore = SentimentModelCore(model_path=model_file_path)

            if not data_processor.tokenizer or not data_processor.max_sequence_length:
                self._parser.error(
                    f"Tokenizer or max_sequence_length could not be loaded from artifacts at: {model_artifacts_path}"
                )

        except FileNotFoundError as e:
            self._parser.error(f"Failed to load model artifacts: {e}")
        except Exception as e:
            self._parser.error(f"An unexpected error occurred while loading components: {e}")

        # --- Process input and make a prediction ---
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
    def _get_default_config_filename(self) -> str:
        """
        Gets the filename for the sentiment model's default configuration.

        :returns: The name of the default configuration file.
        """
        return "sentiment_defaults.yaml"

    @override
    def _get_evaluation_cache_filename(self) -> str:
        """
        Gets the filename for the sentiment model's evaluation cache.

        :returns: The name of the evaluation cache file.
        """
        return self._EVALUATION_CACHE_FILE_NAME

    @override
    def _get_history_filename(self) -> str:
        """
        Returns the history filename for the sentiment model.
        """
        return self._evaluator.HISTORY_FILE_NAME

    @override
    def _load_training_history(self, history_file_path: Path) -> tuple[list[float], list[float]]:
        """
        Loads the training and validation loss history using the sentiment evaluator.

        :param history_file_path: The path to the history file.
        :returns: A tuple containing the training loss list and validation loss list.
        """
        return self._evaluator.load_training_history(history_file_path=history_file_path)

    @override
    def _run_evaluation_for_epoch(self, eval_config: SentimentEvaluationConfig) -> SentimentEvaluationResult:
        """
        Runs the evaluation for a single epoch using the sentiment evaluator.

        :param eval_config: The configuration object for the evaluation.
        :returns: The result object from the evaluation.
        """
        return self._evaluator.run(config=eval_config)

    def _build_evaluation_config(
        self, args: Namespace, original_config_data: dict[str, any], epoch_to_evaluate: int
    ) -> SentimentEvaluationConfig:
        """
        Builds the appropriate SentimentEvaluationConfig based on CLI arguments.

        This helper centralizes the logic for switching between "reproducibility"
        and "exploratory" evaluation modes.

        :param args: The namespace object from argparse.
        :param original_config_data: The loaded dictionary from the model's config.yaml.
        :param epoch_to_evaluate: The specific epoch to be evaluated.
        :returns: A fully constructed, flattened SentimentEvaluationConfig object.
        """
        # Exploratory Mode: A new dataset is specified via CLI.
        if args.dataset_name:
            self._logger.info(f"Building evaluation config for EXPLORATORY mode on dataset '{args.dataset_name}'.")
            return SentimentEvaluationConfig(
                model_id=args.model_id,
                model_epoch=epoch_to_evaluate,
                dataset_file_name=args.dataset_name,
                evaluate_on_full_dataset=True,
                # This is required by the config dataclass but not used for splitting in this mode.
                # It ensures consistency if other parts of the evaluator need it.
                vocabulary_size=original_config_data['vocabulary_size'],
                split_ratios=None,  # Explicitly None as we are not splitting
                random_state=None  # Explicitly None as we are not splitting
            )
        # Reproducibility Mode: No new dataset is specified.
        else:
            self._logger.info("Building evaluation config for REPRODUCIBILITY mode.")
            # noinspection PyTypeChecker
            return SentimentEvaluationConfig(
                model_id=args.model_id,
                model_epoch=epoch_to_evaluate,
                dataset_file_name=original_config_data['dataset_file_name'],
                evaluate_on_full_dataset=False,
                vocabulary_size=original_config_data['vocabulary_size'],
                split_ratios=tuple(original_config_data['split_ratios']),
                random_state=original_config_data['random_state']
            )

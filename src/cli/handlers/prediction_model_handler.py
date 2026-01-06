import random
from argparse import ArgumentParser, Namespace
from datetime import date, timedelta
from logging import Logger
from pathlib import Path
from typing import Final

from numpy.typing import NDArray
from typing_extensions import override

from src.cli.handlers.gradient_model_handler import RegressionModelHandler
from src.core.project_config import ProjectModelType, ProjectPaths
from src.data_handling.box_office import BoxOffice
from src.data_handling.dataset import Dataset
from src.data_handling.file_io import YamlFile
from src.data_handling.movie_collections import MovieData
from src.data_handling.movie_metadata import MovieMetadata
from src.models.base.evaluation import BaseEvaluationResult
from src.models.prediction.components.data_processor import (
    PredictionConfigDict,
    PredictionDataConfig,
    PredictionDataProcessor
)
from src.models.prediction.components.evaluator import (
    PredictionEvaluationConfig,
    PredictionEvaluationResult,
    PredictionEvaluator
)
from src.models.prediction.components.model_core import PredictionModelCore, PredictionPredictConfig
from src.models.prediction.pipelines.training_pipeline import PredictionPipelineConfig, PredictionTrainingPipeline
from src.utilities.plot import plot_multi_line_graph


class PredictionModelHandler(RegressionModelHandler[PredictionConfigDict]):
    """
    Handles CLI commands related to the box office prediction model.

    This class extends `RegressionModelHandler` to provide specific implementations
    for training, predicting with, and evaluating the box office prediction model.
    It manages the complete lifecycle for this specific model type.

    :cvar _EVALUATION_CACHE_FILE_NAME: The filename for the prediction model's evaluation cache.
    """
    _EVALUATION_CACHE_FILE_NAME: Final[str] = "prediction_evaluation_cache.yaml"

    @override
    def __init__(self, parser: ArgumentParser) -> None:
        """
        Initializes the PredictionModelHandler.

        :param parser: The argument parser instance.
        """
        super().__init__(
            parser=parser,
            model_type_name="Prediction",
            model_type=ProjectModelType.PREDICTION,
            evaluator=PredictionEvaluator()
        )

    @override
    def train(self, args: Namespace) -> None:
        """
        Orchestrates the box office prediction model training process.

        This method handles both new and continued training runs by preparing the
        configuration and launching the training pipeline.

        :param args: The namespace object from argparse, containing `model_id` and
                     other training-related parameters.
        :raises SystemExit: If there is a configuration error or the pipeline fails.
        """
        effective_config: PredictionConfigDict = self._prepare_training_config(args=args)

        try:
            artifacts_folder: Path = ProjectPaths.get_model_root_path(
                model_id=args.model_id, model_type=self._model_type
            )
            pipeline_config: PredictionPipelineConfig = PredictionPipelineConfig(**effective_config)
            data_processor: PredictionDataProcessor = PredictionDataProcessor(model_artifacts_path=artifacts_folder)
            model_core: PredictionModelCore = PredictionModelCore()
            pipeline: PredictionTrainingPipeline = PredictionTrainingPipeline(
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
        Makes a prediction using a trained model on a specific movie or random data.

        This method orchestrates the prediction process by loading the model and
        its artifacts, preparing the input data, running the prediction, and
        logging the final result.

        :param args: The namespace object containing command-line arguments,
                     expected to have 'model_id', 'epoch', and either 'movie_name' or 'random'.
        :raises SystemExit: If model components cannot be loaded, input data is not found,
                            or an error occurs during prediction.
        """
        self._logger.info(
            f"Executing: Predict with {self._model_type_name} model '{args.model_id}' (epoch: {args.epoch})."
        )

        # Load Model and Artifacts
        try:
            # Load master config to get training parameters
            config_path: Path = ProjectPaths.get_model_root_path(
                model_id=args.model_id, model_type=self._model_type
            ) / "config.yaml"
            if not config_path.exists():
                self._parser.error(f"Master config file 'config.yaml' not found for model_id '{args.model_id}'.")
            original_config_data: dict[str, any] = YamlFile(path=config_path).load_single_document()

            # Instantiate components
            artifacts_folder: Path = ProjectPaths.get_model_root_path(
                model_id=args.model_id, model_type=self._model_type
            )
            model_file_path: Path = artifacts_folder / f"{args.model_id}_{args.epoch:04d}.keras"

            data_processor: PredictionDataProcessor = PredictionDataProcessor(model_artifacts_path=artifacts_folder)
            model_core: PredictionModelCore = PredictionModelCore(model_path=model_file_path)

            if not data_processor.scaler:
                self._parser.error(f"Scaler artifact not found for model '{args.model_id}'. Cannot make predictions.")

        except (FileNotFoundError, ValueError) as e:
            self._parser.error(f"Failed to load model components: {e}")
        except Exception as e:
            self._parser.error(f"An unexpected error occurred while loading components: {e}")

        # Get Input Data (Movie or Random)
        try:
            input_data: MovieData
            if args.movie_name:
                self._logger.info(f"Mode: Predicting for movie '{args.movie_name}'.")
                dataset_name: str | None = original_config_data.get('dataset_name')
                if not dataset_name:
                    self._parser.error(f"Config for model '{args.model_id}' is missing 'dataset_name'.")

                dataset: Dataset = Dataset(name=dataset_name)
                target_movie_data: MovieData | None = next(
                    (m for m in dataset.movie_data if m.name == args.movie_name), None
                )

                if not target_movie_data:
                    self._parser.error(f"Movie '{args.movie_name}' not found in dataset '{dataset_name}'.")

                input_data = target_movie_data

            elif args.random:
                self._logger.info("Mode: Predicting with randomly generated data.")
                training_week_len: int | None = original_config_data.get('training_week_len')
                if not training_week_len:
                    self._parser.error(f"Config for model '{args.model_id}' is missing 'training_week_len'.")
                # Generate a bit more data than needed to ensure a valid sequence can be extracted
                input_data = self._generate_random_movie_data(weeks=training_week_len + 5)

            else:
                # This case should not be hit due to argparse mutual exclusion
                self._parser.error("Either --movie-name or --random must be specified.")

            # Process Input and Predict
            processing_config: PredictionDataConfig = PredictionDataConfig(
                training_week_len=original_config_data['training_week_len'],
                split_ratios=original_config_data['split_ratios'],  # Not used but required by dataclass
                random_state=original_config_data['random_state']  # Not used but required by dataclass
            )

            processed_input: NDArray[any] = data_processor.process_for_prediction(
                single_input=input_data, config=processing_config
            )

            pred_config: PredictionPredictConfig = PredictionPredictConfig(verbose=0)
            scaled_prediction: NDArray[any] = model_core.predict(data=processed_input, config=pred_config)

            # Inverse transform the prediction
            unscaled_prediction: float = data_processor.scaler.inverse_transform(scaled_prediction)[0][0]

            # Display Result
            self._logger.info("--- Prediction Result ---")
            self._logger.info(f"  Input: {input_data.name}")
            self._logger.info(f"  Predicted Box Office (Next Week): ${unscaled_prediction:,.0f} TWD")
            self._logger.info("-------------------------")

        except (ValueError, TypeError) as e:
            self._logger.error(f"Prediction failed due to a data or configuration error: {e}", exc_info=True)
            self._parser.error(f"Prediction failed: {e}")
        except Exception as e:
            self._logger.error(f"An unexpected error occurred during prediction: {e}", exc_info=True)
            self._parser.error("Prediction failed. Check logs for details.")

    @override
    def _get_default_config_filename(self) -> str:
        """
        Gets the filename for the prediction model's default configuration.

        :returns: The name of the default configuration file.
        """
        return "prediction_defaults.yaml"

    @override
    def _get_evaluation_cache_filename(self) -> str:
        """
        Gets the filename for the prediction model's evaluation cache.

        :returns: The name of the evaluation cache file.
        """
        return self._EVALUATION_CACHE_FILE_NAME

    @override
    def _run_evaluation_for_epoch(self, eval_config: PredictionEvaluationConfig) -> PredictionEvaluationResult:
        """
        Runs the evaluation for a single epoch using the prediction evaluator.

        :param eval_config: The configuration object for the evaluation.
        :returns: The result object from the evaluation.
        """
        return self._evaluator.run(config=eval_config)

    @override
    def _plot_specific_graphs(
        self, eval_results: list[BaseEvaluationResult], output_dir: Path, args: Namespace
    ) -> None:
        """
        Plots graphs specific to the prediction model.

        This calls the parent method to plot regression metrics (Test Loss) and then
        adds logic to plot classification metrics (Range F1-Score).
        """
        # First, call the parent class (RegressionModelHandler) method to plot Test Loss (MSE).
        super()._plot_specific_graphs(eval_results=eval_results, output_dir=output_dir, args=args)

        # Then, implement the F1-Score plotting logic specific to PredictionModel.
        if args.show_f1_score:
            self._logger.info("Plotting prediction-specific metric: Range F1-Score.")

            prediction_results: list[PredictionEvaluationResult] = [
                res for res in eval_results if isinstance(res, PredictionEvaluationResult)
            ]

            if not prediction_results:
                self._logger.warning("No prediction evaluation results found. Cannot plot F1-score.")
                return

            f1_scores_by_epoch: dict[int, float] = {
                res.model_epoch: res.range_classification_report['f1_score']
                for res in prediction_results
                if res.range_classification_report
            }

            if not f1_scores_by_epoch:
                self._logger.info("No Range F1-score data available to plot.")
                return

            epochs: list[int] = sorted(f1_scores_by_epoch.keys())
            f1_data: list[float] = [f1_scores_by_epoch[epoch] for epoch in epochs]

            plot_multi_line_graph(
                title=f"Range F1-Score on Test Set for {prediction_results[0].model_id}",
                save_path=output_dir / "range_f1_score_curve.png",
                x_data=epochs,
                y_datasets=[{"label": "Range F1-Score", "data": f1_data}],
                x_label="Epoch",
                y_label="F1-Score",
                y_formatter=None
            )

    @override
    def _display_specific_metrics(self, result: BaseEvaluationResult, args: Namespace, result_logger: Logger) -> None:
        """
        Displays metrics specific to the prediction model.

        This calls the parent method to display regression metrics (Test Loss) and
        then adds logic to display classification metrics (F1-Score, Confusion Matrix)
        using the provided format-less logger.

        :param result: The evaluation result object.
        :param args: The command-line arguments.
        :param result_logger: The logger instance configured for clean, format-less output.
        """
        # First, call the parent class (RegressionModelHandler) method to display Test Loss (MSE).
        super()._display_specific_metrics(result=result, args=args, result_logger=result_logger)

        # Then, implement the metric display logic specific to PredictionModel.
        if isinstance(result, PredictionEvaluationResult):
            if args.show_f1_score and result.range_classification_report:
                result_logger.info(f"  - Range F1-Score: {result.range_classification_report['f1_score']:.4f}")

            if args.show_confusion_matrix and result.range_classification_report:
                matrix_str: str = result.format_confusion_matrix_string(
                    matrix=result.range_classification_report['confusion_matrix'],
                    names=result.range_classification_report.get('target_names') or []
                )
                result_logger.info(f"  - Confusion Matrix (Range):\n{matrix_str}")

    @override
    def _build_evaluation_config(
        self, args: Namespace, original_config_data:PredictionConfigDict, epoch_to_evaluate: int
    ) -> PredictionEvaluationConfig:
        """
        Builds the evaluation configuration object for a single epoch.

        This method constructs a `PredictionEvaluationConfig` object based on the
        provided command-line arguments and the model's original training configuration.
        It distinguishes between two modes:
        - **Reproducibility Mode (default):** Uses the original dataset and split
          parameters from `config.yaml` to recreate the exact test set.
        - **Exploratory Mode (if `--dataset-name` is used):** Evaluates the model
          on a new, full dataset without splitting.

        :param args: The namespace object from argparse, containing evaluation flags.
        :param original_config_data: The loaded dictionary from the model's `config.yaml`.
        :param epoch_to_evaluate: The specific epoch to be evaluated.
        :returns: A `PredictionEvaluationConfig` object tailored for the evaluation run.
        """

        if (args.show_f1_score or args.show_confusion_matrix) and not args.classification_report:
            self._parser.error(
                "Arguments --show-f1-score and --show-confusion-matrix require "
                "--classification-report {range,trend} to be specified."
            )
        # Determine whether to perform calculations based on the new CLI parameter design.
        calculate_loss: bool = args.test_loss
        # As long as the user requests the report, F1 score, or confusion matrix,
        # classification metrics should be calculated.
        calculate_classification_metrics: bool = bool(
            args.classification_report) or args.show_f1_score or args.show_confusion_matrix
        # In exploratory mode, we evaluate on a new dataset.
        if args.dataset_name:
            self._logger.info(f"Building evaluation config for EXPLORATORY mode on dataset '{args.dataset_name}'.")
            return PredictionEvaluationConfig(
                model_id=args.model_id,
                model_epoch=epoch_to_evaluate,
                dataset_name=args.dataset_name,
                evaluate_on_full_dataset=True,
                training_week_len=original_config_data['training_week_len'],
                split_ratios=None,
                random_state=None,
                calculate_loss=calculate_loss,
                calculate_classification_metrics=calculate_classification_metrics,
                classification_method=args.classification_report,
                f1_average_method=args.f1_average_method or original_config_data.get('f1_average_method', 'macro'),
                box_office_ranges=tuple(args.box_office_ranges) if args.box_office_ranges else tuple(
                    original_config_data.get('box_office_ranges', ()))
            )
            # In reproducibility mode, we recreate the original test set.
        else:
            self._logger.info("Building evaluation config for REPRODUCIBILITY mode.")
            return PredictionEvaluationConfig(
                model_id=args.model_id,
                model_epoch=epoch_to_evaluate,
                dataset_name=original_config_data['dataset_name'],
                evaluate_on_full_dataset=False,
                training_week_len=original_config_data['training_week_len'],
                split_ratios=tuple(original_config_data['split_ratios']),
                random_state=original_config_data['random_state'],
                calculate_loss=calculate_loss,
                calculate_classification_metrics=calculate_classification_metrics,
                classification_method=args.classification_report,
                f1_average_method=args.f1_average_method or original_config_data.get('f1_average_method', 'macro'),
                box_office_ranges=tuple(args.box_office_ranges) if args.box_office_ranges else tuple(
                    original_config_data.get('box_office_ranges', ()))
            )

    def _generate_random_movie_data(self, weeks: int) -> MovieData:
        """
        Generates a MovieData object with random data for demonstration.

        :param weeks: The number of weeks of box office data to generate.
        :returns: A MovieData object populated with random data.
        """
        self._logger.info(f"Generating random movie data with {weeks} weeks of history.")

        # Create a dummy metadata object
        random_metadata: MovieMetadata = MovieMetadata(id=-1, name="Randomly Generated Movie")

        # Generate a list of weekly box office data
        box_office_history: list[BoxOffice] = []
        current_end_date: date = date.today()
        for _ in range(weeks):
            start_date: date = current_end_date - timedelta(days=6)
            random_revenue: int = random.randint(1_000_000, 50_000_000)
            box_office_entry: BoxOffice = BoxOffice(
                start_date=start_date,
                end_date=current_end_date,
                box_office=random_revenue
            )
            box_office_history.append(box_office_entry)
            current_end_date = start_date - timedelta(days=1)

        # The list is generated backwards, so reverse it to be chronological
        box_office_history.reverse()

        return MovieData(
            metadata=random_metadata,
            box_office=box_office_history,
            public_reviews=[],  # No need to generate random reviews
            expert_reviews=[]
        )

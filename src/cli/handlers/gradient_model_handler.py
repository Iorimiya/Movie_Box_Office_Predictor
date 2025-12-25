from abc import abstractmethod, ABC
from argparse import Namespace
from logging import Formatter, Handler, Logger, StreamHandler
from pathlib import Path

from typing_extensions import override

from src.cli.handlers.base_model_handler import BaseModelHandler
from src.core.logging_manager import HandlerSettings, LoggingManager, LogLevel
from src.core.project_config import ProjectPaths
from src.models.base.evaluation import (
    BaseEvaluationResult,
    ClassificationEvaluationResult,
    GradientBasedEvaluationResult,
    RegressionEvaluationResult
)
from src.utilities.plot import PlotDataset, plot_multi_line_graph


class GradientModelHandler(BaseModelHandler, ABC):
    """
    An abstract handler for models trained with gradient descent.

    This class is also abstract because it does not implement all methods from
    BaseModelHandler. It provides common logic for plotting and displaying
    metrics related to epoch-based training history.
    """

    @abstractmethod
    def _plot_specific_graphs(
            self,
            eval_results: list[BaseEvaluationResult],
            output_dir: Path,
            args: Namespace
    ) -> None:
        """
        Plots graphs for metrics specific to the concrete model type.

        Subclasses must implement this to handle plotting for metrics like
        F1-score, accuracy, etc.

        :param eval_results: The list of evaluation results for all epochs.
        :param output_dir: The directory to save the plot images.
        :param args: The command-line arguments.
        """
        pass

    def plot_graph(self, args: Namespace) -> None:
        """
        Generates and saves evaluation graphs for a model series.

        This implementation acts as a template method. It handles the plotting
        of common loss curves and delegates the plotting of model-specific
        graphs to the `_plot_specific_graphs` hook method.

        :param args: The namespace object from argparse.
        """
        try:
            # This method is inherited from BaseModelHandler
            eval_results: list[BaseEvaluationResult] = self._evaluate_all_epochs(
                model_id=args.model_id, args=args
            )
        except (FileNotFoundError, ValueError) as e:
            self._parser.error(str(e))
            return

        if not eval_results:
            self._logger.warning("No evaluation results were generated. Cannot create any plots.")
            return

        output_dir: Path = ProjectPaths.get_model_plots_path(
            model_id=args.model_id, model_type=self._model_type
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Plot Common Graphs (Loss)
        if args.training_loss or args.validation_loss:
            self._plot_loss_graph(eval_results=eval_results, output_dir=output_dir, args=args)

        # Delegate Specific Graphs to Subclass
        self._plot_specific_graphs(eval_results=eval_results, output_dir=output_dir, args=args)

    def _plot_loss_graph(
            self, eval_results: list[BaseEvaluationResult], output_dir: Path, args: Namespace
    ) -> None:
        """
        Plots the common loss curves for a model (Training and Validation).

        This helper method is responsible for plotting metrics that are common
        to all Keras-based training histories.

        :param eval_results: The aggregated evaluation results for the model.
        :param output_dir: The directory to save the plot image.
        :param args: The command-line arguments to check which losses to plot.
        """
        # Filter for results that have loss history
        gradient_results: list[GradientBasedEvaluationResult] = [
            res for res in eval_results if isinstance(res, GradientBasedEvaluationResult)
        ]
        if not gradient_results:
            self._logger.warning("No gradient-based evaluation results found. Cannot plot loss curves.")
            return

        # Assume history length is consistent, take from the first result
        first_result = gradient_results[0]
        history_epochs: list[int] = list(range(1, len(first_result.training_loss_history) + 1))
        datasets_to_plot: list[PlotDataset] = []

        if args.training_loss:
            datasets_to_plot.append({"label": "Training Loss", "data": first_result.training_loss_history})
        if args.validation_loss:
            datasets_to_plot.append({"label": "Validation Loss", "data": first_result.validation_loss_history})

        if not datasets_to_plot:
            self._logger.info("No loss types were selected for plotting.")
            return

        plot_multi_line_graph(
            title=f"Loss Curves for {first_result.model_id}",
            save_path=output_dir / "loss_curves.png",
            x_data=history_epochs,
            y_datasets=datasets_to_plot,
            x_label="Epoch",
            y_label="Loss",
            y_formatter='sci-notation'
        )

    def _display_metrics(self, result: BaseEvaluationResult, args: Namespace) -> None:
        """
        Displays evaluation metrics in a structured format.

        This template method prints a standard header and footer and delegates
        the display of all specific metrics to the `_display_specific_metrics`
        hook method.

        :param result: The evaluation result object.
        :param args: The command-line arguments to check which metrics to display.
        """
        manager: LoggingManager = LoggingManager()
        # Define a unique name for our temporary handler and logger
        result_handler_name: str = "temp_result_handler"
        result_logger_name: str = "result_display"

        try:
            # Setup: Create and configure a temporary, format-less handler
            # Create a formatter that only outputs the message
            result_formatter: Formatter = Formatter(fmt="%(message)s")
            stdout_handler: Handler = manager.get_handler('stdout')
            if not isinstance(stdout_handler, StreamHandler):
                self._logger.error(
                    "Critical: 'stdout' handler not found or is not a StreamHandler. Cannot produce clean output."
                )
                # Fallback to using the standard logger if setup fails
                self._display_specific_metrics(result=result, args=args, result_logger=self._logger)
                return

            # Create settings for a new handler that prints to stdout
            result_handler_settings: HandlerSettings = HandlerSettings(
                name=result_handler_name, level=LogLevel.INFO, output=stdout_handler.stream
            )
            # Add the handler to the manager and set its custom formatter
            result_handler = manager.add_handler(handler_settings=result_handler_settings)
            result_handler.setFormatter(result_formatter)

            # Get a logger and link the new handler to it
            result_logger: Logger = manager.get_logger(result_logger_name)
            manager.link_handler_to_logger(logger_name=result_logger_name, handler_name=result_handler_name)
            result_logger.propagate = False  # Prevent double-printing to the root logger

            # Usage: Use the new logger for clean output
            result_logger.info(f"--- Evaluation Metrics for Model '{result.model_id}' (Epoch {result.model_epoch}) ---")

            # Delegate to subclass for specific metrics, passing the clean logger
            self._display_specific_metrics(result=result, args=args, result_logger=result_logger)

            result_logger.info("----------------------------------------------------")

        finally:
            # Teardown: Always clean up the temporary handler and logger
            if manager.get_handler(result_handler_name):
                self._logger.debug(f"Cleaning up temporary handler: {result_handler_name}")
                manager.remove_handler(name=result_handler_name)
            if manager.get_logger(result_logger_name):
                self._logger.debug(f"Cleaning up temporary logger: {result_logger_name}")
                manager.remove_logger(name=result_logger_name)


class RegressionModelHandler(GradientModelHandler, ABC):
    """
    An abstract handler for regression models trained with gradient descent.

    This class extends GradientModelHandler to implement common logic for
    regression-specific metrics, such as plotting and displaying the test loss (MSE).
    It remains abstract as it does not implement core methods like `train` or `predict`.
    """

    @override
    def _plot_specific_graphs(
            self,
            eval_results: list[BaseEvaluationResult],
            output_dir: Path,
            args: Namespace
    ) -> None:
        """
        Plots graphs specific to regression models, primarily the test loss curve.

        This method implements the abstract hook from the parent class to handle
        regression-specific plotting tasks.

        :param eval_results: The list of evaluation results for all epochs.
        :param output_dir: The directory to save the plot images.
        :param args: The command-line arguments.
        """
        # Plot Test Loss (MSE) if requested
        if args.test_loss:
            self._logger.info("Plotting regression-specific metric: Test Loss (MSE).")
            regression_results: list[RegressionEvaluationResult] = [
                res for res in eval_results if isinstance(res, RegressionEvaluationResult)
            ]
            if not regression_results:
                self._logger.warning("No regression evaluation results found. Cannot plot test loss.")
                return

            test_losses: dict[int, float] = {
                res.model_epoch: res.regression_report['mse']
                for res in regression_results
                if res.regression_report
            }

            epochs_with_test_loss = sorted(test_losses.keys())
            loss_data = [test_losses[epoch] for epoch in epochs_with_test_loss]

            if not loss_data:
                self._logger.info("No test loss data available to plot.")
                return

            plot_multi_line_graph(
                title=f"Test Loss (MSE) for {regression_results[0].model_id}",
                save_path=output_dir / "test_loss_curve.png",
                x_data=epochs_with_test_loss,
                y_datasets=[{"label": "Test Loss (MSE)", "data": loss_data}],
                x_label="Epoch",
                y_label="Mean Squared Error",
                y_formatter='sci-notation'
            )

    @override
    def _display_specific_metrics(self, result: BaseEvaluationResult, args: Namespace, result_logger: Logger) -> None:
        """
        Displays metrics specific to regression models, such as the test loss (MSE).

        :param result: The evaluation result object.
        :param args: The command-line arguments.
        :param result_logger: The logger instance configured for clean, format-less output.
        """
        # Display Test Loss (MSE) if requested
        if args.test_loss and isinstance(result, RegressionEvaluationResult) and result.regression_report:
            result_logger.info(f"  - Test Loss (MSE): {result.regression_report['mse']:.6f}")


class ClassificationModelHandler(GradientModelHandler, ABC):
    """
    An abstract handler for classification models trained with gradient descent.

    This class extends GradientModelHandler to implement common logic for
    classification-specific metrics, such as plotting F1-scores and displaying
    confusion matrices.
    """

    @override
    def _plot_specific_graphs(
            self,
            eval_results: list[BaseEvaluationResult],
            output_dir: Path,
            args: Namespace
    ) -> None:
        """
        Plots graphs specific to classification models, such as the F1-score curve.

        This method implements the abstract hook from the parent class to handle
        classification-specific plotting tasks.

        :param eval_results: The list of evaluation results for all epochs.
        :param output_dir: The directory to save the plot images.
        :param args: The command-line arguments.
        """
        if args.show_f1_score:
            self._logger.info("Plotting classification-specific metric: F1-Score.")

            classification_results: list[ClassificationEvaluationResult] = [
                res for res in eval_results if isinstance(res, ClassificationEvaluationResult)
            ]

            if not classification_results:
                self._logger.warning("No classification evaluation results found. Cannot plot F1-score.")
                return

            f1_scores_by_epoch: dict[int, float] = {
                res.model_epoch: res.classification_report['f1_score']
                for res in classification_results
                if res.classification_report
            }

            if not f1_scores_by_epoch:
                self._logger.info("No F1-score data available to plot.")
                return

            epochs: list[int] = sorted(f1_scores_by_epoch.keys())
            f1_data: list[float] = [f1_scores_by_epoch[epoch] for epoch in epochs]

            plot_multi_line_graph(
                title=f"F1-Score on Test Set for {classification_results[0].model_id}",
                save_path=output_dir / "f1_score_curve.png",
                x_data=epochs,
                y_datasets=[{"label": "Test F1-Score", "data": f1_data}],
                x_label="Epoch",
                y_label="F1-Score",
                y_formatter=None
            )

    @override
    def _display_specific_metrics(self, result: BaseEvaluationResult, args: Namespace, result_logger: Logger) -> None:
        """
        Displays metrics specific to classification models.

        :param result: The evaluation result object.
        :param args: The command-line arguments.
        :param result_logger: The logger instance configured for clean, format-less output.
        """

        if isinstance(result, ClassificationEvaluationResult) and result.classification_report:
            if args.show_f1_score:
                result_logger.info(f"  - F1-Score: {result.classification_report['f1_score']:.4f}")

            if args.show_confusion_matrix:
                matrix_str: str = result.format_confusion_matrix_string(
                    matrix=result.classification_report['confusion_matrix'],
                    names=result.classification_report.get('target_names') or []
                )
                result_logger.info(f"  - Confusion Matrix:\n{matrix_str}")

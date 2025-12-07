from typing import Optional

from numpy import array
from numpy.typing import NDArray
from typing_extensions import override

from src.models.base.keras_setup import keras_base
from src.utilities.metrics import BaseClassificationMetrics

Callback = keras_base.callbacks.Callback


# noinspection PyOverrides
class F1ScoreHistory(Callback):
    """
    A Keras Callback to calculate and record classification metrics on validation data at the end of each epoch.

    This callback uses a provided metrics calculator (an instance of a
    BaseClassificationMetrics subclass) to perform the evaluation. This
    decouples the callback from the specific metric calculation logic.

    :ivar validation_data: A tuple (x_val, y_val) containing the validation data.
    :ivar metrics_calculator: An instance of a class that inherits from
                              BaseClassificationMetrics, responsible for all
                              metric calculations.
    :ivar f1_scores: A list that stores the computed F1 score for each epoch.
    """
    validation_data: tuple[NDArray[any], NDArray[any]]
    metrics_calculator: BaseClassificationMetrics
    f1_scores: list[float]

    @override
    def __init__(
        self,
        *,
        validation_data: tuple[NDArray[any], NDArray[any]],
        metrics_calculator: BaseClassificationMetrics
    ) -> None:
        """
        Initializes the F1ScoreHistory callback.

        :param validation_data: A tuple (x_val, y_val) to be used for evaluation.
        :param metrics_calculator: A pre-configured instance of a metrics
                                   calculator (e.g., RegressionToClassificationMetrics).
        """
        super().__init__()
        self.validation_data: tuple[NDArray[any], NDArray[any]] = validation_data
        self.metrics_calculator: BaseClassificationMetrics = metrics_calculator
        self.f1_scores: list[float] = []

    @override
    def on_epoch_end(self, epoch: int, logs: Optional[dict[str, any]] = None) -> None:
        """
        Called at the end of an epoch to compute and store the F1 score.

        This method gets raw predictions from the model, delegates the metric
        calculation to the `metrics_calculator`, and logs the resulting F1 score.

        :param epoch: The index of the current epoch.
        :param logs: Metric results for this training epoch, and for the validation epoch.
        """
        x_val, y_val_scaled = self.validation_data
        y_pred_scaled: NDArray[any] = self.model.predict(x=x_val, verbose=0)

        # The metrics framework expects NumPy arrays.
        y_true_np: NDArray[any] = array(y_val_scaled)
        y_pred_np: NDArray[any] = array(y_pred_scaled)

        # Delegate the entire calculation process to the metrics calculator.
        report: dict[str, any] = self.metrics_calculator.generate_report(
            y_true=y_true_np,
            y_pred=y_pred_np
        )

        # Extract the F1 score and store it.
        score: float = report.get('f1_score', 0.0)
        self.f1_scores.append(score)

        # Optionally, add it to the Keras logs so it's printed and can be monitored.
        if logs is not None:
            logs['val_f1_score'] = score

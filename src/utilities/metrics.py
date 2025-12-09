from abc import ABC, abstractmethod
from typing import Callable, Optional

from numpy import floating, int_, issubdtype, vectorize
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, classification_report, f1_score
from typing_extensions import override


class BaseClassificationMetrics(ABC):
    """
    An abstract base class for calculating and reporting classification metrics.

    This class defines a template method, `generate_report`, which provides a
    standardized workflow for evaluating classification performance. It relies
    on subclasses to implement the `_transform_to_labels` method, which
    converts raw model outputs into discrete class labels suitable for
    metric calculation.

    :ivar target_names: Optional list of display names for the target classes.
    :ivar f1_average_method: The averaging method for F1 score calculation
                             (e.g., 'binary', 'macro', 'weighted').
    """
    target_names: Optional[list[str]]
    f1_average_method: str

    def __init__(
        self,
        *,
        target_names: Optional[list[str]] = None,
        f1_average_method: str = 'macro'
    ) -> None:
        """
        Initializes the BaseClassificationMetrics.

        :param target_names: Optional display names for the classes in the report.
        :param f1_average_method: The averaging strategy for the F1 score.
        """
        self.target_names: Optional[list[str]] = target_names
        self.f1_average_method: str = f1_average_method

    @abstractmethod
    def _transform_to_labels(
        self,
        *,
        y_true: NDArray[any],
        y_pred: NDArray[any]
    ) -> tuple[NDArray[int], NDArray[int]]:
        """
        Transforms raw true values and predictions into discrete integer labels.

        This is a hook method that must be implemented by all concrete subclasses
        to provide the specific logic for converting their input data format
        (e.g., continuous values, probabilities) into class labels.

        :param y_true: The ground-truth values.
        :param y_pred: The raw predicted values from a model.
        :return: A tuple containing two NumPy arrays: (true_labels, predicted_labels).
        """
        pass

    def generate_report(
        self,
        *,
        y_true: NDArray[any],
        y_pred: NDArray[any]
    ) -> dict[str, any]:
        """
        Generates a comprehensive classification metrics report.

        This template method orchestrates the evaluation by first transforming
        the inputs to labels via `_transform_to_labels`, and then computes
        various metrics like accuracy and a detailed classification report.

        :param y_true: The ground-truth values.
        :param y_pred: The raw predicted values from a model.
        :return: A dictionary containing the calculated metrics, including overall
                 accuracy, F1 score, a structured report dictionary, and a
                 formatted report string.
        """
        # Delegate the transformation step to the concrete subclass.
        true_labels: NDArray[int_]
        predicted_labels: NDArray[int_]
        true_labels, predicted_labels = self._transform_to_labels(y_true=y_true, y_pred=y_pred)

        # Calculate standard classification metrics using the transformed labels.
        accuracy: float = accuracy_score(y_true=true_labels, y_pred=predicted_labels)

        # Calculate the overall F1 score based on the specified averaging method.
        overall_f1: float = f1_score(
            y_true=true_labels, y_pred=predicted_labels, average=self.f1_average_method, zero_division=0
        )

        # Generate both a dictionary and a string version of the detailed report.
        report_dict: dict[str, any] = classification_report(
            y_true=true_labels,
            y_pred=predicted_labels,
            target_names=self.target_names,
            output_dict=True,
            zero_division=0
        )

        report_string: str = classification_report(
            y_true=true_labels,
            y_pred=predicted_labels,
            target_names=self.target_names,
            output_dict=False,
            zero_division=0
        )

        # Compile and return all results in a structured dictionary.
        return {
            'accuracy': accuracy,
            'f1_score': overall_f1,
            'report_dict': report_dict,
            'report_string': report_string
        }


class BinaryClassificationMetrics(BaseClassificationMetrics):
    """
    A concrete metrics calculator for standard binary classification tasks.

    This class assumes that the true values (`y_true`) are already binary labels
    (0 or 1) and the predicted values (`y_pred`) are probabilities that can be
    converted to labels using a 0.5 threshold. It can also handle cases where
    predictions are already provided as integer labels.

    :ivar threshold: The probability threshold to classify a prediction as positive.
    """
    threshold: float

    @override
    def __init__(
        self,
        *,
        target_names: Optional[list[str]] = None,
        f1_average_method: str = 'binary',
        threshold: float = 0.5
    ) -> None:
        """
        Initializes the BinaryClassificationMetrics.

        :param target_names: Optional display names for the classes. Defaults to ['Negative', 'Positive'].
        :param f1_average_method: The averaging strategy, defaulting to 'binary'.
        :param threshold: The cutoff for classifying probabilities as the positive class.
        """
        # If no target names are provided, use a sensible default for binary classification.
        final_target_names: Optional[list[str]] = target_names if target_names is not None else ['Negative', 'Positive']

        super().__init__(target_names=final_target_names, f1_average_method=f1_average_method)
        self.threshold: float = threshold

    @override
    def _transform_to_labels(
        self,
        *,
        y_true: NDArray[any],
        y_pred: NDArray[any]
    ) -> tuple[NDArray[int_], NDArray[int_]]:
        """
        Overrides the base method to convert probability predictions to binary labels.

        This implementation checks if the predictions are floating-point numbers (probabilities)
        or integers (pre-classified labels) and processes them accordingly.

        :param y_true: The ground-truth binary labels (0s and 1s).
        :param y_pred: The predicted values, which can be probabilities (float) or labels (int).
        :return: A tuple of (true_labels, predicted_labels) as integer arrays.
        """
        true_labels: NDArray[int_] = y_true.astype(int_)

        # Check if predictions are probabilities (float) or already labels (int).
        if issubdtype(y_pred.dtype, floating):
            # If they are floats, apply the threshold to convert to binary labels.
            predicted_labels: NDArray[int_] = (y_pred > self.threshold).astype(int_)
        else:
            # If they are already integers, use them directly.
            predicted_labels: NDArray[int_] = y_pred.astype(int_)

        return true_labels, predicted_labels


class RegressionToClassificationMetrics(BaseClassificationMetrics):
    """
    A concrete metrics calculator for evaluating a regression model on a
    classification basis.

    This class transforms continuous true values and predictions into discrete
    class labels using a provided function before calculating classification metrics.
    It is ideal for tasks like evaluating box office predictions against predefined
    revenue ranges.

    :ivar value_to_label_fn: The function used to convert a continuous value to a discrete label.
    """
    value_to_label_fn: Callable[[float], int]

    @override
    def __init__(
        self,
        *,
        value_to_label_fn: Callable[[float], int],
        target_names: Optional[list[str]] = None,
        f1_average_method: str = 'macro'
    ) -> None:
        """
        Initializes the RegressionToClassificationMetrics.

        :param value_to_label_fn: A function that takes a continuous float value
                                  and returns a discrete integer label.
        :param target_names: Optional display names for the classified ranges.
        :param f1_average_method: The averaging strategy, defaulting to 'macro' for multi-class.
        """
        super().__init__(target_names=target_names, f1_average_method=f1_average_method)
        self.value_to_label_fn: Callable[[float], int] = value_to_label_fn

    @override
    def _transform_to_labels(
        self,
        *,
        y_true: NDArray[any],
        y_pred: NDArray[any]
    ) -> tuple[NDArray[int_], NDArray[int_]]:
        """
        Overrides the base method to convert continuous regression values to class labels.

        It applies the `value_to_label_fn` to each element in both the true
        and predicted value arrays.

        :param y_true: The ground-truth continuous values.
        :param y_pred: The predicted continuous values from the model.
        :return: A tuple of (true_labels, predicted_labels) as integer arrays.
        """
        # Vectorize the provided Python function so it can be applied to NumPy arrays efficiently.
        vectorized_transform: Callable[[NDArray[any]], NDArray[int_]] = vectorize(self.value_to_label_fn)

        # Apply the vectorized function to both true and predicted values.
        true_labels: NDArray[int_] = vectorized_transform(y_true)
        predicted_labels: NDArray[int_] = vectorized_transform(y_pred)

        return true_labels, predicted_labels


def classification_report_to_string(true_labels: list[int], predicted_labels: list[int], target_names=None) -> None:
    if target_names is None:
        target_names = ['Negative (0)', 'Positive (1)']
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Overall Accuracy: {accuracy:.4f}\n")
    print(classification_report(true_labels, predicted_labels, target_names=target_names))

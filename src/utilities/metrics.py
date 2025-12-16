from abc import ABC, abstractmethod
from typing import Callable, Optional, TypedDict

from numpy import floating, int_, issubdtype, vectorize
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, mean_squared_error, \
    mean_absolute_error, r2_score
from typing_extensions import override


class RegressionReportDict(TypedDict):
    """
    A structured dictionary for the results of a single regression evaluation.

    :ivar mse: The Mean Squared Error value.
    :ivar mae: The Mean Absolute Error value.
    :ivar r2_score: The R-squared score.
    """
    mse: float
    mae: float
    r2_score: float


class ClassificationReportDict(TypedDict):
    """
    A structured dictionary for the results of a single classification evaluation.

    This is the standard output contract for BaseClassificationMetricsCalculator.generate_report.

    :ivar accuracy: The overall accuracy score.
    :ivar f1_score: The overall F1-score, calculated using the specified average method.
    :ivar confusion_matrix: The confusion matrix.
    :ivar report_dict: A structured dictionary version of the classification report.
    :ivar report_string: A formatted string version of the classification report.
    :ivar target_names: A list of the class names used in the report.
    """
    accuracy: float
    f1_score: float
    confusion_matrix: NDArray[int_]
    report_dict: dict[str, any]
    report_string: str
    target_names: Optional[list[str]]


class RegressionMetricsCalculator:
    """
    A concrete metrics calculator for standard regression tasks.
    """

    @staticmethod
    def generate_report(
        *,
        y_true: NDArray[any],
        y_pred: NDArray[any]
    ) -> RegressionReportDict:
        """
        Generates a standard regression metrics report.

        :param y_true: The ground-truth target values.
        :param y_pred: The estimated target values.
        :return: A dictionary containing MSE, MAE, and R2 score.
        """
        mse_val: float = mean_squared_error(y_true=y_true, y_pred=y_pred)
        mae_val: float = mean_absolute_error(y_true=y_true, y_pred=y_pred)
        r2_val: float = r2_score(y_true=y_true, y_pred=y_pred)

        return RegressionReportDict(mse=mse_val, mae=mae_val, r2_score=r2_val)


class ClassificationMetricsCalculator(ABC):
    """
    An abstract base class for calculating and reporting classification metrics.

    This class defines a template method, `generate_report`, which provides a
    standardized workflow for evaluating classification performance. It relies
    on subclasses to implement the `_transform_to_labels` method, which
    converts raw model outputs into discrete class labels suitable for
    metric calculation.

    :ivar label_map: An optional dictionary mapping integer labels to human-readable string names.
    :ivar f1_average_method: The averaging method for F1 score calculation.
    """
    label_map: Optional[dict[int, str]]
    f1_average_method: str

    def __init__(
        self,
        *,
        label_map: Optional[dict[int, str]] = None,
        f1_average_method: str = 'macro'
    ) -> None:
        """
        Initializes the ClassificationMetricsCalculator.

        :param label_map: An optional dictionary mapping integer labels to their string representations.
        :param f1_average_method: The averaging strategy for the F1 score.
        """
        self.label_map: Optional[dict[int, str]] = label_map
        self.f1_average_method: str = f1_average_method

    @abstractmethod
    def _transform_to_labels(
        self,
        *,
        y_true: NDArray[any],
        y_pred: NDArray[any]
    ) -> tuple[NDArray[int_], NDArray[int_]]:
        """
        Transforms raw true values and predictions into discrete integer labels.

        This is a hook method that must be implemented by all concrete subclasses
        to provide the specific logic for converting their input data format
        (e.g., continuous values, probabilities) into class labels.

        :param y_true: The ground-truth values.
        :param y_pred: The raw predicted values from a model.
        :return: A tuple containing two arrays: (true_labels, predicted_labels).
        """
        pass

    def generate_report(
        self,
        *,
        y_true: NDArray[any],
        y_pred: NDArray[any]
    ) -> ClassificationReportDict:
        """
        Generates a comprehensive classification metrics report.

        This template method orchestrates the evaluation by first transforming
        the inputs to labels via `_transform_to_labels`, and then computes
        various metrics like accuracy, a confusion matrix, and a detailed
        classification report. It internally derives the `labels` and
        `target_names` for sklearn functions from the `label_map`.

        :param y_true: The ground-truth values.
        :param y_pred: The raw predicted values from a model.
        :return: A dictionary containing the calculated metrics, conforming to the
                 `ClassificationReportDict` structure.
        """
        # Delegate the transformation step to the concrete subclass.
        true_labels: NDArray[int_]
        predicted_labels: NDArray[int_]
        true_labels, predicted_labels = self._transform_to_labels(
            y_true=y_true,
            y_pred=y_pred
        )

        possible_labels: Optional[list[int]] = None
        target_names_for_report: Optional[list[str]] = None
        if self.label_map:
            sorted_items: list[tuple[int, str]] = sorted(self.label_map.items())
            possible_labels: list[int] = [item[0] for item in sorted_items]
            target_names_for_report: list[str] = [item[1] for item in sorted_items]

        # Calculate standard classification metrics using the transformed labels.
        accuracy: float = accuracy_score(
            y_true=true_labels,
            y_pred=predicted_labels
        )

        overall_f1: float = f1_score(
            y_true=true_labels,
            y_pred=predicted_labels,
            average=self.f1_average_method,
            zero_division=0,
            labels=possible_labels
        )

        conf_matrix: NDArray[int_] = confusion_matrix(
            y_true=true_labels,
            y_pred=predicted_labels,
            labels=possible_labels
        )

        # Generate both a dictionary and a string version of the detailed report.
        report_dict: dict[str, any] = classification_report(
            y_true=true_labels,
            y_pred=predicted_labels,
            target_names=target_names_for_report,
            output_dict=True,
            zero_division=0,
            labels=possible_labels
        )

        report_string: str = classification_report(
            y_true=true_labels,
            y_pred=predicted_labels,
            target_names=target_names_for_report,
            output_dict=False,
            zero_division=0,
            labels=possible_labels
        )

        # Compile and return all results in a structured dictionary.
        return ClassificationReportDict(
            accuracy=accuracy,
            f1_score=overall_f1,
            confusion_matrix=conf_matrix,
            report_dict=report_dict,
            report_string=report_string,
            target_names=target_names_for_report
        )


class BinaryClassificationMetricsCalculator(ClassificationMetricsCalculator):
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
        label_map: Optional[dict[int, str]] = None,
        f1_average_method: str = 'binary',
        threshold: float = 0.5
    ) -> None:
        """
        Initializes the BinaryClassificationMetricsCalculator.

        :param label_map: An optional dictionary mapping labels {0: 'Neg', 1: 'Pos'}.
        :param f1_average_method: The averaging strategy, defaulting to 'binary'.
        :param threshold: The cutoff for classifying probabilities as the positive class.
        """
        # If no label map is provided, use a sensible default for binary classification.
        final_label_map: Optional[dict[int, str]] = label_map if label_map is not None else {0: 'Negative',
                                                                                             1: 'Positive'}

        super().__init__(
            label_map=final_label_map,
            f1_average_method=f1_average_method
        )
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

        if issubdtype(y_pred.dtype, floating):
            predicted_labels: NDArray[int_] = (y_pred > self.threshold).astype(int_)
        else:
            predicted_labels: NDArray[int_] = y_pred.astype(int_)

        return true_labels, predicted_labels


class PointwiseClassificationMetricsCalculator(ClassificationMetricsCalculator):
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
        label_map: Optional[dict[int, str]] = None,
        f1_average_method: str = 'macro'
    ) -> None:
        """
        Initializes the RegressionToClassificationMetricsCalculator.

        :param value_to_label_fn: A function that takes a continuous float value
                                  and returns a discrete integer label.
        :param label_map: An optional dictionary mapping the integer labels to display names.
        :param f1_average_method: The averaging strategy, defaulting to 'macro' for multi-class.
        """
        super().__init__(
            label_map=label_map,
            f1_average_method=f1_average_method
        )
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
        vectorized_transform: Callable[[NDArray[any]], NDArray[int_]] = vectorize(self.value_to_label_fn)

        true_labels: NDArray[int_] = vectorized_transform(y_true)
        predicted_labels: NDArray[int_] = vectorized_transform(y_pred)

        return true_labels, predicted_labels


class PairwiseClassificationMetricsCalculator(ClassificationMetricsCalculator):
    """
    A metrics calculator for pairwise regression-to-classification evaluation.

    This class is designed for scenarios where the classification label depends
    on a pair of values: a primary value (from `y_true` or `y_pred`) and a
    corresponding reference value. It is ideal for context-dependent tasks
    like trend analysis.

    :ivar value_pair_to_label_fn: A function that takes a value and its reference, returning a label.
    :ivar reference_values: The array of reference values to compare against.
    """
    value_pair_to_label_fn: Callable[[float, float], int]
    reference_values: NDArray[any]

    @override
    def __init__(
        self,
        *,
        value_pair_to_label_fn: Callable[[float, float], int],
        reference_values: NDArray[any],
        label_map: Optional[dict[int, str]] = None,
        f1_average_method: str = 'binary'
    ) -> None:
        """
        Initializes the PairwiseClassificationMetricsCalculator.

        :param value_pair_to_label_fn: A function that takes `(value, reference_value)`
                                       and returns a discrete integer label.
        :param reference_values: A NumPy array of reference values, which must have the
                                 same length as the `y_true` and `y_pred` arrays that
                                 will be passed to `generate_report`.
        :param label_map: Optional display names for the classes (e.g., {0: 'Decrease', 1: 'Increase'}).
        :param f1_average_method: The averaging strategy, defaulting to 'binary' for such tasks.
        """
        super().__init__(
            label_map=label_map,
            f1_average_method=f1_average_method
        )
        self.value_pair_to_label_fn: Callable[[float, float], int] = value_pair_to_label_fn
        self.reference_values: NDArray[any] = reference_values

    @override
    def _transform_to_labels(
        self,
        *,
        y_true: NDArray[any],
        y_pred: NDArray[any]
    ) -> tuple[NDArray[int_], NDArray[int_]]:
        """
        Overrides the base method to convert values to labels based on pairwise comparison.

        It applies the `value_pair_to_label_fn` to each element pair from the
        true/predicted value arrays and the `reference_values` array.

        :param y_true: The ground-truth continuous values.
        :param y_pred: The predicted continuous values from the model.
        :return: A tuple of (true_labels, predicted_labels) as integer arrays.
        :raises ValueError: If the length of `reference_values` does not match `y_true`.
        """
        if len(y_true) != len(self.reference_values):
            raise ValueError(
                f"Length of `y_true` ({len(y_true)}) must match the length of "
                f"`reference_values` ({len(self.reference_values)}) provided during initialization."
            )

        # Vectorize the provided Python function so it can be applied to NumPy arrays efficiently.
        # It now takes two arrays as input.
        vectorized_transform: Callable[
            [NDArray[any], NDArray[any]], NDArray[int_]
        ] = vectorize(self.value_pair_to_label_fn)

        # Apply the vectorized function to pairs of (value, reference_value).
        true_labels: NDArray[int_] = vectorized_transform(y_true, self.reference_values)
        predicted_labels: NDArray[int_] = vectorized_transform(y_pred, self.reference_values)

        return true_labels, predicted_labels

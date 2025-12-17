from numpy import int_
from numpy.typing import NDArray


class ClassificationSummaryMixin:
    """
    A mixin class that provides common summary generation behaviors for
    any evaluation result that contains classification reports.
    """

    @staticmethod
    def format_confusion_matrix_string(matrix: NDArray[int_], names: list[str]) -> str:
        """
        Formats a confusion matrix into a human-readable string for logging.

        This is a static utility method, decoupled from any instance state,
        making it a perfect candidate for a mixin or a standalone utility function.

        :param matrix: The confusion matrix as a NumPy array.
        :param names: A list of string names for the classes, corresponding to the matrix axes.
        :return: A formatted, multi-line string representation of the confusion matrix.
        """
        if matrix.size == 0 or not names:
            return "  [Confusion Matrix is empty or has no labels]"

        header_col_width: int = max(len(name) for name in names)
        cell_width: int = max((len(str(cell)) for cell in matrix.flatten()), default=0)
        cell_width = max(cell_width, max((len(name) for name in names), default=0)) + 2

        header: str = f"{'':<{header_col_width}} |" + "".join([f"{name:^{cell_width}}" for name in names])
        separator: str = '-' * (header_col_width + 1) + '-' * (cell_width * len(names))

        lines: list[str] = [
            f"{'True / Pred':<{header_col_width}} | {'Predicted Labels':^{cell_width * len(names) - 1}}",
            header,
            separator
        ]
        for i, name in enumerate(names):
            row_str: str = f"{name:<{header_col_width}} |"
            for j in range(len(names)):
                row_str += f"{matrix[i, j]:^{cell_width}}"
            lines.append(row_str)

        return "\n".join(["  " + line for line in lines])

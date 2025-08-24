from logging import Logger
from typing import Generic, Optional, TypeVar

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from src.core.compat import TypedDict
from src.core.logging_manager import LoggingManager

X_Type = TypeVar('X_Type', bound=NDArray[any])
Y_Type = TypeVar('Y_Type', bound=NDArray[any])


class SplitDataset(TypedDict, Generic[X_Type, Y_Type]):
    """
    A generic TypedDict representing a dataset split into training, validation, and test sets.

    :ivar x_train: Training data (features).
    :ivar y_train: Training data (labels).
    :ivar x_val: Validation data (features).
    :ivar y_val: Validation data (labels).
    :ivar x_test: Test data (features).
    :ivar y_test: Test data (labels).
    """
    x_train: X_Type
    y_train: Y_Type
    x_val: X_Type
    y_val: Y_Type
    x_test: X_Type
    y_test: Y_Type


class DatasetSplitter(Generic[X_Type, Y_Type]):
    """
    A utility class for splitting datasets into training, validation, and test sets.

    This class provides a standardized, robust, and configurable way to split
    feature (x) and label (y) data, handling various edge cases and supporting
    both shuffled and sequential splitting strategies.

    :ivar logger: A logger instance for logging splitting operations.
    """

    def __init__(self, logger: Optional[Logger] = None) -> None:
        """
        Initializes the DatasetSplitter.

        :param logger: An optional logger instance. If not provided, a new
                       logger will be acquired.
        """
        self.logger: Logger = logger if logger else LoggingManager().get_logger('')

    # noinspection PyTypeChecker
    def split(
        self,
        x_data: X_Type,
        y_data: Y_Type,
        split_ratios: tuple[int, int, int],
        random_state: Optional[int],
        shuffle: bool = True
    ) -> SplitDataset[X_Type, Y_Type]:
        """
        Splits the dataset into training, validation, and testing sets based on specified ratios.

        :param x_data: The full feature dataset.
        :param y_data: The full label dataset.
        :param split_ratios: A tuple of three integers (train, val, test) representing the
                             proportions of the split. For example, (8, 1, 1) for an 80/10/10 split.
        :param random_state: The seed for the random number generator, used for reproducibility.
                             If `shuffle` is False, this has no effect.
        :param shuffle: If True, the data will be shuffled before splitting. If False, the split
                        will be sequential. For time-series data, this should typically be False.
                        Stratification is only applied if `shuffle` is True.
        :returns: A TypedDict containing the data splits.
        :raises ValueError: If the sum of `split_ratios` is zero.
        """
        self.logger.info(
            f"Splitting data with ratios {split_ratios}, random_state={random_state}, shuffle={shuffle}."
        )

        train_ratio, val_ratio, test_ratio = split_ratios
        total_ratio: int = sum(split_ratios)

        if total_ratio == 0:
            raise ValueError("The sum of split_ratios cannot be zero.")

        if len(x_data) == 0:
            self.logger.warning("Input data for splitting is empty. Returning empty splits.")
            # Determine shape for empty array
            empty_x, empty_y = self._create_empty_arrays(x_data, y_data)
            return SplitDataset(
                x_train=empty_x, y_train=empty_y,
                x_val=empty_x, y_val=empty_y,
                x_test=empty_x, y_test=empty_y
            )

        # If not shuffling, perform a sequential split
        if not shuffle:
            num_samples: int = len(x_data)
            train_end_idx: int = int(num_samples * train_ratio / total_ratio)
            val_end_idx: int = int(num_samples * (train_ratio + val_ratio) / total_ratio)

            x_train, y_train = x_data[:train_end_idx], y_data[:train_end_idx]
            x_val, y_val = x_data[train_end_idx:val_end_idx], y_data[train_end_idx:val_end_idx]
            x_test, y_test = x_data[val_end_idx:], y_data[val_end_idx:]

        # If shuffling, use the robust train_test_split method
        else:
            # First split: separate test set
            train_val_ratio: int = train_ratio + val_ratio
            if train_val_ratio == 0:
                x_train_val, y_train_val = self._create_empty_arrays(x_data, y_data)
                x_test, y_test = x_data, y_data
            else:
                can_stratify: bool = self._can_stratify(y_data)
                x_train_val, x_test, y_train_val, y_test = train_test_split(
                    x_data, y_data,
                    test_size=(test_ratio / total_ratio),
                    random_state=random_state,
                    shuffle=True,  # Always true here
                    stratify=y_data if can_stratify else None
                )

            # Second split: separate train and validation sets
            if train_ratio == 0:
                x_train, y_train = self._create_empty_arrays(x_data, y_data)
                x_val, y_val = x_train_val, y_train_val
            elif val_ratio == 0 or len(x_train_val) == 0:
                x_train, y_train = x_train_val, y_train_val
                x_val, y_val = self._create_empty_arrays(x_data, y_data)
            else:
                can_stratify_tv: bool = self._can_stratify(y_train_val)
                x_train, x_val, y_train, y_val = train_test_split(
                    x_train_val, y_train_val,
                    test_size=(val_ratio / train_val_ratio),
                    random_state=random_state,
                    shuffle=True,  # Always true here
                    stratify=y_train_val if can_stratify_tv else None
                )

        self.logger.info(
            f"Data split complete. Train: {len(x_train)}, Validation: {len(x_val)}, Test: {len(x_test)}."
        )

        return SplitDataset(
            x_train=x_train, y_train=y_train,
            x_val=x_val, y_val=y_val,
            x_test=x_test, y_test=y_test
        )

    @staticmethod
    def _can_stratify(y_data: Y_Type) -> bool:
        """
        Checks if the label data allows for stratified splitting.

        Stratification is considered possible if the label array is one-dimensional
        (typical for classification tasks) and every class has at least two members.
        This is a requirement for `sklearn.model_selection.train_test_split`.

        :param y_data: The label data array to check.
        :returns: True if the data can be stratified, False otherwise.
        """
        if y_data.ndim == 1:  # Typical for classification
            _, counts = np.unique(y_data, return_counts=True)
            # Stratification requires at least 2 samples for each class present
            return all(count >= 2 for count in counts)
        return False  # Cannot stratify multidimensional or regression targets this way

    # noinspection PyTypeChecker
    @staticmethod
    def _create_empty_arrays(x_ref: X_Type, y_ref: Y_Type) -> tuple[X_Type, Y_Type]:
        """
        Creates empty arrays with dtypes and dimensions matching reference arrays.

        The created arrays will have a shape of (0, *dims), preserving the
        number of dimensions and data type of the reference arrays.

        :param x_ref: The reference feature array.
        :param y_ref: The reference label array.
        :returns: A tuple containing the empty feature array and empty label array.
        """
        empty_x_shape = (0,) + x_ref.shape[1:] if x_ref.ndim > 1 else (0,)
        empty_y_shape = (0,) + y_ref.shape[1:] if y_ref.ndim > 1 else (0,)
        empty_x = np.array([], dtype=x_ref.dtype).reshape(empty_x_shape)
        empty_y = np.array([], dtype=y_ref.dtype).reshape(empty_y_shape)
        return empty_x, empty_y

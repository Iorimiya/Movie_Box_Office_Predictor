from pathlib import Path
from typing import Optional
from abc import ABC, abstractmethod
from keras.src.models import Sequential
from keras.api.models import load_model
import numpy as np

from tools.util import check_path


class MachineLearningModel(ABC):
    """
    Abstract base class for machine learning models.
    """
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initializes the MachineLearningModel.

        Args:
            model_path (Optional[Path]): Path to the pre-trained model. Defaults to None.
        """
        self._model: Optional[Sequential] = load_model(model_path) if check_path(model_path) else None

    def _save_model(self, file_path: Path) -> None:
        """
        Saves the model to the specified file path.

        Args:
            file_path (Path): The path to save the model.
        """
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True)
        self._model.save(file_path)
        return

    @classmethod
    @abstractmethod
    def _load_training_data(cls, data_path: Path) -> any:
        """
        Abstract method to load training data from a given path.

        Args:
            data_path (Path): The path to the training data.

        Returns:
            any: The loaded training data.
        """
        pass

    @abstractmethod
    def _prepare_data(self, data: any) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Abstract method to prepare data for training and testing.

        Args:
            data (any): The raw data to be prepared.

        Returns:
            tuple[NDArray, NDArray, NDArray, NDArray]: A tuple containing x_train, y_train, x_test, y_test.
        """
        pass

    @abstractmethod
    def _build_model(self, model: Sequential, layers: list[any]) -> None:
        """
        Abstract method to build the model architecture.

        Args:
            model (Sequential): The Sequential model to build upon.
            layers (list[any]): A list of Keras layers to add to the model.
        """
        for layer in layers:
            model.add(layer)
        return

    @abstractmethod
    def train(self, data: any, epoch: int, old_model_path: Optional[Path] = None, ):
        """
        Abstract method to train the model.

        Args:
            data (any): The training data.
            epoch (int): The number of training epochs.
            old_model_path (Optional[Path]): Path to a pre-existing model to continue training. Defaults to None.
        """
        pass

    @abstractmethod
    def predict(self, data_input: any):
        """
        Abstract method to make predictions using the model.

        Args:
            data_input (any): The input data for prediction.

        Returns:
            any: The prediction result.
        """
        pass

    @staticmethod
    def _check_save_folder(folder_path: Path) -> None:
        """
        Checks if a folder exists and creates it if it doesn't.

        Args:
            folder_path (Path): The path to the folder.
        """
        if not folder_path.exists():
            folder_path.mkdir(parents=True)
        return

    def _create_model(self, layers: list[any], old_model_path: Optional[Path] = None) -> Sequential:
        """
        Creates or loads a model.

        Args:
            layers (list[any]): A list of Keras layers to build a new model, if needed.
            old_model_path (Optional[Path]): Path to an existing model to load. Defaults to None.

        Returns:
            Sequential: The created or loaded Sequential model.
        """
        if check_path(old_model_path):
            model: Sequential = load_model(old_model_path)
        else:
            model: Sequential = Sequential()
            self._build_model(model=model, layers=layers)
        return model

    def train_model(self, x_train: np.array, y_train: np.array, epoch: int = 200, batch_size: any = None) -> None:
        """
        Trains the model with the given training data.

        Args:
            x_train (NDArray): The input training data.
            y_train (NDArray): The target training data.
            epoch (int): The number of training epochs. Defaults to 200.
            batch_size (any): The batch size for training. Defaults to None.
        """
        self._model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size, verbose=1)
        return

    def evaluate_model(self, x_test: NDArray, y_test: NDArray) -> float:
        """
        Evaluates the model with the given test data.

        Args:
            x_test (NDArray): The input test data.
            y_test (NDArray): The target test data.

        Returns:
            float: The evaluation loss.
        """
        return self._model.evaluate(x_test, y_test, verbose=0)

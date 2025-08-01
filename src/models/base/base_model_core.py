from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Optional, TypeVar

from keras.api.models import load_model
from keras.src.callbacks import History
from keras.src.models import Model
from numpy.typing import NDArray

from src.utilities.filesystem_utils import is_existing_path

ModelBuildConfigType = TypeVar('ModelBuildConfigType')
TrainConfigType = TypeVar('TrainConfigType')
PredictConfigType = TypeVar('PredictConfigType')
EvaluateConfigType = TypeVar('EvaluateConfigType')


class BaseModelCore(
    Generic[ModelBuildConfigType, TrainConfigType, PredictConfigType, EvaluateConfigType],
    ABC
):
    """
    Abstract base class for the core model architecture.

    Defines a unified, type-safe interface for the entire model lifecycle.
    It is generic over the configuration objects for building, training,
    predicting, and evaluating, ensuring clarity and robustness.

    :ivar _model: The internal Keras model instance.
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        """
        Initializes the BaseModelCore.

        :param model_path: Optional path to a pre-trained Keras model file (.keras).
        """
        self._model: Optional[Model] = load_model(filepath=model_path) \
            if is_existing_path(path_obj=model_path) else None

    @abstractmethod
    def build(self, config: ModelBuildConfigType) -> None:
        """
        Builds and compiles a new model architecture based on a configuration object.

        Subclasses MUST implement this method to define the specific layers and
        compilation settings for their model, using parameters from the provided
        strongly-typed configuration object.

        :param config: A structured configuration object containing all necessary
                       parameters for building the model (e.g., input shape,
                       vocabulary size).
        """
        pass

    @abstractmethod
    def train(self, x_train: NDArray[any], y_train: NDArray[any], config: TrainConfigType) -> History:
        """
        Trains the model on the provided data based on a configuration object.

        Subclasses MUST implement this method to unpack the `config` object
        and pass the appropriate arguments to the underlying Keras `fit` method.

        :param x_train: Training data (features).
        :param y_train: Training data (labels).
        :param config: A structured configuration object containing all necessary
                       parameters for training (e.g., epochs, batch_size, validation_data).
        :returns: A Keras History object containing training history.
        """
        pass

    @abstractmethod
    def predict(self, data: NDArray[any], config: PredictConfigType) -> NDArray[any]:
        """
        Generates predictions for the given input data based on a configuration object.

        :param data: Input data for prediction.
        :param config: A structured configuration object for prediction.
        :returns: A NumPy array containing the predictions.
        """
        pass

    @abstractmethod
    def evaluate(self, x_test: NDArray[any], y_test: NDArray[any], config: EvaluateConfigType) -> any:
        """
        Evaluates the model on the test data based on a configuration object.

        :param x_test: Test data (features).
        :param y_test: Test data (labels).
        :param config: A structured configuration object for evaluation.
        :returns: A scalar loss value, or a list of scalars (loss and metrics) for the model.
        """
        pass

    def save(self, file_path: Path) -> None:
        """
        Saves the Keras model to a file.

        :param file_path: The path where the model file (.keras) will be saved.
        :raises ValueError: If the model has not been built or loaded.
        """
        if not self._model:
            raise ValueError("Model is not built or loaded. Cannot save.")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(filepath=file_path)

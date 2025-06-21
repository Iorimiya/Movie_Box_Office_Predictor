from logging import Logger
from pathlib import Path
from typing import Optional
from abc import ABC, abstractmethod
from keras.src.models import Sequential
from keras.api.models import load_model
from keras.src.callbacks import Callback
from numpy.typing import NDArray

from src.utilities.util import check_path_exists
from src.core.logging_manager import LoggingManager


class LossLoggingCallback(Callback):
    """A Keras Callback that logs training and validation loss at the end of each epoch.

    The loss values are logged using the standard Python logging module at the INFO level.
    If training loss ('loss') is available in the logs, it will be logged.
    If validation loss ('val_loss') is available in the logs, it will also be logged.
    """

    def on_epoch_end(self, epoch: int, logs: Optional[dict[str, any]] = None):
        """Called at the end of an epoch.

        Logs the training and validation loss if they are present in the logs dictionary.

        :param epoch: Integer, index of epoch (0-indexed).
        :param logs: Dictionary of logs. Contains loss values, and optionally
                     metric values. May contain 'loss' for training loss
                     and 'val_loss' for validation loss.
        """
        logger: Logger = LoggingManager().get_logger('machine_learning')
        logs = logs or {}
        loss = logs.get('loss')
        if loss is not None:
            logger.info(f"Epoch {epoch + 1}: Training loss = {loss:.4e}.")
        val_loss = logs.get('val_loss')
        if val_loss is not None:
            logger.info(f"Epoch {epoch + 1}: Validation loss = {val_loss:.4e}.")


class MachineLearningModel(ABC):
    """Abstract base class for machine learning models.

    Provides a common interface and basic functionalities for model loading,
    saving, training, and prediction. Subclasses must implement the abstract methods.
    """

    def __init__(self, model_path: Optional[Path] = None):
        """Initializes the MachineLearningModel.

        If a valid ``model_path`` is provided, the pre-trained Keras model is loaded.
        Otherwise, ``self._model`` is initialized to ``None``.

        :param model_path: Optional path to a pre-trained Keras model file.
        """
        self._model: Optional[Sequential] = load_model(model_path) if check_path_exists(model_path) else None

    def _save_model(self, file_path: Path) -> None:
        """Saves the Keras model (``self._model``) to the specified file path.

        If the parent directory of ``file_path`` does not exist, it will be created.
        Requires ``self._model`` to be an initialized Keras model.

        :param file_path: The path where the model will be saved.
        :raises AttributeError: If ``self._model`` is ``None`` (not loaded or created).
        :raises Exception: For potential I/O errors during folder creation or model saving.
        """
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True)
        self._model.save(file_path)
        return

    @classmethod
    @abstractmethod
    def _load_training_data(cls, data_path: Path) -> any:
        """Abstract method to load training data from a given path.

        Subclasses must implement this method to define how training data is loaded.

        :param data_path: The path to the training data.
        :returns: The loaded training data in a format expected by ``_prepare_data``.
        """
        pass

    @abstractmethod
    def _prepare_data(self, data: any) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Abstract method to prepare raw data for model training and testing.

        Subclasses must implement this method to define data preprocessing steps,
        such as feature extraction, scaling, and splitting into training and testing sets.

        :param data: The raw data to be prepared.
        :returns: A tuple containing ``x_train``, ``y_train``, ``x_test``, ``y_test`` as NumPy arrays.
        """
        pass

    @abstractmethod
    def _build_model(self, model: Sequential, layers: list) -> None:
        """Abstract method to define and build the model architecture.

        This method is responsible for adding layers to the provided Keras ``Sequential`` model
        and compiling it. The base implementation adds layers from the ``layers`` list.
        Subclasses typically override this to define a specific architecture and compilation process.

        :param model: The Keras ``Sequential`` model instance to build upon.
        :param layers: A list of Keras layers to add to the model.
        """
        for layer in layers:
            model.add(layer)
        return

    @abstractmethod
    def train(self, data: any, epoch: int, old_model_path: Optional[Path] = None, ):
        """Abstract method to train the machine learning model.

        Subclasses must implement this method to define the complete training workflow,
        which usually involves data loading, preparation, model creation/loading,
        and fitting the model.

        :param data: The training data, or a path to it.
        :param epoch: The number of training epochs.
        :param old_model_path: Optional path to a pre-existing model to continue training from.
        """
        pass

    @abstractmethod
    def predict(self, data_input: any):
        """Abstract method to make predictions using the trained model.

        Subclasses must implement this method to define how input data is processed
        for prediction and how the model's output is returned.

        :param data_input: The input data for which predictions are to be made.
        :returns: The prediction result from the model.
        """
        pass

    @staticmethod
    def _check_save_folder(folder_path: Path) -> None:
        """Checks if a folder exists at the given path and creates it if it doesn't.

        This includes creating any necessary parent directories.

        :param folder_path: The path to the folder to check and potentially create.
        :raises Exception: For potential I/O or permission errors during folder creation.
        """
        if not folder_path.exists():
            folder_path.mkdir(parents=True)
        return

    def _create_model(self, layers: Optional[list] = None, old_model_path: Optional[Path] = None) -> Sequential:
        """Creates a new Keras ``Sequential`` model or loads an existing one.

        If ``old_model_path`` is provided and valid, the model is loaded from that path.
        Otherwise, if ``layers`` are provided, a new ``Sequential`` model is created
        and built using ``self._build_model``.

        :param layers: An optional list of Keras layers to build a new model.
                       Required if ``old_model_path`` is not provided or invalid.
        :param old_model_path: Optional path to an existing Keras model file to load.
        :raises ValueError: If neither ``layers`` nor a valid ``old_model_path`` is provided.
        :returns: The created or loaded Keras ``Sequential`` model.
        """
        if not old_model_path and not layers:
            raise ValueError('Either layers or old_model_path must be provided.')
        elif check_path_exists(old_model_path):
            model: Sequential = load_model(old_model_path)
        else:
            model: Sequential = Sequential()
            self._build_model(model=model, layers=layers)
        return model

    def train_model(self, x_train: NDArray, y_train: NDArray, epoch: int = 200, batch_size: any = None) -> None:
        """Trains the Keras model (``self._model``) with the given training data.

        Uses the ``LossLoggingCallback`` to log training and validation loss.

        :param x_train: The input training data (features).
        :param y_train: The target training data (labels).
        :param epoch: The number of training epochs.
        :param batch_size: The batch size for training. If ``None``, Keras will use its default.
        :raises AttributeError: If ``self._model`` is ``None`` (not loaded or created).
        """
        self._model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size, verbose=1,
                        callbacks=LossLoggingCallback())
        return

    def evaluate_model(self, x_test: NDArray, y_test: NDArray) -> float:
        """Evaluates the Keras model (``self._model``) with the given test data.

        :param x_test: The input test data (features).
        :param y_test: The target test data (labels).
        :raises AttributeError: If ``self._model`` is ``None`` (not loaded or created).
        :returns: The evaluation loss (or primary metric if multiple are configured during compile).
        """
        return self._model.evaluate(x_test, y_test, verbose=0)

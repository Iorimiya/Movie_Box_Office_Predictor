from dataclasses import dataclass
from typing import Any

from keras.layers import Bidirectional
from numpy.typing import NDArray
from typing_extensions import override

from src.models.base.base_model_core import (
    Adam,
    BaseModelCore,
    Dense,
    Dropout,
    History,
    LSTM,
    Input,
    KerasEvaluateParams,
    KerasFitParams,
    KerasPredictParams,
    Sequential
)


@dataclass(frozen=True)
class BoxOfficeClassificationBuildConfig:
    """
    Configuration for building the Box Office Classification Model.

    :ivar input_shape: The shape of the input sequences (timesteps, features).
    :ivar lstm_units: The number of units in the LSTM layer.
    :ivar dense_units: The number of units in the hidden dense layer.
    :ivar dropout_rate: The dropout rate for regularization.
    :ivar num_classes: The number of output classes.
    :ivar learning_rate: The initial learning rate for the optimizer.
    :ivar clipnorm: The gradient clipping norm.
    """
    input_shape: tuple[int, int]
    lstm_units: int
    dense_units: int
    dropout_rate: float
    num_classes: int
    learning_rate: float
    clipnorm: float


@dataclass(frozen=True)
class BoxOfficeClassificationTrainParams(KerasFitParams):
    """
    Parameters for training the Box Office Classification Model.
    """
    pass


@dataclass(frozen=True)
class BoxOfficeClassificationPredictParams(KerasPredictParams):
    """
    Parameters for generating predictions with the Box Office Classification Model.
    """
    pass


@dataclass(frozen=True)
class BoxOfficeClassificationEvaluateParams(KerasEvaluateParams):
    """
    Parameters for evaluating the Box Office Classification Model.
    """
    pass


class BoxOfficeClassificationModelCore(
    BaseModelCore[
        BoxOfficeClassificationBuildConfig,
        BoxOfficeClassificationTrainParams,
        BoxOfficeClassificationPredictParams,
        BoxOfficeClassificationEvaluateParams
    ]
):
    """
    The core Keras implementation for the Box Office Classification Model.

    This class encapsulates the BiLSTM architecture designed to classify
    movie box office performance into discrete categories.
    """

    @override
    def build(self, config: BoxOfficeClassificationBuildConfig) -> None:
        """
        Builds and compiles the BiLSTM classification model using Sequential API.

        :param config: Configuration for the model architecture and optimizer.
        """
        # Create the model using the parameters from latest.ipynb
        self._model = Sequential([
            Input(shape=config.input_shape),
            Bidirectional(LSTM(config.lstm_units)),
            Dropout(config.dropout_rate),
            Dense(config.dense_units, activation="relu"),
            Dense(config.num_classes, activation="softmax"),
        ])

        # Configure the Adam optimizer with clipping
        optimizer = Adam(learning_rate=config.learning_rate, clipnorm=config.clipnorm)

        # Compile with standard classification metrics
        self._model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    @override
    def train(
        self, x_train: NDArray[Any], y_train: NDArray[Any], params: BoxOfficeClassificationTrainParams
    ) -> History:
        """
        Trains the model using the provided parameters and Keras fit method.

        :param x_train: Feature sequences for training.
        :param y_train: One-hot encoded labels for training.
        :param params: Training parameters including epochs and callbacks.
        :returns: A Keras History object.
        :raises ValueError: If the model has not been built or loaded.
        """
        if not self._model:
            raise ValueError("Model has not been built or loaded. Call build() first.")

        return self._model.fit(
            x=x_train,
            y=y_train,
            epochs=params.epochs,
            batch_size=params.batch_size,
            validation_data=params.validation_data,
            verbose=params.verbose,
            callbacks=params.callbacks,
            initial_epoch=params.initial_epoch
        )

    @override
    def predict(self, data: NDArray[Any], params: BoxOfficeClassificationPredictParams) -> NDArray[Any]:
        """
        Generates class probability predictions for the input data.

        :param data: Input feature sequences.
        :param params: Prediction parameters.
        :returns: A NumPy array of class probabilities.
        :raises ValueError: If the model has not been built or loaded.
        """
        if not self._model:
            raise ValueError("Model has not been built or loaded. Call build() first.")

        return self._model.predict(x=data, batch_size=params.batch_size, verbose=params.verbose)

    @override
    def evaluate(
        self, x_test: NDArray[Any], y_test: NDArray[Any], params: BoxOfficeClassificationEvaluateParams
    ) -> Any:
        """
        Evaluates the model on test data using loss and accuracy.

        :param x_test: Feature sequences for evaluation.
        :param y_test: One-hot encoded labels for evaluation.
        :param params: Evaluation parameters.
        :returns: A list containing loss and evaluation metrics.
        :raises ValueError: If the model has not been built or loaded.
        """
        if not self._model:
            raise ValueError("Model has not been built or loaded. Call build() first.")

        return self._model.evaluate(x=x_test, y=y_test, batch_size=params.batch_size, verbose=params.verbose)

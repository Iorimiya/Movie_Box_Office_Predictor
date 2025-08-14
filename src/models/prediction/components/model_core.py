from dataclasses import dataclass

from numpy.typing import NDArray
from typing_extensions import override

from src.models.base.base_model_core import BaseModelCore, BaseTrainConfig, BasePredictConfig, BaseEvaluateConfig
from src.models.base.keras_setup import keras_base

History = keras_base.callbacks.History
Sequential = keras_base.models.Sequential
Dense = keras_base.layers.Dense
Dropout = keras_base.layers.Dropout
Input = keras_base.layers.Input
LSTM = keras_base.layers.LSTM
Masking = keras_base.layers.Masking
Adam = keras_base.optimizers.Adam
ExponentialDecay = keras_base.optimizers.schedules.ExponentialDecay


@dataclass(frozen=True)
class PredictionBuildConfig:
    """
    Configuration for building the PredictionModelCore.

    :ivar input_shape: The shape of the input data, e.g., (sequence_length, num_features).
    :ivar lstm_units: The number of units in the LSTM layer.
    :ivar dropout_rate: The dropout rate to apply after the LSTM layer.
    """
    input_shape: tuple[int, int]
    lstm_units: int
    dropout_rate: float



@dataclass(frozen=True)
class PredictionTrainConfig(BaseTrainConfig):
    """
    Configuration for training the prediction model.
    Inherits all common training parameters from BaseTrainConfig.
    """
    pass


@dataclass(frozen=True)
class PredictionPredictConfig(BasePredictConfig):
    """
    Configuration for predicting with the prediction model.
    Inherits all common prediction parameters from BasePredictConfig.
    """
    pass


@dataclass(frozen=True)
class PredictionEvaluateConfig(BaseEvaluateConfig):
    """
    Configuration for evaluating the prediction model.
    Inherits all common evaluation parameters from BaseEvaluateConfig.
    """
    pass


# --- Model Core Implementation ---

class PredictionModelCore(
    BaseModelCore[PredictionBuildConfig, PredictionTrainConfig, PredictionPredictConfig, PredictionEvaluateConfig]
):
    """
    Defines the core architecture of the LSTM-based box office prediction model.

    This class implements the `build` method required by `BaseModelCore` to
    construct a specific LSTM network for time-series regression.
    """

    @override
    def build(self, config: PredictionBuildConfig) -> None:
        """
        Builds and compiles a new Keras Sequential model for box office prediction.

        The architecture consists of an LSTM layer with a Masking layer to handle
        padded sequences, followed by Dropout and a Dense output layer. It is
        compiled with an Adam optimizer and Mean Squared Error loss.

        :param config: The configuration object containing model build parameters.
        """
        self._model = Sequential(layers=[
            Input(shape=config.input_shape),
            Masking(mask_value=0.0),
            LSTM(units=config.lstm_units, activation='relu'),
            Dropout(rate=config.dropout_rate),
            Dense(units=1)
        ])
        self._compile_model()

    @override
    def train(self, x_train: NDArray[any], y_train: NDArray[any], config: PredictionTrainConfig) -> History:
        """
        Trains the box office prediction model using parameters from the config object.

        :param x_train: The training data (features).
        :param y_train: The training data (labels).
        :param config: A configuration object containing training parameters.
        :returns: A Keras `History` object containing a record of training loss values.
        :raises ValueError: If the model is not built or loaded before training.
        """
        if not self._model:
            raise ValueError("Model is not built or loaded. Cannot start training.")

        return self._model.fit(
            x=x_train,
            y=y_train,
            epochs=config.epochs,
            batch_size=config.batch_size,
            validation_data=config.validation_data,
            callbacks=config.callbacks,
            verbose=config.verbose,
            initial_epoch=config.initial_epoch
        )

    @override
    def predict(self, data: NDArray[any], config: PredictionPredictConfig) -> NDArray[any]:
        """
        Generates box office predictions using parameters from the config object.

        :param data: The input data for which to make predictions.
        :param config: A configuration object containing prediction parameters.
        :returns: A NumPy array of predictions.
        :raises ValueError: If the model is not built or loaded before prediction.
        """
        if not self._model:
            raise ValueError("Model is not built or loaded. Cannot make predictions.")

        return self._model.predict(x=data, batch_size=config.batch_size, verbose=config.verbose)

    @override
    def evaluate(self, x_test: NDArray[any], y_test: NDArray[any], config: PredictionEvaluateConfig) -> any:
        """
        Evaluates the box office prediction model using parameters from the config object.

        :param x_test: The test data (features).
        :param y_test: The test data (labels).
        :param config: A configuration object containing evaluation parameters.
        :returns: A scalar loss value (Mean Squared Error).
        :raises ValueError: If the model is not built or loaded before evaluation.
        """
        if not self._model:
            raise ValueError("Model is not built or loaded. Cannot evaluate.")

        return self._model.evaluate(x=x_test, y=y_test, batch_size=config.batch_size, verbose=config.verbose)

    def _compile_model(self) -> None:
        """
        Compiles the Keras model with a specific optimizer and loss function
        for regression.

        This uses an Adam optimizer with an exponential learning rate decay
        and gradient clipping.

        :raises ValueError: If the model has not been created by calling `build` first.
        """
        if not self._model:
            raise ValueError("Model has not been created yet. Call 'build' first.")

        clip_norm_value: float = 1.0
        initial_learning_rate: float = 0.001
        decay_steps: int = 1000
        decay_rate: float = 0.96

        optimizer = Adam(
            learning_rate=ExponentialDecay(
                initial_learning_rate=initial_learning_rate,
                decay_steps=decay_steps,
                decay_rate=decay_rate
            ),
            clipnorm=clip_norm_value
        )
        self._model.compile(optimizer=optimizer, loss='mse')

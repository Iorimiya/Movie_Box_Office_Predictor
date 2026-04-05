from dataclasses import dataclass
from typing import Any

from numpy.typing import NDArray
from typing_extensions import override

from src.models.base.base_model_core import (
    Adam,
    BaseModelCore,
    Dense,
    Dropout,
    ExponentialDecay,
    History,
    LSTM,
    Input, Masking,
    KerasEvaluateParams,
    KerasFitParams,
    KerasPredictParams,
    Sequential
)


@dataclass(frozen=True)
class BoxOfficeRegressionBuildConfig:
    """
    Configuration for building the BoxOfficeRegressionModelCore.

    :ivar input_shape: The shape of the input data, e.g., (sequence_length, num_features).
    :ivar lstm_units: The number of units in the LSTM layer.
    :ivar dropout_rate: The dropout rate to apply after the LSTM layer.
    """
    input_shape: tuple[int, int]
    lstm_units: int
    dropout_rate: float


@dataclass(frozen=True)
class BoxOfficeRegressionFitParams(KerasFitParams):
    """
    Configuration for training the Box Office Regression Model.
    Inherits all common training parameters from KerasFitParams.
    """
    pass


@dataclass(frozen=True)
class BoxOfficeRegressionPredictParams(KerasPredictParams):
    """
    Configuration for predicting with the Box Office Regression Model.
    Inherits all common box_office_regression parameters from KerasPredictParams.
    """
    pass


@dataclass(frozen=True)
class BoxOfficeRegressionEvaluateParams(KerasEvaluateParams):
    """
    Configuration for evaluating the Box Office Regression Model.
    Inherits all common evaluation parameters from KerasEvaluateParams.
    """
    pass


class BoxOfficeRegressionModelCore(
    BaseModelCore[
        BoxOfficeRegressionBuildConfig,
        BoxOfficeRegressionFitParams,
        BoxOfficeRegressionPredictParams,
        BoxOfficeRegressionEvaluateParams
    ]
):
    """
    Defines the core architecture of the LSTM-based Box Office Regression Model.

    This class implements the `build` method required by `BaseModelCore` to
    construct a specific LSTM network for time-series regression.
    """

    @override
    def build(self, config: BoxOfficeRegressionBuildConfig) -> None:
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
    def train(self, x_train: NDArray[Any], y_train: NDArray[Any], params: BoxOfficeRegressionFitParams) -> History:
        """
        Trains the Box Office Regression Model using parameters from the params object.

        :param x_train: The training data (features).
        :param y_train: The training data (labels).
        :param params: A configuration object containing training parameters.
        :returns: A Keras `History` object containing a record of training loss values.
        :raises ValueError: If the model is not built or loaded before training.
        """
        if not self._model:
            raise ValueError("Model is not built or loaded. Cannot start training.")

        return self._model.fit(
            x=x_train,
            y=y_train,
            epochs=params.epochs,
            batch_size=params.batch_size,
            validation_data=params.validation_data,
            callbacks=params.callbacks,
            verbose=params.verbose,
            initial_epoch=params.initial_epoch
        )

    @override
    def predict(self, data: NDArray[Any], params: BoxOfficeRegressionPredictParams) -> NDArray[Any]:
        """
        Generates box office predictions using parameters from the params object.

        :param data: The input data for which to make predictions.
        :param params: A configuration object containing parameters of Box Office Regression Model.
        :returns: A NumPy array of predictions.
        :raises ValueError: If the model is not built or loaded before prediction.
        """
        if not self._model:
            raise ValueError("Model is not built or loaded. Cannot make predictions.")

        return self._model.predict(x=data, batch_size=params.batch_size, verbose=params.verbose)

    @override
    def evaluate(self, x_test: NDArray[Any], y_test: NDArray[Any], params: BoxOfficeRegressionEvaluateParams) -> Any:
        """
        Evaluates the Box Office Regression Model using parameters from the params object.

        :param x_test: The test data (features).
        :param y_test: The test data (labels).
        :param params: A configuration object containing evaluation parameters.
        :returns: A scalar loss value (Mean Squared Error).
        :raises ValueError: If the model is not built or loaded before evaluation.
        """
        if not self._model:
            raise ValueError("Model is not built or loaded. Cannot evaluate.")

        return self._model.evaluate(x=x_test, y=y_test, batch_size=params.batch_size, verbose=params.verbose)

    def _compile_model(self) -> None:
        """
        Compiles the Keras model with a specific optimizer and loss function for regression.

        This uses an Adam optimizer with an exponential learning rate decay and gradient clipping.

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

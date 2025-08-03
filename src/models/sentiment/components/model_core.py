from dataclasses import dataclass, field
from typing import Optional

from keras.api.models import Sequential
from keras.src.callbacks import Callback, History
from keras.src.layers import LSTM, Dense, Dropout, Embedding, Input
from numpy.typing import NDArray
from typing_extensions import override

from src.models.base.base_model_core import BaseModelCore


@dataclass(frozen=True)
class SentimentBuildConfig:
    """
    Configuration for building the SentimentModelCore.

    :ivar vocabulary_size: The maximum number of words to keep, based on word frequency.
    :ivar embedding_dim: The dimensionality of the embedding vectors.
    :ivar lstm_units: The number of units in the LSTM layer.
    :ivar max_sequence_length: The maximum length of input sequences after padding.
    """
    vocabulary_size: int
    embedding_dim: int
    lstm_units: int
    max_sequence_length: int


@dataclass(frozen=True)
class SentimentTrainConfig:
    """
    Configuration for training the sentiment analysis model.

    :ivar epochs: The total number of epochs to train the model.
    :ivar batch_size: The batch size for training.
    :ivar validation_data: A tuple containing validation features and labels.
    :ivar callbacks: A list of Keras callbacks to use during training.
    """
    epochs: int
    batch_size: int
    validation_data: tuple[NDArray[any], NDArray[any]]
    verbose: int | str = 1
    callbacks: list[Callback] = field(default_factory=list)
    initial_epoch: int = 0


@dataclass(frozen=True)
class SentimentPredictConfig:
    """
    Configuration for predicting with the sentiment model.

    :ivar batch_size: The batch size for prediction.
    :ivar verbose: Verbosity mode for Keras `predict`.
    """
    batch_size: Optional[int] = None
    verbose: int | str = 'auto'


@dataclass(frozen=True)
class SentimentEvaluateConfig:
    """
    Configuration for evaluating the sentiment model.

    :ivar batch_size: The batch size for evaluation.
    :ivar verbose: Verbosity mode for Keras `evaluate`.
    """
    batch_size: Optional[int] = None
    verbose: int | str = 'auto'


# --- Model Core Implementation ---

class SentimentModelCore(
    BaseModelCore[SentimentBuildConfig, SentimentTrainConfig, SentimentPredictConfig, SentimentEvaluateConfig]
):
    """
    Defines the core architecture of the LSTM-based sentiment analysis model.

    This class implements the `build` method required by `BaseModelCore` to
    construct a specific LSTM network for binary text classification.
    """

    @override
    def build(self, config: SentimentBuildConfig) -> None:
        """
        Builds and compiles a new Keras Sequential model for sentiment analysis.

        This method defines an LSTM-based architecture suitable for binary
        classification tasks and compiles it with an Adam optimizer and
        Binary Crossentropy loss.

        :param config: The configuration object containing model build parameters.
        """
        self._model = Sequential(layers=[
            Input(shape=(config.max_sequence_length,)),
            Embedding(input_dim=config.vocabulary_size, output_dim=config.embedding_dim),
            LSTM(units=config.lstm_units, return_sequences=False),
            Dropout(rate=0.5),
            Dense(units=1, activation='sigmoid')
        ])
        self._compile_model()

    @override
    def train(self, x_train: NDArray[any], y_train: NDArray[any], config: SentimentTrainConfig) -> History:
        """
        Trains the sentiment analysis model using parameters from the config object.

        :param x_train: The training data (features).
        :param y_train: The training data (labels).
        :param config: A configuration object containing training parameters like epochs and batch size.
        :returns: A Keras `History` object containing a record of training loss values and metrics.
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
    def predict(self, data: NDArray[any], config: SentimentPredictConfig) -> NDArray[any]:
        """
        Generates sentiment predictions using parameters from the config object.

        :param data: The input data for which to make predictions.
        :param config: A configuration object containing prediction parameters.
        :returns: A NumPy array of predictions.
        :raises ValueError: If the model is not built or loaded before prediction.
        """
        if not self._model:
            raise ValueError("Model is not built or loaded. Cannot make predictions.")

        return self._model.predict(x=data, batch_size=config.batch_size, verbose=config.verbose)

    @override
    def evaluate(self, x_test: NDArray[any], y_test: NDArray[any], config: SentimentEvaluateConfig) -> any:
        """
        Evaluates the sentiment model using parameters from the config object.

        :param x_test: The test data (features).
        :param y_test: The test data (labels).
        :param config: A configuration object containing evaluation parameters.
        :returns: A scalar loss value, or a list of scalars (loss and metrics) for the model.
        :raises ValueError: If the model is not built or loaded before evaluation.
        """
        if not self._model:
            raise ValueError("Model is not built or loaded. Cannot evaluate.")

        return self._model.evaluate(x=x_test, y=y_test, batch_size=config.batch_size, verbose=config.verbose)

    def _compile_model(self) -> None:
        """
        Compiles the Keras model with a specific optimizer and loss function
        for binary classification.

        :raises ValueError: If the model has not been created by calling `build` first.
        """
        if not self._model:
            raise ValueError("Model has not been created yet. Call 'build' first.")

        self._model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

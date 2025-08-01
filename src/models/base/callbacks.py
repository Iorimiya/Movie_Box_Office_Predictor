from keras.src.callbacks import Callback
from numpy.typing import NDArray
from sklearn.metrics import f1_score
from typing_extensions import override


class F1ScoreHistory(Callback):
    """
    A Keras Callback to calculate and record the F1 score on validation data at the end of each epoch.

    :ivar validation_data: A tuple (x_val, y_val) containing the validation data.
    :ivar f1_scores: A list that stores the computed F1 score for each epoch.
    """

    @override
    def __init__(self, validation_data: tuple[NDArray[any], NDArray[any]]):
        """
        Initializes the F1ScoreHistory callback.

        :param validation_data: A tuple (x_val, y_val) to be used for calculating the F1 score.
        """
        super().__init__()
        self.validation_data: tuple[NDArray[any], NDArray[any]] = validation_data
        self.f1_scores: list[float] = []

    @override
    def on_epoch_end(self, epoch: int, logs: dict[str, any] | None = None) -> None:
        """
        Called at the end of an epoch to compute and store the F1 score.

        :param epoch: The index of the current epoch.
        :param logs: Metric results for this training epoch, and for the validation epoch.
        """
        x_val, y_val = self.validation_data
        y_pred_probs: NDArray[any] = self.model.predict(x=x_val, verbose=0)
        y_pred_labels: NDArray[any] = (y_pred_probs > 0.5).astype("int32")

        score: float = f1_score(y_true=y_val, y_pred=y_pred_labels, average='binary', zero_division=0)
        self.f1_scores.append(score)

        # Optionally, add it to the Keras logs so it's printed
        if logs is not None:
            logs['val_f1_score'] = score

        print(f" — val_f1_score: {score:.4f}")

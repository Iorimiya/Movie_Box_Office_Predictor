from pathlib import Path
from typing import Optional
from abc import ABC, abstractmethod
from keras.src.models import Sequential
from keras.api.models import load_model
import numpy as np

from tools.util import check_path


class MachineLearningModel(ABC):
    def __init__(self, model_path: Optional[Path] = None):
        self._model: Optional[Sequential] = load_model(model_path) if check_path(model_path) else None

    def _save_model(self, file_path: Path) -> None:
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True)
        self._model.save(file_path)
        return

    @classmethod
    @abstractmethod
    def _load_training_data(cls, data_path: Path) -> any:
        pass

    @abstractmethod
    def _prepare_data(self, data: any) -> tuple[np.array, np.array, np.array, np.array]:
        pass

    @abstractmethod
    def _build_model(self, model: Sequential, layers: list[any]) -> None:
        for layer in layers:
            model.add(layer)
        return

    @abstractmethod
    def train(self, data: any, epoch: int, old_model_path: Optional[Path] = None, ):
        pass

    @abstractmethod
    def predict(self, data_input: any):
        pass

    @staticmethod
    def _check_save_folder(folder_path: Path) -> None:
        if not folder_path.exists():
            folder_path.mkdir(parents=True)
        return

    def _create_model(self, layers: list[any], old_model_path: Optional[Path] = None) -> Sequential:
        if check_path(old_model_path):
            model: Sequential = load_model(old_model_path)
        else:
            model: Sequential = Sequential()
            self._build_model(model=model, layers=layers)
        return model

    def train_model(self, x_train: np.array, y_train: np.array, epoch: int = 200, batch_size: any = None) -> None:
        self._model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size, verbose=1)
        return

    def evaluate_model(self, x_test: np.array, y_test: np.array) -> float:
        return self._model.evaluate(x_test, y_test, verbose=0)

from pathlib import Path
from typing import Optional
from keras.src.models import Sequential
from keras.api.models import load_model

from tools.util import check_path


class MachineLearningModel:
    def __init__(self, model_path: Optional[Path] = None):
        self._model: Optional[Sequential] = load_model(model_path) if check_path(model_path) else None

    def _save_model(self, file_path: Path) -> None:
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True)
        self._model.save(file_path)
        return

    @staticmethod
    def _check_save_folder(folder_path: Path) -> None:
        if not folder_path.exists():
            folder_path.mkdir(parents=True)
        return

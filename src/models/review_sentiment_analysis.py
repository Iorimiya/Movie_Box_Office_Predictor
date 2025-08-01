import pickle
from logging import Logger
from pathlib import Path
from typing import Optional

import jieba
import numpy as np
import pandas as pd
from keras.src.layers import Embedding, LSTM, Dense, Dropout, Input
from keras.src.models import Sequential
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from numpy import int32, int64, float32
from numpy.typing import NDArray
from pandas import Series
from sklearn.model_selection import train_test_split

from src.core.logging_manager import LoggingManager
from src.models.machine_learning_model import MachineLearningModel
from src.utilities.filesystem_utils import is_existing_path

class ReviewSentimentAnalyseModel(MachineLearningModel):
    """
    A machine learning model for analyzing review sentiment.

    This model uses an LSTM network to classify the sentiment of text reviews
    as positive or negative. It handles tokenization, sequence padding,
    model training, and prediction.
    """

    def __init__(self, model_path: Optional[Path] = None, tokenizer_path: Optional[Path] = None, num_words: int = 5000,
                 review_max_length: int = 100):
        """
        Initializes the ReviewSentimentAnalyseModel.

        :param model_path: Path to the pre-trained Keras model file. If provided, the model will be loaded. Defaults to None.
        :param tokenizer_path: Path to the pre-trained Tokenizer pickle file. If provided, the tokenizer will be loaded. Defaults to None.
        :param num_words: The maximum number of words to keep, based on word frequency (vocabulary size). Defaults to 5000.
        :param review_max_length: The maximum length of a review sequence after padding. Defaults to 100.
        """
        super().__init__(model_path=model_path)
        self.__tokenizer: Optional[Tokenizer] = self.__load_tokenizer(tokenizer_path) if is_existing_path(
            tokenizer_path) else None
        self.__num_words: int = num_words
        self.__review_max_len: int = review_max_length
        self.__logger: Logger = LoggingManager().get_logger('root')
        return

    def __text_to_sequences(self, texts: list[str] | str) -> NDArray:
        """
        Converts text(s) to sequences of integers and pads them to a fixed length.

        Requires the tokenizer (``self.__tokenizer``) to be initialized.

        :param texts: A single string or a list of strings to be converted.
        :raises AttributeError: If ``self.__tokenizer`` has not been initialized (e.g., by loading or fitting).
        :returns: A NumPy array of padded sequences (``NDArray``).
        """
        sequence = self.__tokenizer.texts_to_sequences(texts if isinstance(texts, list) else [texts])
        return pad_sequences(sequence, maxlen=self.__review_max_len)

    def __save_tokenizer(self, file_path: Path) -> None:
        """
        Saves the current tokenizer (``self.__tokenizer``) to a file using pickle.

        The parent directory of ``file_path`` will be created if it doesn't exist.

        :param file_path: The path where the tokenizer will be saved.
        :raises AttributeError: If ``self.__tokenizer`` is None (not initialized).
        :raises Exception: For potential I/O errors during file writing or folder creation.
        """
        self._check_save_folder(file_path.parent)
        with open(file_path, 'wb') as handle:
            # noinspection PyTypeChecker
            pickle.dump(self.__tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return

    @staticmethod
    def __load_tokenizer(file_path: Path) -> Tokenizer:
        """
        Loads a tokenizer from a pickle file.

        :param file_path: Path to the tokenizer pickle file. Defaults to ``Constants.REVIEW_SENTIMENT_ANALYSIS_TOKENIZER_PATH``.
        :raises FileNotFoundError: If the tokenizer file does not exist.
        :raises pickle.UnpicklingError: If the file cannot be unpickled.
        :raises Exception: For other potential I/O errors.
        :returns: The loaded ``Tokenizer`` object.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")
        with open(file_path, 'rb') as handle:
            tokenizer: Tokenizer = pickle.load(handle)
        return tokenizer

    @classmethod
    def _load_training_data(cls, data_path: Path) -> any:
        """
        Loads training data from a CSV file into a pandas DataFrame.

        :param data_path: Path to the CSV file.
        :raises FileNotFoundError: If the file does not exist.
        :raises pd.errors.EmptyDataError: If the CSV file is empty.
        :raises Exception: For other potential I/O or parsing errors.
        :returns: The loaded training data as a pandas DataFrame.
        """
        LoggingManager().get_logger('root').info("Loading training data.")
        return pd.read_csv(data_path)

    def _prepare_data(self, data: any) -> tuple[NDArray[int32], NDArray[int64], NDArray[int32], NDArray[int64]]:
        """
        Prepares data for training and testing a sentiment analysis model.

        This involves:
        1. Separating positive and negative words from the input DataFrame.
        2. Constructing sample review sentences using these words.
        3. Creating corresponding labels (1 for positive, 0 for negative).
        4. Performing word segmentation using jieba.
        5. Initializing and fitting a Tokenizer on the segmented texts.
        6. Converting texts to padded sequences.
        7. Splitting the data into training and testing sets.

        :param data: Input data, expected to be a pandas DataFrame with 'is_positive' (boolean) and 'word' (string) columns.
        :raises KeyError: If 'is_positive' or 'word' columns are missing from the input DataFrame.
        :returns: A tuple containing (x_train, y_train, x_test, y_test) as NumPy arrays.
        """
        self.__logger.info("Change data format start.")
        positive_words: Series = data[data['is_positive']].dropna().loc[:, 'word']
        negative_words: Series = data[~data['is_positive']].dropna().loc[:, 'word']

        positive_samples: list[str] = ["這是一個非常" + word + "的電影，值得推薦！" for word in positive_words]
        negative_samples: list[str] = ["這是一個非常" + word + "的電影，完全不推薦！" for word in negative_words]

        positive_labels: list[int] = [1] * len(positive_samples)
        negative_labels: list[int] = [0] * len(negative_samples)

        texts: list[str] = positive_samples + negative_samples
        labels: list[int] = positive_labels + negative_labels

        texts: list[str] = [" ".join(jieba.lcut(text)) for text in texts]

        self.__tokenizer: Tokenizer = Tokenizer(num_words=self.__num_words)
        self.__tokenizer.fit_on_texts(texts)

        x_data: NDArray[int32] = self.__text_to_sequences(texts)
        y_data: NDArray[int64] = np.array(labels)

        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
        return x_train, y_train, x_test, y_test

    def _build_model(self, model: Sequential, layers: list) -> None:
        """
        Builds and compiles the Keras Sequential model for binary sentiment classification.

        The model is compiled with the 'adam' optimizer, 'binary_crossentropy' loss (suitable for binary classification),
        and 'accuracy' as a metric.

        :param model: The Keras ``Sequential`` model instance.
        :param layers: A list of Keras layers to add to the model.
        :returns: None
        """
        super()._build_model(model=model, layers=layers)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, data: any,
              tokenizer_save_path,
              model_save_folder: Path,
              model_save_name: str,
              old_model_path: Optional[Path] = None,
              epoch: int = 1000
              ) -> None:
        """
        Trains the sentiment analysis model.

        If ``old_model_path`` is provided and valid, training continues from the loaded model.
        Otherwise, a new model is created with an Embedding layer, LSTM layer, Dropout, and a Dense output layer.
        The tokenizer used for preparing data is saved, and the trained model is saved after training.

        :param data: Training data.
        :param tokenizer_save_path: Path to save the tokenizer.
        :param model_save_folder: Folder to save the model.
        :param model_save_name: Base name for the model file.
        :param old_model_path: Optional path to a pre-trained Keras model.
        :param epoch: Number of training epochs.
        :returns: None
        """
        self.__logger.info("Training procedure start.")

        x_train, y_train, x_test, y_test = self._prepare_data(data)

        if old_model_path:
            self._model: Sequential = self._create_model(old_model_path=old_model_path)
            new_epoch: int = int(old_model_path.stem.split('_')[-1]) + epoch
            save_name: str = f"{model_save_name}_{new_epoch}"
        else:
            self._model: Sequential = self._create_model(layers=[
                Input(shape=(self.__review_max_len,)),
                Embedding(input_dim=self.__num_words, output_dim=64),
                LSTM(units=128, return_sequences=False),
                Dropout(0.5),
                Dense(units=1, activation='sigmoid')
            ])
            save_name: str = f"{model_save_name}_{epoch}"
        self.train_model(x_train, y_train, epoch, batch_size=32)
        loss: float = self.evaluate_model(x_test, y_test)
        self.__logger.info(f"Model validation loss: {loss}.")
        self.__save_tokenizer(file_path=tokenizer_save_path)
        self._save_model(model_save_folder.joinpath(f"{save_name}.keras"))
        return None

    def predict(self, data_input: str = "這是一個非常感人的產品，值得推薦！") -> float:
        """
        Predicts the sentiment score for a given text input.

        :param data_input: The text string to predict sentiment for.
        :raises ValueError: If the model or tokenizer has not been loaded.
        :returns: The predicted sentiment score, a float between 0 and 1.
        """

        if not self._model or not self.__tokenizer:
            raise ValueError("model and tokenizer must be loaded.")
        data_input: str = " ".join(jieba.lcut(data_input))
        input_text: NDArray[int32] = self.__text_to_sequences(data_input)
        prediction: NDArray[float32] = self._model.predict(input_text)

        return float(prediction[0][0])

    def predict_sentiment_label(self,data_input:str) -> bool:
        """
        Predicts the sentiment label (positive/negative) for a given text input.

        :param data_input: The text string to predict sentiment for.
        :returns: True if the sentiment is positive (score > 0.5), False otherwise.
        """
        return True if self.predict(data_input=data_input) > 0.5 else False

    def simple_train(self, input_data: Path, tokenizer_save_path: Path,
                     model_save_name: str,model_save_folder:Path,
                     old_model_path: Optional[Path] = None, epoch: int = 1000
                     ) -> None:
        """
        A simplified interface to load data from a CSV file and train the model.

        This method first loads training data using ``_load_training_data`` and then calls
        the main ``train`` method.

        :param input_data: Path to the CSV file.
        :param tokenizer_save_path: Path to save the tokenizer.
        :param model_save_name: Base name for the model file.
        :param model_save_folder: Folder to save the model.
        :param old_model_path: Optional path to a pre-trained Keras model.
        :param epoch: Number of training epochs.
        :raises ValueError: If ``input_data`` is not a ``Path`` instance.
        """
        if isinstance(input_data, Path):
            train_data: any = self._load_training_data(data_path=input_data)
        else:
            raise ValueError
        self.train(data=train_data, old_model_path=old_model_path, epoch=epoch, model_save_folder=model_save_folder,
                   model_save_name=model_save_name, tokenizer_save_path=tokenizer_save_path)

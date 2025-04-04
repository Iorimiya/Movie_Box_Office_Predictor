import jieba
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from numpy import ndarray
from typing import Optional
from sklearn.model_selection import train_test_split
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.src.models import Sequential
from keras.src.layers import Embedding, LSTM, Dense, Dropout, Input

from machine_learning_model.machine_learning_model import MachineLearningModel
from movie_data import load_index_file
from tools.util import check_path
from tools.constant import Constants


class ReviewSentimentAnalyseModel(MachineLearningModel):
    """
    A machine learning model for analyzing review sentiment.
    """
    def __init__(self, model_path: Optional[Path] = None, tokenizer_path: Optional[Path] = None, num_words: int = 5000,
                 review_max_length: int = 100):
        """
        Initializes the ReviewSentimentAnalyseModel.

        Args:
            model_path (Optional[Path]): Path to the pre-trained model. Defaults to None.
            tokenizer_path (Optional[Path]): Path to the pre-trained tokenizer. Defaults to None.
            num_words (int): Size of the vocabulary. Defaults to 5000.
            review_max_length (int): Maximum length of a review. Defaults to 100.
        """
        super().__init__(model_path=model_path)
        self.__tokenizer: Optional[Tokenizer] = self.__load_tokenizer(tokenizer_path) if check_path(
            tokenizer_path) else None
        self.__num_words = num_words  # 詞彙表大小
        self.__review_max_len = review_max_length  # 每條影評的最大長度
        return

    def __text_to_sequences(self, texts: list[str] | str) -> ndarray:
        """
        Converts text to sequences of integers and pads them.

        Args:
            texts (list[str] | str): Input text(s).

        Returns:
            NDArray: Padded sequence(s) of integers.
        """
        sequence = self.__tokenizer.texts_to_sequences(texts if isinstance(texts, list) else [texts])  # 轉為數字序列
        return pad_sequences(sequence, maxlen=self.__review_max_len)  # 填充序列

    def __save_tokenizer(self, file_path: Path) -> None:
        """
        Saves the tokenizer to a file.

        Args:
            file_path (Path): Path to save the tokenizer.
        """
        self._check_save_folder(file_path.parent)
        with open(file_path, 'wb') as handle:
            pickle.dump(self.__tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return

    @staticmethod
    def __load_tokenizer(file_path: Path = Constants.REVIEW_SENTIMENT_ANALYSIS_TOKENIZER_PATH) -> Tokenizer:
        """
        Loads the tokenizer from a file.

        Args:
         file_path (Path): Path to the tokenizer file. Defaults to REVIEW_SENTIMENT_ANALYSIS_TOKENIZER_PATH.

        Returns:
         Tokenizer: Loaded tokenizer.

        Raises:
         FileNotFoundError: If the file does not exist.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")
        with open(file_path, 'rb') as handle:
            tokenizer: Tokenizer = pickle.load(handle)
        return tokenizer

    @classmethod
    def _load_training_data(cls, data_path: Path) -> any:
        """
        Loads training data from a CSV file.

        Args:
            data_path (Path): Path to the CSV file.

        Returns:
            any: Loaded training data as a pandas DataFrame.
        """
        logging.info("loading training data.")
        return pd.read_csv(data_path)

    def _prepare_data(self, data: any) -> tuple[np.array, np.array, np.array, np.array]:
        """
        Prepares data for training and testing.

        Args:
            data (any): Input data as a pandas DataFrame.

        Returns:
            tuple[NDArray, NDArray, NDArray, NDArray]: Tuple containing x_train, y_train, x_test, y_test.
        """
        logging.info("change data to dataset start.")
        positive_words = data[data['is_positive']].dropna().loc[:, 'word']
        negative_words = data[~data['is_positive']].dropna().loc[:, 'word']

        # 構造影評資料集
        positive_samples = ["這是一個非常" + word + "的電影，值得推薦！" for word in positive_words]
        negative_samples = ["這是一個非常" + word + "的電影，完全不推薦！" for word in negative_words]

        # 創建標籤
        positive_labels = [1] * len(positive_samples)  # 正面為1
        negative_labels = [0] * len(negative_samples)  # 負面為0

        # 合併影評與標籤
        texts = positive_samples + negative_samples
        labels = positive_labels + negative_labels

        texts = [" ".join(jieba.lcut(text)) for text in texts]  # 分詞函數

        # 使用 Tokenizer 將影評轉為數字序列
        self.__tokenizer = Tokenizer(num_words=self.__num_words)
        self.__tokenizer.fit_on_texts(texts)  # 建立詞彙表

        x_data = self.__text_to_sequences(texts)
        y_data = np.array(labels)  # 標籤

        # 拆分訓練集和測試集
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
        return x_train, y_train, x_test, y_test

    def _build_model(self,model: Sequential, layers: list[any]) -> None:
        """
        Builds and compiles the model.

        Args:
            model (Sequential): The Sequential model.
            layers (list[any]): List of layers to add to the model.
        """
        super()._build_model(model=model, layers=layers)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, data: any,
              old_model_path: Optional[Path] = None,
              epoch: int = 1000,
              model_save_folder: Path = Constants.REVIEW_SENTIMENT_ANALYSIS_MODEL_FOLDER,
              model_save_name: str = Constants.REVIEW_SENTIMENT_ANALYSIS_MODEL_NAME,
              tokenizer_save_path: Path = Constants.REVIEW_SENTIMENT_ANALYSIS_TOKENIZER_PATH) -> None:
        """
        Trains the model.

        Args:
            data (any): Training data.
            old_model_path (Optional[Path]): Path to an existing model. Defaults to None.
            epoch (int): Number of training epochs. Defaults to 1000.
            model_save_folder (Path): Folder to save the trained model. Defaults to REVIEW_SENTIMENT_ANALYSIS_MODEL_FOLDER.
            model_save_name (str): Name to save the trained model. Defaults to REVIEW_SENTIMENT_ANALYSIS_MODEL_NAME.
            tokenizer_save_path (Path): Path to save the tokenizer. Defaults to REVIEW_SENTIMENT_ANALYSIS_TOKENIZER_PATH.
        """
        logging.info("training procedure start.")

        x_train, y_train, x_test, y_test = self._prepare_data(data)

        self._model: Sequential = self._create_model(layers=[
            Input(shape=(self.__review_max_len,)),
            Embedding(input_dim=self.__num_words, output_dim=64),  # 嵌入層
            LSTM(units=128, return_sequences=False),  # LSTM 層
            Dropout(0.5),  # Dropout 防止過擬合
            Dense(units=1, activation='sigmoid')  # 輸出層
        ],
            old_model_path=old_model_path)
        self.train_model(x_train, y_train, epoch, batch_size=32)
        loss: float = self.evaluate_model(x_test, y_test)
        logging.info(f"model test loss: {loss}.")
        self.__save_tokenizer(file_path=tokenizer_save_path)
        self._save_model(model_save_folder.joinpath(f"{model_save_name}_{epoch}.keras"))
        return None

    def predict(self, data_input: str = "這是一個非常感人的產品，值得推薦！") -> bool:
        """
        Predicts the sentiment of a review.

        Args:
            data_input (str): Input review text. Defaults to "這是一個非常感人的產品，值得推薦！".

        Returns:
            bool: True if the sentiment is positive, False otherwise.

        Raises:
            ValueError: If the model or tokenizer is not loaded.
        """
        if not self._model or not self.__tokenizer:
            raise AssertionError("model and tokenizer must be loaded.")
        data_input = " ".join(jieba.lcut(data_input))  # 分詞
        input_text = self.__text_to_sequences(data_input)
        # 預測結果
        prediction = self._model.predict(input_text)

        # 結果解釋
        return True if prediction[0][0] > 0.5 else False

    def simple_train(self, input_data: Path, old_model_path: Optional[Path] = None, epoch: int = 1000,
                     model_save_folder: Path = Constants.REVIEW_SENTIMENT_ANALYSIS_MODEL_FOLDER,
                     model_save_name: str = Constants.REVIEW_SENTIMENT_ANALYSIS_MODEL_NAME,
                     tokenizer_save_path: Path = Constants.REVIEW_SENTIMENT_ANALYSIS_TOKENIZER_PATH) -> None:
        """
        Simplified training process.

        Args:
            input_data (Path): Path to the training data.
            old_model_path (Optional[Path]): Path to an existing model. Defaults to None.
            epoch (int): Number of training epochs. Defaults to 1000.
            model_save_folder (Path): Folder to save the trained model. Defaults to REVIEW_SENTIMENT_ANALYSIS_MODEL_FOLDER.
            model_save_name (str): Name to save the trained model. Defaults to REVIEW_SENTIMENT_ANALYSIS_MODEL_NAME.
            tokenizer_save_path (Path): Path to save the tokenizer. Defaults to REVIEW_SENTIMENT_ANALYSIS_TOKENIZER_PATH.

        Raises:
            ValueError: If input_data is not a Path.
        """
        if isinstance(input_data, Path):
            train_data: any = self._load_training_data(data_path=input_data)
        else:
            raise ValueError
        self.train(data=train_data, old_model_path=old_model_path, epoch=epoch, model_save_folder=model_save_folder,
                   model_save_name=model_save_name, tokenizer_save_path=tokenizer_save_path)

    def simple_predict(self, input_data: str | None) -> None:
        """
        Simplified prediction process.

        Args:
            input_data (str | None): Input review text or None to process all reviews from index file.

        Raises:
            ValueError: If input_data is not a string or None.
        """
        if isinstance(input_data, str):
            print(self.predict(input_data))
        elif input_data is None:
            for movie in load_index_file():
                movie.load_public_review()
                for review in movie.public_reviews:
                    review.sentiment_score = self.predict(review.content)
                movie.save_public_review(Constants.PUBLIC_REVIEW_FOLDER)
        else:
            raise ValueError
        return

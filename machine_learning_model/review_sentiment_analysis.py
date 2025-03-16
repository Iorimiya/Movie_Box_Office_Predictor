import jieba
import pickle
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
from keras.api.models import load_model

from machine_learning_model.machine_learning_model import MachineLearningModel
from tools.util import check_path
from tools.constant import Constants


class ReviewSentimentAnalyseModel(MachineLearningModel):
    def __init__(self, model_path: Optional[Path] = None, tokenizer_path: Optional[Path] = None, num_words: int = 5000,
                 review_max_length: int = 100):
        super().__init__(model_path=model_path)
        self.__tokenizer: Optional[Tokenizer] = self.__load_tokenizer(tokenizer_path) if check_path(
            tokenizer_path) else None
        self.__num_words = num_words  # 詞彙表大小
        self.__review_max_len = review_max_length  # 每條影評的最大長度
        return

    def __text_to_sequences(self, texts: list[str] | str) -> ndarray:
        sequence = self.__tokenizer.texts_to_sequences(texts if isinstance(texts, list) else [texts])  # 轉為數字序列
        return pad_sequences(sequence, maxlen=self.__review_max_len)  # 填充序列

    def __save_tokenizer(self, file_path: Path) -> None:
        self._check_save_folder(file_path.parent)
        with open(file_path, 'wb') as handle:
            pickle.dump(self.__tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return

    def __load_tokenizer(self, file_path: Path = Constants.REVIEW_SENTIMENT_ANALYSIS_TOKENIZER_PATH) -> Tokenizer:
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")
        with open(file_path, 'rb') as handle:
            tokenizer: Tokenizer = pickle.load(handle)
        return tokenizer

    def train(self, data_path: Path,
              old_model_path: Optional[Path] = None,
              epoch: int = 1000,
              model_save_folder: Path = Constants.REVIEW_SENTIMENT_ANALYSIS_MODEL_FOLDER,
              model_save_name: str = Constants.REVIEW_SENTIMENT_ANALYSIS_MODEL_NAME,
              tokenizer_save_path: Path = Constants.REVIEW_SENTIMENT_ANALYSIS_TOKENIZER_PATH) -> None:
        # load words from file
        data_frame = pd.read_csv(data_path)
        positive_words = data_frame[data_frame['is_positive']].dropna().loc[:, 'word']
        negative_words = data_frame[~data_frame['is_positive']].dropna().loc[:, 'word']

        # 構造影評資料集
        positive_samples = ["這是一個非常" + word + "的產品，值得推薦！" for word in positive_words]
        negative_samples = ["這是一個非常" + word + "的產品，完全不推薦！" for word in negative_words]

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

        self.__save_tokenizer(file_path=tokenizer_save_path)
        x_data = self.__text_to_sequences(texts)
        y_data = np.array(labels)  # 標籤

        # 拆分訓練集和測試集
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

        if check_path(old_model_path):
            self._model = load_model(old_model_path)
        else:
            # 建立 LSTM 模型
            self._model = Sequential([
                Input(shape=(self.__review_max_len,)),
                Embedding(input_dim=self.__num_words, output_dim=64),  # 嵌入層
                LSTM(units=128, return_sequences=False),  # LSTM 層
                Dropout(0.5),  # Dropout 防止過擬合
                Dense(units=1, activation='sigmoid')  # 輸出層
            ])

            # 編譯模型
            self._model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # 訓練模型
        self._model.fit(x_train, y_train, epochs=epoch, batch_size=32, validation_data=(x_test, y_test))
        self._save_model(model_save_folder.joinpath(f"{model_save_name}_{epoch}.keras"))
        return None

    def test(self, review: str = "這是一個非常感人的產品，值得推薦！") -> bool:
        review = " ".join(jieba.lcut(review))  # 分詞
        input_text = self.__text_to_sequences(review)
        # 預測結果
        prediction = self._model.predict(input_text)

        # 結果解釋
        return True if prediction[0][0] > 0.5 else False

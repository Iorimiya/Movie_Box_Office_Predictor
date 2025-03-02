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
from keras.src.layers import Embedding, LSTM, Dense, Dropout
from keras.api.models import load_model




class EmotionAnalyser:
    def __init__(self, model_path: Path | None = None, tokenizer_path: Path | None = None, num_words: int = 5000,
                 review_max_length: int = 100):
        self.__model: Optional[Sequential] = \
            load_model(model_path) if isinstance(model_path, Path) and model_path.exists() else None
        self.__tokenizer: Optional[Tokenizer] = None
        if isinstance(tokenizer_path, Path) and tokenizer_path.exists():
            self.__load_tokenizer(tokenizer_path)

        self.__num_words = num_words  # 詞彙表大小
        self.__review_max_len = review_max_length  # 每條影評的最大長度
        return

    def __text_to_sequences(self, texts: list[str] | str) -> ndarray:
        sequence = self.__tokenizer.texts_to_sequences(texts if isinstance(texts, list) else [texts])  # 轉為數字序列
        return pad_sequences(sequence, maxlen=self.__review_max_len)  # 填充序列

    def __save_tokenizer(self, file_path: Path = 'tokenizer.pickle') -> None:
        with open(file_path, 'wb') as handle:
            pickle.dump(self.__tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return

    def __load_tokenizer(self, file_path: Path = 'tokenizer.pickle') -> None:
        with open(file_path, 'rb') as handle:
            tokenizer:Tokenizer = pickle.load(handle)
        self.__tokenizer = tokenizer

    def train(self, data_path: Path, tokenizer_save_folder: Path, tokenizer_save_name: str,
              model_save_folder: Path, model_save_name: str, epoch: int = 10) -> None:
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

        self.__save_tokenizer(tokenizer_save_folder.joinpath(tokenizer_save_name))
        x_data = self.__text_to_sequences(texts)
        y_data = np.array(labels)  # 標籤

        # 拆分訓練集和測試集
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

        # 建立 LSTM 模型
        self.__model = Sequential([
            Embedding(input_dim=self.__num_words, output_dim=64, input_length=self.__review_max_len),  # 嵌入層
            LSTM(units=128, return_sequences=False),  # LSTM 層
            Dropout(0.5),  # Dropout 防止過擬合
            Dense(units=1, activation='sigmoid')  # 輸出層
        ])

        # 編譯模型
        self.__model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # 訓練模型
        _ = self.__model.fit(x_train, y_train, epochs=epoch, batch_size=32, validation_data=(x_test, y_test))
        save_path = model_save_folder.joinpath(f"{model_save_name}_{epoch}.keras")
        self.__model.save(save_path)
        return None

    def test(self, review: str = "這是一個非常感人的產品，值得推薦！") -> bool:
        review = " ".join(jieba.lcut(review))  # 分詞
        input_text = self.__text_to_sequences(review)
        # 預測結果
        prediction = self.__model.predict(input_text)

        # 結果解釋
        return True if prediction[0][0] > 0.5 else False

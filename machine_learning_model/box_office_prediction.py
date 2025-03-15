import numpy as np
from pathlib import Path
from typing import Optional
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense, Masking, Input
from keras.api.models import load_model
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
from typing import TypedDict
from joblib import dump as scaler_dump, load as scaler_load
import yaml
import random

from tools.constant import Constants
from movie_data import MovieData, load_index_file, PublicReview, IndexLoadMode


class MoviePredictionInputData(TypedDict):
    box_office: int
    replies_count: int
    review_sentiment_score: list[bool]
    review_contents: list[str]


class MoviePredictionSystem:
    def __init__(self, model_path: Optional[Path] = None, transform_scaler_path: Optional[Path] = None,
                 training_setting_path: Optional[Path] = None) -> None:
        self.__model: Optional[Sequential] = \
            load_model(model_path) if isinstance(model_path, Path) and model_path.exists() else None
        self.__transform_scaler: any = \
            scaler_load(transform_scaler_path) if isinstance(transform_scaler_path, Path) \
                                                  and transform_scaler_path.exists() else None
        self.training_week_limit: Optional[int] = None
        self.training_data_len: Optional[int] = None
        if training_setting_path and training_setting_path.exists():
            self.load_training_setting(training_setting_path)
        return

    def load_training_setting(self, path: Path, encoding: str = Constants.DEFAULT_ENCODING) -> None:
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")
        with open(path, mode='r', encoding=encoding) as file:
            load_data: dict[str:int] = yaml.load(file, yaml.loader.BaseLoader)
        self.training_data_len = int(load_data['training_data_len'])
        self.training_week_limit = int(load_data['training_week_limit'])
        return

    def save_training_setting(self, path: Path, encoding: str = Constants.DEFAULT_ENCODING) -> None:
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        yaml.Dumper.ignore_aliases = lambda self, _: True
        with open(path, mode='w', encoding=encoding) as file:
            yaml.dump(
                {'training_data_len': self.training_data_len, 'training_week_limit': self.training_week_limit}, file,
                allow_unicode=True)

    @staticmethod
    def generate_random_data(num_movies: int, weeks_range: tuple[int, int], reviews_range: tuple[int, int]) \
            -> list[list[MoviePredictionInputData]]:
        data = []
        for _ in range(num_movies):
            num_weeks = random.randint(weeks_range[0], weeks_range[1])
            movie_data = []
            for _ in range(num_weeks):
                num_reviews: int = random.randint(0, reviews_range[1])
                week_data: MoviePredictionInputData = MoviePredictionInputData(
                    box_office=random.randint(1000000, 10000000),
                    review_contents=["這部電影真棒！" if random.random() > 0.5 else "這部電影不太行。" for _ in
                                     range(num_reviews)],
                    review_sentiment_score=[random.choice([True, False]) for _ in range(num_reviews)],
                    replies_count=random.randint(0, num_reviews * 2))
                movie_data.append(week_data)
            data.append(movie_data)
        return data

    @staticmethod
    def load_data(index_path: Optional[Path] = None) -> list[list[MoviePredictionInputData]]:
        movie_data: list[MovieData] = load_index_file(file_path=index_path, mode=IndexLoadMode.FULL)
        training_data: list = []
        for movie in movie_data:
            new_movie_data: list[MoviePredictionInputData] = []
            for week in movie.box_office:
                box_office: int = week.box_office
                review_contents: list[str] = []
                review_sentiment_score: list[bool] = []
                replies_count: int = 0
                week_reviews: list[PublicReview] = list(
                    filter(lambda review: week.start_date <= review.date <= week.end_date, movie.public_reviews))
                for review in week_reviews:
                    replies_count += review.reply_count
                    review_sentiment_score.append(review.emotion_analyse)
                    review_contents.append(review.content)
                new_movie_data.append(MoviePredictionInputData(box_office=box_office, review_contents=review_contents,
                                                               review_sentiment_score=review_sentiment_score,
                                                               replies_count=replies_count))
            training_data.append(new_movie_data)
        return training_data

    @staticmethod
    def preprocess_data(data: list[list[MoviePredictionInputData]]) -> list[list[list[int | float]]]:
        return [[[week["box_office"], sum(week["review_sentiment_score"]) / len(week["review_sentiment_score"]) if week[
            "review_sentiment_score"] else 0, week["replies_count"]] for week in movie] for movie in data]

    def prepare_sequences(self, data: list[list[list[int | float]]]) -> tuple[np.array, np.array]:
        x, y = [], []
        for movie in data:
            for i in range(len(movie) - self.training_week_limit):
                seq_x: list[list[int | float]] = movie[i:i + self.training_week_limit]
                seq_y: int = movie[i + self.training_week_limit][0]
                x.append(seq_x)
                y.append(seq_y)
        if self.training_data_len:
            x = pad_sequences(x, maxlen=self.training_data_len, dtype='float32', padding='post')
        return np.array(x), np.array(y)

    def train_and_evaluate_model(self, x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array,
                                 epoch: int = 200) -> tuple[Sequential, float]:
        model: Sequential = Sequential()
        model.add(Input(shape=(self.training_data_len, x_train.shape[2])))
        model.add(Masking(mask_value=0.0))
        model.add(LSTM(50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(x_train, y_train, epochs=epoch, verbose=1)
        loss: float = model.evaluate(x_test, y_test, verbose=0)
        return model, loss

    def predict_next_week_box_office(self, user_input: list[MoviePredictionInputData]) -> float:
        processed_input: list[list[int | float]] = self.preprocess_data([user_input])[0]
        scaled_input: np.ndarray = self.__transform_scaler.transform(np.array(processed_input)[:, 0].reshape(-1, 1))
        processed_input_scaled: np.ndarray = np.array(processed_input)
        processed_input_scaled[:, 0] = scaled_input.flatten()
        input_sequence: list[np.ndarray] = [processed_input_scaled[-self.training_week_limit:]]
        input_sequence_padded: np.ndarray = pad_sequences(input_sequence, maxlen=self.training_data_len,
                                                          dtype='float32',
                                                          padding='post')

        prediction_scaled: float = self.__model.predict(input_sequence_padded)[0, 0]
        prediction: float = self.__transform_scaler.inverse_transform([[prediction_scaled]])[0, 0]
        return prediction

    def train(self, data: list[list[MoviePredictionInputData]], epoch: int = 1000,
              model_save_path: Optional[Path] = None,
              setting_save_path: Path = Constants.BOX_OFFICE_PREDICTION_SETTING_PATH,
              scaler_save_path: Path = Constants.BOX_OFFICE_PREDICTION_SCALER_PATH, training_week_limit: int = 4,
              split_rate: float = 0.8) -> None:
        split_index: int = int(len(data) * split_rate)
        train_data: list[list[MoviePredictionInputData]] = data[:split_index]
        test_data: list[list[MoviePredictionInputData]] = data[split_index:]

        processed_train_data: list[list[list[int | float]]] = self.preprocess_data(train_data)
        processed_test_data: list[list[list[int | float]]] = self.preprocess_data(test_data)

        self.training_week_limit = training_week_limit
        self.training_data_len = max(len(movie) for movie in processed_train_data + processed_test_data)
        x_train, y_train = self.prepare_sequences(processed_train_data)
        x_test, y_test = self.prepare_sequences(processed_test_data)

        self.__transform_scaler: MinMaxScaler = MinMaxScaler()
        y_train_scaled: np.ndarray = self.__transform_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        x_train_scaled: np.ndarray = x_train.copy()
        for i in range(x_train.shape[0]):
            x_train_scaled[i, :, 0] = self.__transform_scaler.transform(x_train[i, :, 0].reshape(-1, 1)).flatten()

        y_test_scaled: np.ndarray = self.__transform_scaler.transform(y_test.reshape(-1, 1)).flatten()
        x_test_scaled: np.ndarray = x_test.copy()
        for i in range(x_test.shape[0]):
            x_test_scaled[i, :, 0] = self.__transform_scaler.transform(x_test[i, :, 0].reshape(-1, 1)).flatten()

        self.__model, loss = self.train_and_evaluate_model(x_train_scaled, y_train_scaled, x_test_scaled, y_test_scaled,
                                                           epoch)
        print(f"模型測試損失：{loss}")
        if not model_save_path:
            model_save_path = Constants.BOX_OFFICE_PREDICTION_MODEL_PATH.parent.joinpath(
                f"prediction_model_{epoch}.keras")
        self.__model.save(model_save_path)
        self.save_training_setting(setting_save_path)
        scaler_dump(self.__transform_scaler, scaler_save_path)

    def test(self, user_movie_data: list[MoviePredictionInputData]):
        prediction: Optional[float] = self.predict_next_week_box_office(user_movie_data)
        if prediction is not None:
            print(f"預測下週票房：{prediction}")

    def train_with_auto_generated_data(self):
        train_data: list[list[MoviePredictionInputData]] = self.generate_random_data(50, (4, 10), (
            0, 5))
        self.train(train_data)

    def test_with_auto_generated_data(self):
        test_data: list[MoviePredictionInputData] = self.generate_random_data(1, (1, 20), (1, 100))[0]
        self.test(test_data)

import numpy as np
from pathlib import Path
from typing import Optional
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense, Masking, Input, Dropout
from keras.api.models import load_model
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
from typing import TypedDict
from joblib import dump as scaler_dump, load as scaler_load
import yaml
import random

from machine_learning_model.machine_learning_model import MachineLearningModel
from tools.constant import Constants
from tools.util import check_path
from movie_data import MovieData, load_index_file, PublicReview, IndexLoadMode


class MoviePredictionInputData(TypedDict):
    box_office: int
    replies_count: int
    review_sentiment_score: list[bool]
    review_contents: list[str]


class MoviePredictionModel(MachineLearningModel):
    def __init__(self, model_path: Optional[Path] = None, transform_scaler_path: Optional[Path] = None,
                 training_setting_path: Optional[Path] = None) -> None:
        super().__init__(model_path=model_path)
        self.__transform_scaler: Optional[MinMaxScaler] = scaler_load(transform_scaler_path) if check_path(
            transform_scaler_path) else None
        self.__training_week_limit: Optional[int] = None
        self.__training_data_len: Optional[int] = None
        if check_path(training_setting_path):
            self.__load_setting(training_setting_path)
        return

    def __save_training_setting(self, file_path: Path, encoding: str = Constants.DEFAULT_ENCODING) -> None:
        self._check_save_folder(file_path.parent)
        yaml.Dumper.ignore_aliases = lambda self_, _: True
        with open(file_path, mode='w', encoding=encoding) as file:
            yaml.dump(
                {'training_data_len': self.__training_data_len, 'training_week_limit': self.__training_week_limit},
                file,
                allow_unicode=True)

    def __load_setting(self, file_path: Path, encoding: str = Constants.DEFAULT_ENCODING) -> None:
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")
        with open(file_path, mode='r', encoding=encoding) as file:
            load_data: dict[str:int] = yaml.load(file, yaml.loader.BaseLoader)
        self.__training_data_len = int(load_data['training_data_len'])
        self.__training_week_limit = int(load_data['training_week_limit'])
        return

    def __save_scaler(self, file_path: Path) -> None:
        self._check_save_folder(file_path.parent)
        scaler_dump(self.__transform_scaler, file_path)
        return

    @staticmethod
    def __generate_random_data(num_movies: int, weeks_range: tuple[int, int], reviews_range: tuple[int, int]) \
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
    def __load_data(index_path: Optional[Path] = Constants.INDEX_PATH) -> list[list[MoviePredictionInputData]]:
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
                    filter(lambda review_: week.start_date <= review_.date <= week.end_date, movie.public_reviews))
                for review in week_reviews:
                    replies_count += review.reply_count
                    review_sentiment_score.append(review.sentiment_score)
                    review_contents.append(review.content)
                new_movie_data.append(MoviePredictionInputData(box_office=box_office, review_contents=review_contents,
                                                               review_sentiment_score=review_sentiment_score,
                                                               replies_count=replies_count))
            training_data.append(new_movie_data)
        return training_data

    @staticmethod
    def __preprocess_data(data: list[list[MoviePredictionInputData]]) -> list[list[list[int | float]]]:
        return [[[week["box_office"], sum(week["review_sentiment_score"]) / len(week["review_sentiment_score"]) if week[
            "review_sentiment_score"] else 0, week["replies_count"]] for week in movie] for movie in data]

    def __prepare_sequences(self, data: list[list[list[int | float]]]) -> tuple[np.array, np.array]:
        x, y = [], []
        for movie in data:
            for i in range(len(movie) - self.__training_week_limit):
                seq_x: list[list[int | float]] = movie[i:i + self.__training_week_limit]
                seq_y: int = movie[i + self.__training_week_limit][0]
                x.append(seq_x)
                y.append(seq_y)
        if self.__training_data_len:
            x = pad_sequences(x, maxlen=self.__training_data_len, dtype='float32', padding='post')
        return np.array(x), np.array(y)

    def __train_and_evaluate_model(self, x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array,
                                   epoch: int = 200, old_model_path: Optional[Path] = None) -> tuple[Sequential, float]:
        if check_path(old_model_path):
            model = load_model(old_model_path)
        else:
            model: Sequential = Sequential()
            model.add(Input(shape=(self.__training_data_len, x_train.shape[2])))
            model.add(Masking(mask_value=0.0))
            model.add(LSTM(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
        model.fit(x_train, y_train, epochs=epoch, verbose=1)
        loss: float = model.evaluate(x_test, y_test, verbose=0)
        return model, loss

    def __predict_next_week_box_office(self, user_input: list[MoviePredictionInputData]) -> float:
        processed_input: list[list[int | float]] = self.__preprocess_data([user_input])[0]
        scaled_input: np.ndarray = self.__transform_scaler.transform(np.array(processed_input)[:, 0].reshape(-1, 1))
        processed_input_scaled: np.ndarray = np.array(processed_input)
        processed_input_scaled[:, 0] = scaled_input.flatten()
        input_sequence: list[np.ndarray] = [processed_input_scaled[-self.__training_week_limit:]]
        input_sequence_padded: np.ndarray = pad_sequences(input_sequence, maxlen=self.__training_data_len,
                                                          dtype='float32',
                                                          padding='post')

        prediction_scaled: float = self._model.predict(input_sequence_padded)[0, 0]
        prediction: float = self.__transform_scaler.inverse_transform([[prediction_scaled]])[0, 0]
        return prediction

    def train(self, data: list[list[MoviePredictionInputData]],
              old_model_path: Optional[Path] = None,
              epoch: int = 1000,
              model_save_folder: Path = Constants.BOX_OFFICE_PREDICTION_MODEL_FOLDER,
              model_save_name: str = Constants.BOX_OFFICE_PREDICTION_MODEL_NAME,
              setting_save_path: Path = Constants.BOX_OFFICE_PREDICTION_SETTING_PATH,
              scaler_save_path: Path = Constants.BOX_OFFICE_PREDICTION_SCALER_PATH,
              training_week_limit: int = 4,
              split_rate: float = 0.8) -> None:
        split_index: int = int(len(data) * split_rate)
        train_data: list[list[MoviePredictionInputData]] = data[:split_index]
        test_data: list[list[MoviePredictionInputData]] = data[split_index:]

        processed_train_data: list[list[list[int | float]]] = self.__preprocess_data(train_data)
        processed_test_data: list[list[list[int | float]]] = self.__preprocess_data(test_data)

        self.__training_week_limit = training_week_limit
        self.__training_data_len = max(len(movie) for movie in processed_train_data + processed_test_data)
        x_train, y_train = self.__prepare_sequences(processed_train_data)
        x_test, y_test = self.__prepare_sequences(processed_test_data)

        self.__transform_scaler: MinMaxScaler = MinMaxScaler()
        y_train_scaled: np.ndarray = self.__transform_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        x_train_scaled: np.ndarray = x_train.copy()
        for i in range(x_train.shape[0]):
            x_train_scaled[i, :, 0] = self.__transform_scaler.transform(x_train[i, :, 0].reshape(-1, 1)).flatten()

        y_test_scaled: np.ndarray = self.__transform_scaler.transform(y_test.reshape(-1, 1)).flatten()
        x_test_scaled: np.ndarray = x_test.copy()
        for i in range(x_test.shape[0]):
            x_test_scaled[i, :, 0] = self.__transform_scaler.transform(x_test[i, :, 0].reshape(-1, 1)).flatten()

        self._model, loss = self.__train_and_evaluate_model(x_train_scaled, y_train_scaled, x_test_scaled,
                                                            y_test_scaled,
                                                            epoch, old_model_path)
        print(f"模型測試損失：{loss}")
        self._save_model(file_path=model_save_folder.joinpath(f"{model_save_name}_{epoch}.keras"))
        self.__save_training_setting(setting_save_path)
        self.__save_scaler(scaler_save_path)

    def test(self, user_movie_data: list[MoviePredictionInputData]):
        prediction: Optional[float] = self.__predict_next_week_box_office(user_movie_data)
        if prediction is not None:
            print(f"預測下週票房：{prediction}")

    def movie_train(self, epoch: int = 1000):
        train_data: list[list[MoviePredictionInputData]] = self.__load_data()
        self.train(train_data, epoch=epoch)

    def train_with_auto_generated_data(self, epoch: int = 1000):
        train_data: list[list[MoviePredictionInputData]] = self.__generate_random_data(50, (4, 10), (
            0, 5))
        self.train(train_data, epoch=epoch)

    def test_with_auto_generated_data(self):
        test_data: list[MoviePredictionInputData] = self.__generate_random_data(1, (1, 20), (1, 100))[0]
        self.test(test_data)

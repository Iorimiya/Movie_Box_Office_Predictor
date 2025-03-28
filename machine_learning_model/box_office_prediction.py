import yaml
import random
import logging
import numpy as np
from pathlib import Path
from typing import Optional, TypedDict
from sklearn.preprocessing import MinMaxScaler
from joblib import dump as scaler_dump, load as scaler_load
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense, Masking, Input, Dropout
from keras_preprocessing.sequence import pad_sequences

from tools.util import check_path
from tools.constant import Constants
from machine_learning_model.machine_learning_model import MachineLearningModel
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
        self.__split_rate: Optional[float] = None
        if check_path(training_setting_path):
            self.__load_training_setting(training_setting_path)
        return

    def __save_training_setting(self, file_path: Path, encoding: str = Constants.DEFAULT_ENCODING) -> None:
        self._check_save_folder(file_path.parent)
        yaml.Dumper.ignore_aliases = lambda self_, _: True
        with open(file_path, mode='w', encoding=encoding) as file:
            yaml.dump(
                {'training_data_len': self.__training_data_len, 'training_week_limit': self.__training_week_limit},
                file,
                allow_unicode=True)

    def __load_training_setting(self, file_path: Path, encoding: str = Constants.DEFAULT_ENCODING) -> None:
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
    def __transform_single_movie_data(movie: MovieData) -> list[MoviePredictionInputData]:
        output: list[MoviePredictionInputData] = []
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
            output.append(MoviePredictionInputData(box_office=box_office, review_contents=review_contents,
                                                   review_sentiment_score=review_sentiment_score,
                                                   replies_count=replies_count))
        return output

    @staticmethod
    def __preprocess_data(data: list[list[MoviePredictionInputData]]) -> list[list[list[int | float]]]:
        return [[[week["box_office"], sum(week["review_sentiment_score"]) / len(week["review_sentiment_score"]) if week[
            "review_sentiment_score"] else 0, week["replies_count"]] for week in movie] for movie in data]

    @classmethod
    def _load_training_data(cls, data_path: Path) -> list[list[MoviePredictionInputData]]:
        logging.info("loading training data.")
        movie_data: list[MovieData] = load_index_file(file_path=data_path, mode=IndexLoadMode.FULL)
        training_data: list[list[MoviePredictionInputData]] = [cls.__transform_single_movie_data(movie=movie) for movie
                                                               in movie_data]
        return training_data

    def _prepare_data(self, data: any) -> tuple[np.array, np.array, np.array, np.array]:
        processed_data: list[list[list[int | float]]] = self.__preprocess_data(data)
        self.__training_data_len = max(len(movie) for movie in processed_data)

        # prepare_sequences
        x, y = [], []
        for movie in processed_data:
            for i in range(len(movie) - self.__training_week_limit):
                seq_x: list[list[int | float]] = movie[i:i + self.__training_week_limit]
                seq_y: int = movie[i + self.__training_week_limit][0]
                x.append(seq_x)
                y.append(seq_y)
        if self.__training_data_len:
            x = pad_sequences(x, maxlen=self.__training_data_len, dtype='float32', padding='post')

        x, y = np.array(x), np.array(y)

        split_index: int = int(len(x) * self.__split_rate)
        x_train, y_train, x_test, y_test = x[:split_index], y[:split_index], x[split_index:], y[split_index:]

        # Standardization
        self.__transform_scaler: MinMaxScaler = MinMaxScaler()
        y_train_scaled: np.ndarray = self.__transform_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        x_train_scaled: np.ndarray = x_train.copy()
        for i in range(x_train.shape[0]):
            x_train_scaled[i, :, 0] = self.__transform_scaler.transform(x_train[i, :, 0].reshape(-1, 1)).flatten()

        y_test_scaled: np.ndarray = self.__transform_scaler.transform(y_test.reshape(-1, 1)).flatten()
        x_test_scaled: np.ndarray = x_test.copy()
        for i in range(x_test.shape[0]):
            x_test_scaled[i, :, 0] = self.__transform_scaler.transform(x_test[i, :, 0].reshape(-1, 1)).flatten()
        return x_train_scaled, y_train_scaled, x_test_scaled, y_test_scaled

    def _build_model(self, model: Sequential, layers: list[any]) -> None:
        super()._build_model(model=model, layers=layers)
        model.compile(optimizer='adam', loss='mse')

    def train(self, data: list[list[MoviePredictionInputData]],
              old_model_path: Optional[Path] = None,
              epoch: int = 1000,
              model_save_folder: Path = Constants.BOX_OFFICE_PREDICTION_MODEL_FOLDER,
              model_save_name: str = Constants.BOX_OFFICE_PREDICTION_MODEL_NAME,
              setting_save_path: Path = Constants.BOX_OFFICE_PREDICTION_SETTING_PATH,
              scaler_save_path: Path = Constants.BOX_OFFICE_PREDICTION_SCALER_PATH,
              training_week_limit: int = 4,
              split_rate: float = 0.8) -> None:

        logging.info("training procedure start.")
        self.__training_week_limit = training_week_limit
        self.__split_rate = split_rate
        x_train, y_train, x_test, y_test = self._prepare_data(data)

        self._model: Sequential = self._create_model(layers=[
            Input(shape=(self.__training_data_len, x_train.shape[2])),
            Masking(mask_value=0.0),
            LSTM(128, activation='relu'),
            Dropout(0.5),
            Dense(1)
        ],
            old_model_path=old_model_path)
        self.train_model(x_train, y_train, epoch)
        loss: float = self.evaluate_model(x_test, y_test)
        logging.info(f"model test loss: {loss}.")
        self._save_model(file_path=model_save_folder.joinpath(f"{model_save_name}_{epoch}.keras"))
        np.save(setting_save_path.with_name('x_test.npy'), x_test)
        np.save(setting_save_path.with_name('y_test.npy'), y_test)
        self.__save_training_setting(setting_save_path)
        self.__save_scaler(scaler_save_path)

    def predict(self, data_input: list[MoviePredictionInputData]) -> float:
        if not self._model or not self.__transform_scaler or not self.__training_data_len or not self.__training_week_limit:
            raise AssertionError('model, settings, and scaler must be loaded.')
        processed_input: list[list[int | float]] = self.__preprocess_data([data_input])[0]
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

    def evaluate_trend(self, test_data_folder_path: Path = Constants.BOX_OFFICE_PREDICTION_SETTING_PATH.parent) -> None:
        if not self._model or not self.__transform_scaler or not self.__training_data_len or not self.__training_week_limit:
            raise AssertionError('model, settings, and scaler must be loaded.')
        try:
            x_test_loaded = np.load(test_data_folder_path.joinpath("x_test.npy"))
            y_test_loaded = np.load(test_data_folder_path.joinpath("y_test.npy"))
        except FileNotFoundError:
            print(f"錯誤：在 '{test_data_folder_path}' 目錄中找不到 X_test.npy 或 y_test.npy。")
            return None

        correct_predictions = 0
        total_predictions = 0

        for i in range(len(x_test_loaded)):
            if len(x_test_loaded[i]) >= self.__training_week_limit:
                input_sequence = x_test_loaded[i][-self.__training_week_limit:].reshape(
                    (1, self.__training_week_limit, x_test_loaded.shape[-1]))

                # 標準化輸入序列的票房特徵
                input_sequence_scaled = input_sequence.copy()
                for j in range(input_sequence_scaled.shape[1]):
                    input_sequence_scaled[0, j, 0] = self.__transform_scaler.transform(
                        input_sequence[0, j, 0].reshape(-1, 1)).flatten()

                predicted_box_office_scaled = self._model.predict(input_sequence_scaled)[0, 0]
                predicted_box_office = \
                    self.__transform_scaler.inverse_transform(np.array([[predicted_box_office_scaled]]))[
                        0, 0]

                # 判斷預測趨勢
                current_week_actual_box_office = \
                    self.__transform_scaler.inverse_transform(np.array([[x_test_loaded[i][-1, 0]]]))[0, 0]
                predicted_trend = 1 if predicted_box_office > current_week_actual_box_office else 0

                # 判斷實際趨勢
                if i < len(y_test_loaded):
                    actual_next_week_box_office = \
                        self.__transform_scaler.inverse_transform(np.array([[y_test_loaded[i]]]))[
                            0, 0]
                    actual_trend = 1 if actual_next_week_box_office > current_week_actual_box_office else 0

                    if predicted_trend == actual_trend:
                        correct_predictions += 1
                    total_predictions += 1
                else:
                    print(f"警告：X_test 的長度超過 y_test，無法判斷實際趨勢。")

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"趨勢預測準確率：{accuracy:.2%}")
        return

    def simple_train(self, input_data: Path | list[MovieData] | None,
                     old_model_path: Optional[Path] = None,
                     epoch: int = 1000,
                     model_save_folder: Path = Constants.BOX_OFFICE_PREDICTION_MODEL_FOLDER,
                     model_save_name: str = Constants.BOX_OFFICE_PREDICTION_MODEL_NAME,
                     setting_save_path: Path = Constants.BOX_OFFICE_PREDICTION_SETTING_PATH,
                     scaler_save_path: Path = Constants.BOX_OFFICE_PREDICTION_SCALER_PATH,
                     training_week_limit: int = 4,
                     split_rate: float = 0.8) -> None:
        if input_data is None:
            train_data: list[list[MoviePredictionInputData]] = self.__generate_random_data(50, (4, 10), (0, 5))
            model_save_name = "gen_data"
        elif isinstance(input_data, list):
            train_data: list[list[MoviePredictionInputData]] = [self.__transform_single_movie_data(movie=movie) for
                                                                movie in input_data]
        elif isinstance(input_data, Path):
            train_data: list[list[MoviePredictionInputData]] = self._load_training_data(data_path=input_data)
        else:
            raise ValueError
        self.train(data=train_data, old_model_path=old_model_path, epoch=epoch, model_save_folder=model_save_folder,
                   model_save_name=model_save_name, setting_save_path=setting_save_path,
                   scaler_save_path=scaler_save_path, training_week_limit=training_week_limit, split_rate=split_rate)

    def simple_predict(self, input_data: MovieData | None) -> None:
        if input_data is None:
            test_data: list[MoviePredictionInputData] = self.__generate_random_data(1, (1, 20), (1, 100))[0]
        elif isinstance(input_data, MovieData):
            test_data = MoviePredictionModel.__transform_single_movie_data(input_data)
        else:
            raise ValueError
        print(self.predict(test_data))
        return

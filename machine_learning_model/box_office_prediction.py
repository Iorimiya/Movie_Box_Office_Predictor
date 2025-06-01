import yaml
import random
import numpy as np
from pathlib import Path
from logging import Logger

from numpy.typing import NDArray
from numpy import float32, float64, int64
from typing import Optional, TypedDict, Literal

from sklearn.preprocessing import MinMaxScaler
from joblib import dump as scaler_dump, load as scaler_load
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense, Masking, Input, Dropout
from keras.src.optimizers import Adam
from keras.src.optimizers.schedules import ExponentialDecay
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score

from tools.util import check_path, recreate_folder
from tools.constant import Constants
from tools.logging_manager import LoggingManager
from machine_learning_model.machine_learning_model import MachineLearningModel
from movie_data import MovieData, load_index_file, PublicReview, IndexLoadMode


class MoviePredictionInputData(TypedDict):
    box_office: int
    replies_count: int
    review_sentiment_score: list[bool]
    review_contents: list[str]


class MoviePredictionModel(MachineLearningModel):
    """
    A machine learning model for predicting movie box office revenue.

    This class utilizes an LSTM network to predict future box office revenue based on historical data,
    including box office performance, review sentiment, and engagement metrics.
    It handles data loading, preprocessing, model training, prediction, and evaluation.

    Attributes:
        __transform_scaler (Optional[MinMaxScaler]): Scaler for transforming the box office feature. Loaded from `transform_scaler_path`.
        __training_week_limit (Optional[int]): The number of past weeks used as input for prediction. Loaded from `training_setting_path`.
        __training_data_len (Optional[int]): The maximum sequence length of training data after padding. Loaded from `training_setting_path`.
        __split_ratios (Optional[float]): The ratio for splitting training data into training and testing sets. Loaded during training.
    """

    def __init__(self, model_path: Optional[Path] = None, transform_scaler_path: Optional[Path] = None,
                 training_setting_path: Optional[Path] = None) -> None:
        """
        Initializes the MoviePredictionModel.

        Args:
            model_path (Optional[Path]): Path to a pre-trained model file. If provided, the model will be loaded.
            transform_scaler_path (Optional[Path]): Path to a saved MinMaxScaler file. If provided, the scaler will be loaded.
            training_setting_path (Optional[Path]): Path to a YAML file containing training settings (training_data_len, training_week_limit).
        """
        super().__init__(model_path=model_path)
        self.__transform_scaler: Optional[MinMaxScaler] = scaler_load(transform_scaler_path) if check_path(
            transform_scaler_path) else None
        self.__training_week_limit: Optional[int] = None
        self.__training_data_len: Optional[int] = None
        self.__split_ratios: Optional[tuple[int, int, int]] = None
        self.__logger: Logger = LoggingManager().get_logger('machine_learning')
        if check_path(training_setting_path):
            self.__load_training_setting(training_setting_path)
        return

    def __save_training_setting(self, file_path: Path, encoding: str = Constants.DEFAULT_ENCODING) -> None:
        """
        Saves the training settings (training_data_len, training_week_limit) to a YAML file.

        Args:
           file_path (Path): The path to save the training settings file.
           encoding (str): The encoding to use when writing the file (default: Constants.DEFAULT_ENCODING).
        """
        self._check_save_folder(file_path.parent)
        yaml.Dumper.ignore_aliases = lambda self_, _: True
        with open(file_path, mode='w', encoding=encoding) as file:
            yaml.dump(
                {'training_data_len': self.__training_data_len, 'training_week_limit': self.__training_week_limit},
                file,
                allow_unicode=True)

    def __load_training_setting(self, file_path: Path, encoding: str = Constants.DEFAULT_ENCODING) -> None:
        """
        Loads the training settings (training_data_len, training_week_limit) from a YAML file.

        Args:
            file_path (Path): The path to the training settings file.
            encoding (str): The encoding to use when reading the file (default: Constants.DEFAULT_ENCODING).

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")
        with open(file_path, mode='r', encoding=encoding) as file:
            load_data: dict[str:int] = yaml.load(file, yaml.loader.BaseLoader)
        self.__training_data_len = int(load_data['training_data_len'])
        self.__training_week_limit = int(load_data['training_week_limit'])
        return

    def __save_scaler(self, file_path: Path) -> None:
        """
        Saves the MinMaxScaler used for scaling the box office feature.

        Args:
            file_path (Path): The path to save the scaler file.
        """
        self._check_save_folder(file_path.parent)
        scaler_dump(self.__transform_scaler, file_path)
        return

    def __save_all(self, base_folder: Path, model_name: str,
                   dataset: dict[str, NDArray[float32 | float64] | list[int]]) -> None:
        """
        Saves the trained model, test data, training settings, and scaler to the specified folder.

        Args:
            base_folder (Path): The base directory to save the files in.
            model_name (str): The name of the model to use in the saved filename.
            dataset (tuple[NDArray[float32], NDArray[float64], list[int]]): A tuple containing the test data:
                - x_test (NDArray[float32]): The input test data.
                - y_test (NDArray[float64]): The target test data.
                - lengths_test (list[int]): The original sequence lengths of the test data before padding.
        """
        recreate_folder(path=base_folder)
        # x_test, y_test, lengths_test = dataset
        self._save_model(file_path=base_folder.joinpath(f"{model_name}.keras"))

        for key, value in dataset.items():
            np.save(base_folder.joinpath(f"{key}.npy"), value)
        self.__save_training_setting(base_folder.joinpath('setting.yaml'))
        self.__save_scaler(base_folder.joinpath('scaler.gz'))

    @classmethod
    def _load_training_data(cls, data_path: Path) -> list[list[MoviePredictionInputData]]:
        """
        Loads movie data from an index file and transforms it into the model's input format.

        Args:
            data_path (Path): The path to the index file containing movie data.

        Returns:
            list[list[MoviePredictionInputData]]: A list of movies, where each movie is a list of
                                                 MoviePredictionInputData for each week.
        """
        LoggingManager().get_logger('root').info("Loading training data.")
        movie_data: list[MovieData] = load_index_file(file_path=data_path, mode=IndexLoadMode.FULL)
        training_data: list[list[MoviePredictionInputData]] = [cls.__transform_single_movie_data(movie=movie) for movie
                                                               in movie_data]
        return training_data

    @staticmethod
    def _load_set_data(set_data_folder_path: Path, set_type: Literal['validation', 'test']) -> Optional[
        tuple[NDArray[float32], NDArray[float64], NDArray[int64]]]:
        """
        Loads test data from the specified folder.

        Args:
            set_data_folder_path (Path): The directory path containing x_test.npy, y_test.npy and sequence_lengths.npy.

        Returns:
            Optional[tuple[NDArray[float32], NDArray[float64], NDArray[int64]]]:
                - x_test_loaded (NDArray[float32]): Loaded test input data.
                - y_test_loaded (NDArray[float64]): Loaded test target data.
                - lengths_test (NDArray[int64]): Loaded sequence lengths for test data.
                Returns None if loading fails.
        """
        dataset_name: str
        match set_type:
            case 'validation':
                dataset_name = "val"
            case 'test':
                dataset_name = "test"
            case _:
                raise ValueError(f"Invalid set type: {set_type}")
        try:
            x_loaded: NDArray[float32] = np.load(set_data_folder_path.joinpath(f"x_{dataset_name}.npy"))
            y_loaded: NDArray[float64] = np.load(set_data_folder_path.joinpath(f"y_{dataset_name}.npy"))
            lengths: NDArray[int64] = np.load(set_data_folder_path.joinpath(f"lengths_{dataset_name}.npy"))
            return x_loaded, y_loaded, lengths
        except FileNotFoundError:
            LoggingManager().get_logger('root').error(
                f"Could not find x_val.npy, y_val.npy or lengths_val.npy in '{set_data_folder_path}'.")
            return None

    @staticmethod
    def __generate_random_data(num_movies: int, weeks_range: tuple[int, int], reviews_range: tuple[int, int]) \
            -> list[list[MoviePredictionInputData]]:
        """
        Generates random movie prediction input data for testing purposes.

        Args:
            num_movies (int): The number of movies to generate data for.
            weeks_range (tuple[int, int]): The range (min, max) for the number of weeks of data per movie.
            reviews_range (tuple[int, int]): The range (min, max) for the number of reviews per week.

        Returns:
            list[list[MoviePredictionInputData]]: A list where each inner list represents a movie,
                                                 and contains a list of MoviePredictionInputData for each week.
        """
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
        """
        Transforms a single MovieData object into a list of MoviePredictionInputData.

        Args:
            movie (MovieData): The MovieData object to transform.

        Returns:
            list[MoviePredictionInputData]: A list of MoviePredictionInputData, where each element
                                            represents the data for a single week of the movie's performance.
        """
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
        """
        Preprocesses the raw input data into a numerical format suitable for the model.

        This function extracts 'box_office', the average of 'review_sentiment_score', and 'replies_count'
        for each week of each movie. Weeks with zero or negative box office are filtered out.

        Args:
            data (list[list[MoviePredictionInputData]]): The raw input data, a list of movies, where each movie
                                                        is a list of weekly MoviePredictionInputData.

        Returns:
            list[list[list[int | float]]]: A list of movies, where each movie is a list of weekly features
                                            (box_office, average_sentiment, replies_count).
        """
        return [[[week["box_office"], sum(week["review_sentiment_score"]) / len(week["review_sentiment_score"]) if week[
            "review_sentiment_score"] else 0, week["replies_count"]] for week in movie if week["box_office"] > 0] for
                movie in data]

    def __scale_box_office_feature(self, data: NDArray) -> NDArray:
        """
        Scales the box office feature using the loaded MinMaxScaler.

        Args:
            data (NDArray): A 2-dimensional NumPy array where the first column is the box office data.

        Returns:
            NDArray: A NumPy array with the box office feature scaled.

        Raises:
            ValueError: If the scaler has not been loaded or if the input data is not 2-dimensional.
        """
        if not self.__transform_scaler:
            raise ValueError("scaler must exist.")
        if data.ndim != 2:
            raise ValueError("Data inputted must have 2 dimensions.")
        scaled_data: NDArray = data.copy()
        scaled_data[:, 0] = self.__transform_scaler.transform(scaled_data[:, 0].reshape(-1, 1)).flatten()
        return scaled_data

    def _prepare_data(self, data: any) -> tuple[
        NDArray[float32], NDArray[float64], NDArray[float32], NDArray[float64], NDArray[float32], NDArray[float64],
        list[int], list[int], list[int]]:
        """
        Prepares the input data for training and testing.

        This function preprocesses the data, creates sequences for the LSTM model, pads the sequences to a uniform length,
        splits the data into training and testing sets, and scales the box office target variable.

        Args:
            data (any): The raw input data, typically a list of lists of MoviePredictionInputData.

        Returns:
            tuple[NDArray[float32], NDArray[float64], NDArray[float32], NDArray[float64]]: A tuple containing
                - x_train_scaled (NDArray[float32]): Scaled training input sequences.
                - y_train_scaled (NDArray[float64]): Scaled training target box office values.
                - x_test_scaled (NDArray[float32]): Scaled testing input sequences.
                - y_test_scaled (NDArray[float64]): Scaled testing target box office values.
                - lengths_train (List[int]): The original lengths of the training input sequences before padding.
                - lengths_test (List[int]): The original lengths of the testing input sequences before padding.
        """
        processed_data: list[list[list[int | float]]] = self.__preprocess_data(data)
        self.__training_data_len = max(len(movie) for movie in processed_data)

        # prepare_sequences
        x, y = [], []
        lengths: list[int] = []
        for movie in processed_data:
            for i in range(len(movie) - self.__training_week_limit):
                seq_x: list[list[int | float]] = movie[i:i + self.__training_week_limit]
                seq_y: int = movie[i + self.__training_week_limit][0]
                x.append(seq_x)
                y.append(seq_y)
                lengths.append(len(seq_x))

        if self.__training_data_len:
            x = pad_sequences(x, maxlen=self.__training_data_len, dtype='float32', padding='post')

        x, y = np.array(x), np.array(y)

        total_ratio_parts: int = sum(self.__split_ratios)
        if total_ratio_parts == 0:
            # This case implies all ratios are 0, which is problematic.
            # For safety, assign all to train, though this should ideally be caught by a check on split_ratios values.
            # print("Warning: Sum of split_ratios is zero. Assigning all data to training set.")
            train_end_idx = len(x)
            val_end_idx = len(x)
        else:
            train_ratio_part: int = self.__split_ratios[0]
            val_ratio_part: int = self.__split_ratios[1]
            num_samples: int = len(x)
            train_end_idx = int(np.floor(num_samples * train_ratio_part / total_ratio_parts))
            val_end_idx = int(np.floor(num_samples * (train_ratio_part + val_ratio_part) / total_ratio_parts))

            # Adjust to prevent empty validation set if val_ratio_part > 0 and rounding caused issues
            if val_ratio_part > 0 and train_end_idx == val_end_idx and num_samples > train_end_idx:
                if val_end_idx < num_samples:
                    val_end_idx += 1
                elif train_end_idx > 0:  # Can we decrement train_end_idx?
                    train_end_idx -= 1

            # Ensure train_end_idx is not after val_end_idx due to adjustments
            if train_end_idx > val_end_idx:
                train_end_idx = val_end_idx
        x_train: NDArray[float32] = x[:train_end_idx]
        y_train: NDArray[float64] = y[:train_end_idx]
        lengths_train: list[int] = lengths[:train_end_idx]

        x_val: NDArray[float32] = x[train_end_idx:val_end_idx]
        y_val: NDArray[float64] = y[train_end_idx:val_end_idx]
        lengths_val: list[int] = lengths[train_end_idx:val_end_idx]

        x_test: NDArray[float32] = x[val_end_idx:]
        y_test: NDArray[float64] = y[val_end_idx:]
        lengths_test: list[int] = lengths[val_end_idx:]

        # x_train, y_train, x_test, y_test = x[:split_index], y[:split_index], x[split_index:], y[split_index:]
        # lengths_train, lengths_test = lengths[:split_index], lengths[split_index:]

        # Standardization
        self.__transform_scaler: MinMaxScaler = MinMaxScaler()
        y_train_scaled: NDArray[float64] = self.__transform_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled: NDArray[float64] = self.__transform_scaler.transform(y_val.reshape(-1, 1)).flatten()
        y_test_scaled: NDArray[float64] = self.__transform_scaler.transform(y_test.reshape(-1, 1)).flatten()

        x_train_scaled: NDArray[float32] = x_train.copy()
        for i in range(x_train.shape[0]):
            x_train_scaled[i, :, 0] = self.__scale_box_office_feature(x_train[i, :, 0].reshape(-1, 1)).flatten()

        x_val_scaled: NDArray[float32] = x_val.copy()
        for i in range(x_val.shape[0]):
            x_val_scaled[i, :, 0] = self.__scale_box_office_feature(x_val[i, :, 0].reshape(-1, 1)).flatten()

        x_test_scaled: NDArray[float32] = x_test.copy()
        for i in range(x_test.shape[0]):
            x_test_scaled[i, :, 0] = self.__scale_box_office_feature(x_test[i, :, 0].reshape(-1, 1)).flatten()

        return x_train_scaled, y_train_scaled, x_val_scaled, y_val_scaled, x_test_scaled, y_test_scaled, lengths_train, lengths_val, lengths_test

    def _build_model(self, model: Sequential, layers: list) -> None:
        """
        Builds and compiles the Keras Sequential model.

        Args:
            model (Sequential): The Keras Sequential model instance.
            layers (list[any]): A list of Keras layers to add to the model.
        """
        super()._build_model(model=model, layers=layers)
        clip_norm_value: float = 1.0
        initial_learning_rate: float = 0.001
        decay_steps: int = 1000
        decay_rate: float = 0.96
        optimizer = Adam(
            learning_rate=ExponentialDecay(initial_learning_rate=initial_learning_rate, decay_steps=decay_steps,
                                           decay_rate=decay_rate), clipnorm=clip_norm_value)
        model.compile(optimizer=optimizer, loss='mse')

    def train(self, data: list[list[MoviePredictionInputData]],
              old_model_path: Optional[Path] = None,
              epoch: int | tuple[int, int] = 1000,
              model_name: str = Constants.BOX_OFFICE_PREDICTION_MODEL_NAME,
              training_week_limit: int = 4,
              split_ratios: tuple[int, int, int] = (8, 2, 2)) -> None:
        """
        Trains the movie box office prediction model.

        Args:
           data (list[list[MoviePredictionInputData]]): The training data, a list of movies, where each movie
                                                       is a list of weekly MoviePredictionInputData.
           old_model_path (Optional[Path]): Path to a pre-trained model to continue training from. Defaults to None.
           epoch (int): The number of training epochs. Defaults to 1000.
           model_name (str): The base name for the saved model file. Defaults to Constants.BOX_OFFICE_PREDICTION_MODEL_NAME.
           training_week_limit (int): The number of past weeks to use as input for prediction. Defaults to 4.
           split_ratios (tuple[int,int,int]): The ratio for splitting the data into training and testing sets. Defaults to 0.8.
        """
        self.__logger.info("Training procedure start.")
        self.__training_week_limit = training_week_limit
        self.__split_ratios = split_ratios
        x_train, y_train, x_val, y_val, x_test, y_test, lengths_train, lengths_val, lengths_test = \
            self._prepare_data(data)

        if old_model_path:
            self._model: Sequential = self._create_model(old_model_path=old_model_path)
            base_epoch: int = int(old_model_path.stem.split('_')[-1])
        else:
            self._model: Sequential = self._create_model(layers=[
                Input(shape=(self.__training_data_len, x_train.shape[2])),
                Masking(mask_value=0.0),
                LSTM(128, activation='relu'),
                Dropout(0.5),
                Dense(1)
            ])
            base_epoch: int = 0

        if isinstance(epoch, int):
            max_epoch, save_interval = epoch, epoch
        elif isinstance(epoch, tuple) and len(epoch) == 2 and all(isinstance(e, int) for e in epoch):
            max_epoch, save_interval = epoch
        else:
            raise ValueError("Epoch must be int or tuple[int,int]")

        for current_save_epoch in range(base_epoch + save_interval, max_epoch + save_interval, save_interval):
            save_name: str = f"{model_name}_{current_save_epoch}"
            self.train_model(x_train, y_train, save_interval)
            loss: float = self.evaluate_model(x_val, y_val)
            self.__logger.info(f"Model validation loss: {loss}.")

            base_save_folder: Path = Constants.BOX_OFFICE_PREDICTION_FOLDER.joinpath(f'{save_name}')

            # save training results
            self.__save_all(base_folder=base_save_folder, model_name=save_name,
                            dataset={
                                'x_train': x_train, 'y_train': y_train,
                                'x_val': x_val, 'y_val': y_val,
                                'x_test': x_test, 'y_test': y_test,
                                'lengths_train': lengths_train,
                                'lengths_val': lengths_val,
                                'lengths_test': lengths_test
                            })

    def predict(self, data_input: list[MoviePredictionInputData]) -> float:
        """
        Predicts the box office revenue for the next week based on the input data.

        Args:
            data_input (list[MoviePredictionInputData]): The input data for the movie, a list of
                                                       MoviePredictionInputData for the past weeks.

        Returns:
            float: The predicted box office revenue for the next week.

        Raises:
            ValueError: If the model, settings, or scaler have not been loaded.
        """
        if not self._model or not self.__transform_scaler or not self.__training_data_len or not self.__training_week_limit:
            raise ValueError('model, settings, and scaler must be loaded.')
        processed_input: list[list[int | float]] = self.__preprocess_data([data_input])[0]
        processed_input_array: NDArray[float64] = np.array(processed_input)
        scaled_input: NDArray[float64] = self.__scale_box_office_feature(processed_input_array[:, :1])
        processed_input_scaled: NDArray[float64] = processed_input_array.copy()
        processed_input_scaled[:, 0] = scaled_input.flatten()
        input_sequence: list[NDArray[float64]] = [processed_input_scaled[-self.__training_week_limit:]]
        input_sequence_padded: NDArray[float32] = pad_sequences(input_sequence, maxlen=self.__training_data_len,
                                                                dtype='float32',
                                                                padding='post')
        prediction_scaled: float = self._model.predict(input_sequence_padded)[0, 0]
        prediction: float = self.__transform_scaler.inverse_transform([[prediction_scaled]])[0, 0]
        return prediction

    def __evaluate_predictions(self, x_test_loaded: NDArray[float32], y_test_loaded: NDArray[float64],
                               lengths_test: list[int] | NDArray[int64], prediction_logic: callable) -> tuple[int, int]:
        """
        Evaluates predictions based on a given prediction logic.

        Args:
            x_test_loaded: Loaded test input data.
            y_test_loaded: Loaded test target data.
            lengths_test (list[int] | NDArray[int64]): A list containing the original lengths of each input sequence in `x_test_loaded` before padding.
            prediction_logic: A callable that takes predicted and actual box office values and returns True if the prediction is considered correct, False otherwise.

        Returns:
            A tuple containing the number of correct predictions and the total number of predictions.
        """
        correct_predictions: int = 0
        total_predictions: int = 0

        for i in range(len(x_test_loaded)):
            valid_len: int | int64 = lengths_test[i]
            if valid_len >= self.__training_week_limit:
                input_sequence: NDArray[float32] = x_test_loaded[i][:valid_len][-self.__training_week_limit:].reshape(
                    (1, self.__training_week_limit, x_test_loaded.shape[-1]))

                # Standardize the box office feature of the input sequence
                input_sequence_scaled: NDArray[float32] = input_sequence.copy()
                for j in range(input_sequence_scaled.shape[1]):
                    input_sequence_scaled[0, j, 0] = self.__transform_scaler.transform(
                        input_sequence[0, j, 0].reshape(-1, 1)).flatten()

                predicted_box_office_scaled: NDArray[float32] = self._model.predict(input_sequence_scaled, verbose=0)[
                    0, 0]
                predicted_box_office: float = \
                    self.__transform_scaler.inverse_transform([[predicted_box_office_scaled]])[0, 0]

                if i < len(y_test_loaded):
                    actual_next_week_box_office: float = \
                        self.__transform_scaler.inverse_transform([[y_test_loaded[i]]])[0, 0]
                    LoggingManager().get_logger('root').info(
                        f"Predicted box office / actual box office: {predicted_box_office} / {actual_next_week_box_office}.")

                    if prediction_logic(predicted_box_office, actual_next_week_box_office,
                                        x_test_loaded[i][valid_len - 1, 0]):
                        correct_predictions += 1
                    total_predictions += 1
                else:
                    LoggingManager().get_logger('root').warning(
                        "Length of x_test exceeds y_test, cannot determine actual value.")

        return correct_predictions, total_predictions

    @staticmethod
    def __log_and_print_evaluation_results(correct_predictions: int, total_predictions: int,
                                           evaluation_type: str) -> float:
        """
        Logs and prints the evaluation results.

        Args:
            correct_predictions (int): The number of correct predictions made by the model.
            total_predictions (int): The total number of predictions evaluated.
            evaluation_type (str): A string describing the type of evaluation being performed
                (e.g., "Trend", "Range"). This string will be included in the output messages.

        Returns:
            The accuracy of the model evaluation.
        """
        logger: Logger = LoggingManager().get_logger('root')
        accuracy: float = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"Correct prediction / Total prediction: {correct_predictions} / {total_predictions}.")
        logger.info(f"Correct prediction / Total prediction: {correct_predictions} / {total_predictions}.")
        print(f"{evaluation_type} prediction accuracy: {accuracy:.2%}.")
        logger.info(f"{evaluation_type} prediction accuracy: {accuracy:.2%}.")
        return accuracy

    def evaluate_loss(self,
                      test_data_folder_path: Path = Constants.BOX_OFFICE_PREDICTION_DEFAULT_MODEL_FOLDER) -> float:
        """
        Evaluates the model's test validation loss.

        Args:
            test_data_folder_path: The directory path containing x_test.npy and y_test.npy.

        Returns:
            the model's test validation loss.
        """
        if not self._model or not self.__transform_scaler or not self.__training_data_len or not self.__training_week_limit:
            raise AssertionError('model, settings, and scaler must be loaded.')
        x_test_loaded, y_test_loaded, lengths_test = self._load_set_data(test_data_folder_path, set_type='validation')
        return self.evaluate_model(x_test=x_test_loaded, y_test=y_test_loaded)

    def evaluate_trend(self,
                       test_data_folder_path: Path = Constants.BOX_OFFICE_PREDICTION_DEFAULT_MODEL_FOLDER) -> float:
        """
        Evaluates the model's accuracy in predicting the trend of box office revenue.

        Args:
            test_data_folder_path: The directory path containing x_test.npy and y_test.npy.

        Returns:
            The accuracy of the model evaluation.
        """
        if not self._model or not self.__transform_scaler or not self.__training_data_len or not self.__training_week_limit:
            raise AssertionError('model, settings, and scaler must be loaded.')
        x_test_loaded, y_test_loaded, lengths_test = self._load_set_data(test_data_folder_path, set_type='test')

        def trend_prediction_logic(predicted: float, actual: float, current: float) -> bool:
            predicted_trend = 1 if predicted > current else 0
            actual_trend = 1 if actual > current else 0
            return predicted_trend == actual_trend

        correct_predictions, total_predictions = self.__evaluate_predictions(x_test_loaded, y_test_loaded,
                                                                             lengths_test, trend_prediction_logic)
        accuracy: float = self.__log_and_print_evaluation_results(correct_predictions, total_predictions, "Trend")
        return accuracy

    def evaluate_range(self, box_office_ranges: tuple[int,] = (1000000, 10000000, 90000000),
                       test_data_folder_path: Path = Constants.BOX_OFFICE_PREDICTION_DEFAULT_MODEL_FOLDER) -> float:
        """
        Evaluates the model's prediction accuracy based on box office ranges.

        Args:
            box_office_ranges: A tuple of box office range boundaries, e.g., (100, 200, 400).
            test_data_folder_path: The directory path containing x_test.npy and y_test.npy.

        Returns:
            The accuracy of the model evaluation.
        """
        if not self._model or not self.__transform_scaler or not self.__training_data_len or not self.__training_week_limit:
            raise AssertionError('model, settings, and scaler must be loaded.')
        x_test_loaded, y_test_loaded, lengths_test = self._load_set_data(test_data_folder_path, set_type="test")

        # Create box office ranges
        ranges = sorted(list(box_office_ranges))
        thresholds = [-float('inf')] + ranges + [float('inf')]
        LoggingManager().get_logger('root').info(f"Box office ranges: {thresholds}.")

        def get_box_office_range_index(box_office: float) -> int:
            for i in range(len(thresholds) - 1):
                if thresholds[i] <= box_office < thresholds[i + 1]:
                    return i
            raise ValueError(f"Invalid value {box_office} of box office.")

        def range_prediction_logic(predicted: float, actual: float, _: any = None) -> bool:
            predicted_range_index = get_box_office_range_index(predicted)
            actual_range_index = get_box_office_range_index(actual)
            return predicted_range_index == actual_range_index

        correct_predictions, total_predictions = self.__evaluate_predictions(x_test_loaded, y_test_loaded,
                                                                             lengths_test, range_prediction_logic)
        accuracy: float = self.__log_and_print_evaluation_results(correct_predictions, total_predictions, "Range")
        return accuracy

    def calculate_f1_score_for_ranges(self,
                                      box_office_ranges: tuple[int, ...] = (1000000, 10000000, 90000000),
                                      test_data_folder_path: Path = Constants.BOX_OFFICE_PREDICTION_DEFAULT_MODEL_FOLDER,
                                      average_method: str = 'macro') -> float:
        """
        Calculates the F1 score for box office range predictions.

        Args:
            box_office_ranges: A tuple of box office range boundaries.
            test_data_folder_path: The directory path containing test data.
            average_method: The averaging method for F1 score calculation in a multi-class setting.
                            Common values: 'micro', 'macro', 'weighted', 'samples'.

        Returns:
            The calculated F1 score, or 0.0 if evaluation cannot be performed.
        """
        if not self._model or not self.__transform_scaler or \
                not self.__training_data_len or not self.__training_week_limit:
            self.__logger.error("Model, settings, and scaler must be loaded for F1 score calculation.")
            # 或者 raise ValueError(...)
            return 0.0

        loaded_data = self._load_set_data(test_data_folder_path, set_type='test')
        if loaded_data is None:
            self.__logger.error(f"Could not load test data from {test_data_folder_path} for F1 score calculation.")
            return 0.0
        x_test_loaded, y_test_loaded, lengths_test = loaded_data

        if x_test_loaded.size == 0 or y_test_loaded.size == 0:
            self.__logger.warning("Loaded test data (x_test or y_test) is empty. Cannot calculate F1 score.")
            return 0.0

        # --- 複用 evaluate_range 中的範圍定義邏輯 ---
        ranges = sorted(list(box_office_ranges))
        thresholds = [-float('inf')] + ranges + [float('inf')]
        self.__logger.info(f"Box office ranges for F1 score: {thresholds}.")

        def get_box_office_range_index(box_office: float) -> int:
            for i in range(len(thresholds) - 1):
                if thresholds[i] <= box_office < thresholds[i + 1]:
                    return i
            # 處理 box_office 可能等於 thresholds[-1] (inf) 的邊界情況
            if box_office == thresholds[-1] and thresholds[-1] == float('inf'):
                return len(thresholds) - 2  # 最後一個有效的範圍索引
            self.__logger.warning(
                f"Box office value {box_office} did not fall into defined ranges. Assigning to last valid range index.")
            return len(thresholds) - 2  # 如果值超出定義範圍，則分配給最後一個有效範圍

        y_true_labels = []
        y_pred_labels = []

        num_features = x_test_loaded.shape[-1] if x_test_loaded.ndim == 3 and x_test_loaded.shape[0] > 0 else 0
        if num_features == 0 and x_test_loaded.shape[0] > 0:
            self.__logger.warning("x_test_loaded has 0 features, cannot proceed with F1 calculation.")
            return 0.0

        for i in range(len(x_test_loaded)):
            # --- 與 __evaluate_predictions 中類似的預測邏輯 ---
            # x_test_loaded[i] 是一個已填充的序列
            # 其票房特徵 (索引 0) 已經被縮放
            # lengths_test[i] 是填充前的序列長度 (對於有效序列，通常是 self.__training_week_limit)

            input_for_prediction: NDArray[float32] = x_test_loaded[i].reshape(1, self.__training_data_len, num_features)

            predicted_box_office_scaled: float = self._model.predict(input_for_prediction, verbose=0)[0, 0]
            predicted_box_office_unscaled: float = \
                self.__transform_scaler.inverse_transform([[predicted_box_office_scaled]])[0, 0]

            if i < len(y_test_loaded):  # y_test_loaded 包含縮放後的目標值
                actual_next_week_box_office_scaled: float = float(y_test_loaded[i])
                actual_next_week_box_office_unscaled: float = \
                    self.__transform_scaler.inverse_transform([[actual_next_week_box_office_scaled]])[0, 0]

                try:
                    true_label = get_box_office_range_index(actual_next_week_box_office_unscaled)
                    pred_label = get_box_office_range_index(predicted_box_office_unscaled)
                    y_true_labels.append(true_label)
                    y_pred_labels.append(pred_label)
                except ValueError as e:
                    self.__logger.warning(f"Skipping sample {i} due to error in getting range index: {e}")
                    continue
            else:
                self.__logger.warning(f"Index {i} out of bounds for y_test_loaded (len: {len(y_test_loaded)}).")

        if not y_true_labels or not y_pred_labels:
            self.__logger.warning("No labels collected for F1 score calculation.")
            return 0.0

        # 計算 F1 分數
        # 注意：如果所有預測都屬於一個類別，而真實標籤中沒有該類別，或者反之，
        # sklearn 會拋出警告並將該類別的 F1 設為 0。
        # zero_division=0 參數可以在 precision, recall 或 F-score 為 0 時避免警告，並返回 0。
        f1 = f1_score(y_true_labels, y_pred_labels, average=average_method, zero_division=0)
        self.__logger.info(f"F1 Score (average='{average_method}') for ranges: {f1:.4f}")
        self.__logger.info(f"Range labels true: {y_true_labels[:20]}...")  # Log a sample
        self.__logger.info(f"Range labels pred: {y_pred_labels[:20]}...")  # Log a sample

        # 您也可以計算並記錄每個類別的 F1 分數 (如果 average='macro' 或 'weighted')
        if average_method in ['macro', 'weighted', None]:  # 'None' returns per-class scores
            per_class_f1 = f1_score(y_true_labels, y_pred_labels, average=None, zero_division=0)
            unique_labels = sorted(list(set(y_true_labels) | set(y_pred_labels)))  # 所有出現過的標籤
            for label_idx, class_f1 in enumerate(per_class_f1):
                # 嘗試將標籤索引映射回更有意義的範圍描述 (如果需要)
                # 這裡的 label_idx 可能不直接對應 thresholds 的索引，取決於 unique_labels 的內容
                # 為了簡單起見，我們先用 unique_labels 中的標籤
                if label_idx < len(unique_labels):
                    class_label_for_log = unique_labels[label_idx]
                    self.__logger.info(f"  F1 Score for range class '{class_label_for_log}': {class_f1:.4f}")
                else:  # 理論上不應該發生
                    self.__logger.info(f"  F1 Score for class index {label_idx}: {class_f1:.4f}")

        return f1

    def simple_train(self, input_data: Path | list[MovieData] | None,
                     old_model_path: Optional[Path] = None,
                     epoch: int | tuple[int, int] = 1000,
                     model_name: str = Constants.BOX_OFFICE_PREDICTION_MODEL_NAME,
                     training_week_limit: int = 4,
                     split_ratios: tuple[int, int, int] = (8, 2, 2)) -> None:
        """
        A simplified training function that can accept different types of input data.

        Args:
            input_data (Path | list[MovieData] | None): The input training data.
                - If Path: Path to an index file containing movie data.
                - If list[MovieData]: A list of MovieData objects.
                - If None: Generates random training data for testing.
            old_model_path (Optional[Path]): Path to a pre-trained model to continue training from. Defaults to None.
            epoch (int): The number of training epochs. Defaults to 1000.
            model_name (str): The base name for the saved model file. Defaults to Constants.BOX_OFFICE_PREDICTION_MODEL_NAME.
            training_week_limit (int): The number of past weeks to use as input for prediction. Defaults to 4.
            split_ratios (float): The ratio for splitting the data into training and testing sets. Defaults to 0.8.
        """
        if input_data is None:
            train_data: list[list[MoviePredictionInputData]] = self.__generate_random_data(100, (4, 30), (0, 200))
            model_name = "gen_data"
        elif isinstance(input_data, list):
            train_data: list[list[MoviePredictionInputData]] = [self.__transform_single_movie_data(movie=movie) for
                                                                movie in input_data]
        elif isinstance(input_data, Path):
            train_data: list[list[MoviePredictionInputData]] = self._load_training_data(data_path=input_data)
        else:
            raise ValueError
        self.train(data=train_data, old_model_path=old_model_path, epoch=epoch, model_name=model_name,
                   training_week_limit=training_week_limit, split_ratios=split_ratios)

    def simple_predict(self, input_data: MovieData | None) -> None:
        """
        A simplified prediction function that can accept different types of input data.

        Args:
            input_data (MovieData | None): The input data for prediction.
                - If MovieData: A MovieData object.
                - If None: Generates random test data for prediction.
        """
        if input_data is None:
            test_data: list[MoviePredictionInputData] = self.__generate_random_data(1, (1, 20), (1, 100))[0]
        elif isinstance(input_data, MovieData):
            test_data = MoviePredictionModel.__transform_single_movie_data(input_data)
        else:
            raise ValueError
        print(self.predict(test_data))
        return

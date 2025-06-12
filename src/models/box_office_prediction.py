import yaml
import random
import numpy as np
from pathlib import Path
from logging import Logger
from numpy.typing import NDArray
from numpy import float32, float64, int64
from typing import Optional, TypedDict

from sklearn.preprocessing import MinMaxScaler
from joblib import dump as scaler_dump, load as scaler_load
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense, Masking, Input, Dropout
from keras.src.optimizers import Adam
from keras.src.optimizers.schedules import ExponentialDecay
from keras_preprocessing.sequence import pad_sequences

from src.utilities.util import check_path, recreate_folder
from src.core.constants import Constants
from src.core.logging_manager import LoggingManager
from src.models.machine_learning_model import MachineLearningModel
from src.data_handling.movie_data import MovieData, load_index_file, PublicReview, IndexLoadMode


class MoviePredictionInputData(TypedDict):
    """A TypedDict representing the input data structure for a single week of a movie's performance.

    This structure is used as an intermediate representation when transforming raw movie data
    into a format suitable for the prediction model.

    :ivar box_office: The box office revenue for the week.
    :ivar replies_count: The total number of replies to reviews for the week.
    :ivar review_sentiment_score: A list of boolean values representing the sentiment
                                  (e.g., True for positive, False for negative) of individual reviews for the week.
    :ivar review_contents: A list of strings, where each string is the content of a review for the week.
    """
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

    :ivar __transform_scaler: Scaler for transforming the box office feature. Loaded from ``transform_scaler_path``.
    :ivar __training_week_limit: The number of past weeks used as input for prediction. Loaded from ``training_setting_path``.
    :ivar __training_data_len: The maximum sequence length of training data after padding. Loaded from ``training_setting_path``.
    :ivar __split_rate: The ratio for splitting training data into training and testing sets. Set during the ``train`` method.
    :ivar __logger: Logger instance for logging messages.
    """

    def __init__(self, model_path: Optional[Path] = None, transform_scaler_path: Optional[Path] = None,
                 training_setting_path: Optional[Path] = None) -> None:
        """
        Initializes the MoviePredictionModel.

        :param model_path: Path to a pre-trained Keras model file. If provided, the model will be loaded.
        :param transform_scaler_path: Path to a saved ``MinMaxScaler`` file. If provided, the scaler will be loaded.
        :param training_setting_path: Path to a YAML file containing training settings
                                      (e.g., ``training_data_len``, ``training_week_limit``).
        """
        super().__init__(model_path=model_path)
        self.__transform_scaler: Optional[MinMaxScaler] = scaler_load(transform_scaler_path) if check_path(
            transform_scaler_path) else None
        self.__training_week_limit: Optional[int] = None
        self.__training_data_len: Optional[int] = None
        self.__split_rate: Optional[float] = None
        self.__logger: Logger = LoggingManager().get_logger('machine_learning')
        if check_path(training_setting_path):
            self.__load_training_setting(training_setting_path)
        return

    def __save_training_setting(self, file_path: Path, encoding: str = Constants.DEFAULT_ENCODING) -> None:
        """
        Saves the training settings (``training_data_len``, ``training_week_limit``) to a YAML file.

        :param file_path: The path to save the training settings file.
        :param encoding: The encoding to use when writing the file.
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
        Loads the training settings (``training_data_len``, ``training_week_limit``) from a YAML file.

        :param file_path: The path to the training settings file.
        :param encoding: The encoding to use when reading the file.
        :raises FileNotFoundError: If the specified file does not exist.
        :raises yaml.YAMLError: If there is an error parsing the YAML file.
        :raises KeyError: If expected keys (``training_data_len``, ``training_week_limit``) are missing from the loaded data.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")
        with open(file_path, mode='r', encoding=encoding) as file:
            load_data: dict[str, int] = yaml.load(file, yaml.loader.BaseLoader)
        self.__training_data_len = int(load_data['training_data_len'])
        self.__training_week_limit = int(load_data['training_week_limit'])
        return

    def __save_scaler(self, file_path: Path) -> None:
        """
        Saves the ``MinMaxScaler`` (``self.__transform_scaler``) used for scaling the box office feature.

        :param file_path: The path to save the scaler file.
        :raises AttributeError: If ``self.__transform_scaler`` is ``None``.
        :raises Exception: For potential I/O errors during file writing or folder creation.
        """
        self._check_save_folder(file_path.parent)
        scaler_dump(self.__transform_scaler, file_path)
        return

    def __save_all(self, base_folder: Path, model_name: str,
                   test_data: tuple[NDArray[float32], NDArray[float64], list[int]]) -> None:
        """
        Saves the trained model, test data, training settings, and scaler to the specified folder.

        The target folder will be recreated if it already exists.

        :param base_folder: The base directory to save the files in.
        :param model_name: The name of the model to use in the saved Keras model filename.
        :param test_data: A tuple containing the test data:
                          - x_test: The input test data.
                          - y_test: The target test data.
                          - lengths_test: The original sequence lengths of the test data before padding.
        :raises Exception: For potential I/O errors during file or folder operations.
        """
        recreate_folder(path=base_folder)
        x_test, y_test, lengths_test = test_data
        self._save_model(file_path=base_folder.joinpath(f"{model_name}.keras"))
        np.save(base_folder.joinpath('x_test.npy'), x_test)
        np.save(base_folder.joinpath('y_test.npy'), y_test)
        np.save(base_folder.joinpath('sequence_lengths.npy'), lengths_test)
        self.__save_training_setting(base_folder.joinpath('setting.yaml'))
        self.__save_scaler(base_folder.joinpath('scaler.gz'))

    @classmethod
    def _load_training_data(cls, data_path: Path) -> list[list[MoviePredictionInputData]]:
        """
        Loads movie data from an index file and transforms it into the model's input format.

        :param data_path: The path to the index file containing movie data.
        :returns: A list of movies, where each movie is a list of ``MoviePredictionInputData`` for each week.
        :raises FileNotFoundError: If the ``data_path`` does not exist.
        :raises Exception: For other potential errors during file loading or data transformation.
        """
        LoggingManager().get_logger('root').info("Loading training data.")
        movie_data: list[MovieData] = load_index_file(file_path=data_path, mode=IndexLoadMode.FULL)
        training_data: list[list[MoviePredictionInputData]] = [cls.__transform_single_movie_data(movie=movie) for movie
                                                               in movie_data]
        return training_data

    @staticmethod
    def _load_test_data(test_data_folder_path: Path) -> Optional[
        tuple[NDArray[float32], NDArray[float64], NDArray[int64]]]:
        """
        Loads test data (x_test, y_test, lengths_test) from NumPy files within the specified folder.

        File names are expected to be 'x_test.npy', 'y_test.npy', and 'sequence_lengths.npy'.

        :param test_data_folder_path: The directory path containing the test data NumPy files.
        :returns: A tuple containing the loaded x_test, y_test, and lengths_test arrays,
                  or ``None`` if any of the files are not found.
        """
        try:
            x_test_loaded: NDArray[float32] = np.load(test_data_folder_path.joinpath("x_test.npy"))
            y_test_loaded: NDArray[float64] = np.load(test_data_folder_path.joinpath("y_test.npy"))
            lengths_test: NDArray[int64] = np.load(test_data_folder_path.joinpath("sequence_lengths.npy"))
            return x_test_loaded, y_test_loaded, lengths_test
        except FileNotFoundError:
            LoggingManager().get_logger('root').error(
                f"Could not find x_test.npy, y_test.npy or sequence_lengths.npy in '{test_data_folder_path}'.")
            return None

    @staticmethod
    def __generate_random_data(num_movies: int, weeks_range: tuple[int, int], reviews_range: tuple[int, int]) \
        -> list[list[MoviePredictionInputData]]:
        """
        Generates random movie prediction input data for testing or demonstration purposes.

        :param num_movies: The number of movies to generate data for.
        :param weeks_range: A tuple (min_weeks, max_weeks) specifying the range for the number of weeks of data per movie.
        :param reviews_range: A tuple (min_reviews, max_reviews) specifying the range for the number of reviews per week.
        :returns: A list where each inner list represents a movie,
                  and contains a list of ``MoviePredictionInputData`` for each week.
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
        Transforms a single ``MovieData`` object into a list of ``MoviePredictionInputData`` objects,
        one for each week of the movie's box office performance.

        :param movie: The ``MovieData`` object to transform.
        :returns: A list of ``MoviePredictionInputData``, where each element
                  represents the aggregated data for a single week.
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
        Preprocesses the raw input data (list of movies, each with weekly ``MoviePredictionInputData``)
        into a numerical format suitable for the model.

        This function extracts 'box_office', the average of 'review_sentiment_score' (as a float 0-1),
        and 'replies_count' for each week of each movie.
        Weeks with zero or negative box office are filtered out.
        Movies with no valid weeks after filtering are excluded.

        :param data: The raw input data, a list of movies, where each movie
                     is a list of weekly ``MoviePredictionInputData``.
        :returns: A list of movies, where each movie is a list of weekly features.
                  Each weekly feature list contains [box_office, average_sentiment, replies_count].
                  Returns an empty list if no movies have valid data after processing.
        """
        return [[[week["box_office"], sum(week["review_sentiment_score"]) / len(week["review_sentiment_score"]) if week[
            "review_sentiment_score"] else 0, week["replies_count"]] for week in movie if week["box_office"] > 0] for
                movie in data]

    def __scale_box_office_feature(self, data: NDArray) -> NDArray:
        """
        Scales the box office feature (assumed to be the first column) of the input data
        using the model's ``MinMaxScaler`` (``self.__transform_scaler``).

        :param data: A 2-dimensional NumPy array where the first column is the box office data to be scaled.
        :returns: A NumPy array with the box office feature scaled.
        :raises ValueError: If ``self.__transform_scaler`` has not been loaded/initialized,
                            or if the input ``data`` is not 2-dimensional.
        :raises AttributeError: If ``self.__transform_scaler`` is ``None``.
        """
        if not self.__transform_scaler:
            raise ValueError("scaler must exist.")
        if data.ndim != 2:
            raise ValueError("Data inputted must have 2 dimensions.")
        scaled_data: NDArray = data.copy()
        scaled_data[:, 0] = self.__transform_scaler.transform(scaled_data[:, 0].reshape(-1, 1)).flatten()
        return scaled_data

    def _prepare_data(self, data: any) -> tuple[
        NDArray[float32], NDArray[float64], NDArray[float32], NDArray[float64], list[int], list[int]]:
        """
        Prepares the input data for training and testing the LSTM model.

        Steps include:
        1. Preprocessing the data using ``__preprocess_data``.
        2. Determining ``self.__training_data_len`` based on the longest movie sequence.
        3. Creating input sequences (x) and target values (y) based on ``self.__training_week_limit``.
        4. Padding input sequences to ``self.__training_data_len``.
        5. Splitting the data into training and testing sets based on ``self.__split_rate``.
        6. Initializing and fitting ``self.__transform_scaler`` on the training target (y_train)
           and then scaling y_train and y_test.
        7. Scaling the box office feature (first feature) in x_train and x_test using the same scaler.

        :param data: The raw input data, typically a list of lists of ``MoviePredictionInputData``.
        :returns: A tuple containing:
                  - x_train_scaled: Scaled training input sequences.
                  - y_train_scaled: Scaled training target box office values.
                  - x_test_scaled: Scaled testing input sequences.
                  - y_test_scaled: Scaled testing target box office values.
                  - lengths_train: The original lengths of the training input sequences before padding.
                  - lengths_test: The original lengths of the testing input sequences before padding.
        :raises ValueError: If ``self.__training_week_limit`` or ``self.__split_rate`` is not set,
                            or if no sequences can be generated.
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

        split_index: int = int(len(x) * self.__split_rate)
        x_train, y_train, x_test, y_test = x[:split_index], y[:split_index], x[split_index:], y[split_index:]
        lengths_train, lengths_test = lengths[:split_index], lengths[split_index:]

        # Standardization
        self.__transform_scaler: MinMaxScaler = MinMaxScaler()
        y_train_scaled: NDArray[float64] = self.__transform_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled: NDArray[float64] = self.__transform_scaler.transform(y_test.reshape(-1, 1)).flatten()

        x_train_scaled: NDArray[float32] = x_train.copy()
        for i in range(x_train.shape[0]):
            x_train_scaled[i, :, 0] = self.__scale_box_office_feature(x_train[i, :, 0].reshape(-1, 1)).flatten()

        x_test_scaled: NDArray[float32] = x_test.copy()
        for i in range(x_test.shape[0]):
            x_test_scaled[i, :, 0] = self.__scale_box_office_feature(x_test[i, :, 0].reshape(-1, 1)).flatten()

        return x_train_scaled, y_train_scaled, x_test_scaled, y_test_scaled, lengths_train, lengths_test

    def _build_model(self, model: Sequential, layers: list) -> None:
        """
        Builds and compiles the Keras ``Sequential`` model for box office prediction.

        The model is compiled with an Adam optimizer (using ``ExponentialDecay`` for the learning rate
        and ``clipnorm``) and 'mse' (mean squared error) as the loss function.

        :param model: The Keras ``Sequential`` model instance to build upon.
        :param layers: A list of Keras layers to add to the model. This parameter is passed
                       from the superclass but typically, for this specific model, layers are defined
                       directly in the ``train`` method if a new model is created.
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
              split_rate: float = 0.8) -> None:
        """
        Trains the movie box office prediction model.

        If ``old_model_path`` is provided, training continues from that model. Otherwise, a new
        LSTM-based model is created. The model is trained in intervals, and after each
        interval, its performance is evaluated on the test set, and the model along with
        associated data (test set, settings, scaler) is saved.

        :param data: The training data, a list of movies, where each movie
                     is a list of weekly ``MoviePredictionInputData``.
        :param old_model_path: Path to a pre-trained model to continue training from.
                               If provided, the epoch number is inferred from its filename.
        :param epoch: The total number of training epochs or a tuple (max_epoch, save_interval).
                      If an int, it's treated as (max_epoch, max_epoch), meaning one save at the end.
        :param model_name: The base name for the saved model file (epoch number will be appended).
        :param training_week_limit: The number of past weeks to use as input for prediction.
        :param split_rate: The ratio for splitting the data into training and testing sets (e.g., 0.8 for 80% train).
        :raises ValueError: If ``epoch`` is not an int or a tuple of two ints.
        """
        self.__logger.info("Training procedure start.")
        self.__training_week_limit = training_week_limit
        self.__split_rate = split_rate
        x_train, y_train, x_test, y_test, _, lengths_test = self._prepare_data(data)

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
            loss: float = self.evaluate_model(x_test, y_test)
            self.__logger.info(f"Model validation loss: {loss}.")

            base_save_folder: Path = Constants.BOX_OFFICE_PREDICTION_FOLDER.joinpath(f'{save_name}')

            # save training results
            self.__save_all(base_folder=base_save_folder, model_name=save_name,
                            test_data=(x_test, y_test, lengths_test))

    def predict(self, data_input: list[MoviePredictionInputData]) -> float:
        """
        Predicts the box office revenue for the next week based on the input data.

        The input data is preprocessed, scaled, and padded before being fed to the model.
        The predicted scaled value is then inverse-transformed to get the actual predicted box office value.

        :param data_input: The input data for the movie, a list of ``MoviePredictionInputData``
                           representing the performance of past weeks.
        :returns: The predicted box office revenue for the next week.
        :raises ValueError: If the model (``self._model``), settings (``self.__training_data_len``,
                            ``self.__training_week_limit``), or scaler (``self.__transform_scaler``)
                            have not been loaded or initialized.
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
        Evaluates predictions on the loaded test set based on a given custom prediction logic.

        This method iterates through the test samples, makes predictions, and uses the
        ``prediction_logic`` function to determine if each prediction is "correct".
        It handles scaling and unscaling of values as needed for the prediction and logic function.

        :param x_test_loaded: Loaded and scaled test input data (features).
        :param y_test_loaded: Loaded and scaled test target data (labels).
        :param lengths_test: A list or array containing the original lengths of each input sequence
                               in ``x_test_loaded`` before padding. This is used to extract the
                               correct portion of the sequence for prediction if ``self.__training_week_limit``
                               is different from the padded length.
        :param prediction_logic: A callable that takes (predicted_unscaled, actual_unscaled, current_unscaled_from_input)
                                 and returns ``True`` if the prediction is considered correct, ``False`` otherwise.
                                 The 'current_unscaled_from_input' is the unscaled box office value from the
                                 last week of the input sequence.
        :returns: A tuple containing the number of correct predictions and the total number of predictions made.
        :raises ValueError: If model, settings, or scaler are not loaded/initialized.
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
        Logs and prints the evaluation results (count and accuracy).

        :param correct_predictions: The number of correct predictions made by the model.
        :param total_predictions: The total number of predictions evaluated.
        :param evaluation_type: A string describing the type of evaluation being performed
                                (e.g., "Trend", "Range"), used in log/print messages.
        :returns: The accuracy of the model evaluation (correct_predictions / total_predictions).
                  Returns 0.0 if ``total_predictions`` is 0.
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
        Evaluates the model's loss on the test set.

        Loads test data from the specified folder and uses ``self.evaluate_model``
        (which calls Keras ``model.evaluate``) to compute the loss.

        :param test_data_folder_path: The directory path containing 'x_test.npy' and 'y_test.npy'.
        :returns: The model's loss on the test set.
        :raises AssertionError: If model, settings, or scaler are not loaded/initialized.
                                (Note: ``evaluate_model`` itself might raise if ``self._model`` is None).
        :raises FileNotFoundError: If test data files are not found in ``test_data_folder_path``.
        """
        if not self._model or not self.__transform_scaler or not self.__training_data_len or not self.__training_week_limit:
            raise AssertionError('model, settings, and scaler must be loaded.')
        x_test_loaded, y_test_loaded, lengths_test = self._load_test_data(test_data_folder_path)
        return self.evaluate_model(x_test=x_test_loaded, y_test=y_test_loaded)

    def evaluate_trend(self,
                       test_data_folder_path: Path = Constants.BOX_OFFICE_PREDICTION_DEFAULT_MODEL_FOLDER) -> float:
        """
        Evaluates the model's accuracy in predicting the trend (increase or decrease)
        of box office revenue compared to the last week of the input sequence.

        :param test_data_folder_path: The directory path containing 'x_test.npy', 'y_test.npy',
                                     and 'sequence_lengths.npy'.
        :returns: The trend prediction accuracy.
        :raises AssertionError: If model, settings, or scaler are not loaded/initialized.
        :raises FileNotFoundError: If test data files are not found in ``test_data_folder_path``.
        """
        if not self._model or not self.__transform_scaler or not self.__training_data_len or not self.__training_week_limit:
            raise AssertionError('model, settings, and scaler must be loaded.')
        x_test_loaded, y_test_loaded, lengths_test = self._load_test_data(test_data_folder_path)

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
        Evaluates the model's prediction accuracy based on whether the predicted and actual
        box office values fall into the same predefined monetary range.

        :param box_office_ranges: A tuple of integers defining the upper boundaries of box office ranges.
                                  For example, ``(1M, 10M, 90M)`` creates ranges:
                                  <1M, 1M to <10M, 10M to <90M, >=90M.
        :param test_data_folder_path: The directory path containing 'x_test.npy', 'y_test.npy',
                                      and 'sequence_lengths.npy'.
        :returns: The range prediction accuracy.
        :raises AssertionError: If model, settings, or scaler are not loaded/initialized.
        :raises FileNotFoundError: If test data files are not found in ``test_data_folder_path``.
        :raises ValueError: If a box office value in ``get_box_office_range_index`` does not fall into any defined range
                            (should be rare if ranges cover -inf to inf).
        """
        if not self._model or not self.__transform_scaler or not self.__training_data_len or not self.__training_week_limit:
            raise AssertionError('model, settings, and scaler must be loaded.')
        x_test_loaded, y_test_loaded, lengths_test = self._load_test_data(test_data_folder_path)

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

    def simple_train(self, input_data: Path | list[MovieData] | None,
                     old_model_path: Optional[Path] = None,
                     epoch: int | tuple[int, int] = 1000,
                     model_name: str = Constants.BOX_OFFICE_PREDICTION_MODEL_NAME,
                     training_week_limit: int = 4,
                     split_rate: float = 0.8) -> None:
        """
        A simplified training function that accepts different types of input data.

        It handles data loading/generation based on ``input_data`` type and then calls the main ``train`` method.

        :param input_data: The input training data.
                           - If ``Path``: Path to an index file containing movie data.
                           - If ``list[MovieData]``: A list of ``MovieData`` objects.
                           - If ``None``: Generates random training data for testing.
        :param old_model_path: Path to a pre-trained model to continue training from.
        :param epoch: The number of training epochs or a tuple (max_epoch, save_interval).
        :param model_name: The base name for the saved model file. If ``input_data`` is ``None``,
                           this is overridden to "gen_data".
        :param training_week_limit: The number of past weeks to use as input for prediction.
        :param split_rate: The ratio for splitting the data into training and testing sets.
        :raises ValueError: If ``input_data`` is of an unexpected type.
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
                   training_week_limit=training_week_limit, split_rate=split_rate)

    def simple_predict(self, input_data: MovieData | None) -> None:
        """
        A simplified prediction function that accepts different types of input data and prints the prediction.

        :param input_data: The input data for prediction.
                           - If ``MovieData``: A ``MovieData`` object.
                           - If ``None``: Generates random test data for prediction.
        :raises ValueError: If ``input_data`` is of an unexpected type, or if the model/settings/scaler
                            are not loaded when ``self.predict`` is called.
        """
        if input_data is None:
            test_data: list[MoviePredictionInputData] = self.__generate_random_data(1, (1, 20), (1, 100))[0]
        elif isinstance(input_data, MovieData):
            test_data = MoviePredictionModel.__transform_single_movie_data(input_data)
        else:
            raise ValueError
        print(self.predict(test_data))
        return

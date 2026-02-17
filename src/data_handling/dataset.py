from dataclasses import dataclass, field, replace
from logging import Logger
from pathlib import Path
from time import sleep
from typing import Final, Literal, Optional

from tqdm import tqdm

from src.core.logging_manager import LoggingManager
from src.core.project_config import ProjectDatasetType, ProjectPaths
from src.data_collection.box_office_collector import BoxOfficeCollector
from src.data_collection.review_collector import ReviewCollector, TargetWebsite
from src.data_handling.file_io import CsvFile
from src.data_handling.movie_collections import MovieData, MovieSessionData
from data_handling.repositories.repository import MovieRepository
from src.data_handling.reviews import PublicReview
from data_handling.repositories.yaml_repository import YamlMovieRepository
from src.sentiment_analysis.llm_client import DailyRateLimitExceededError, LLMClient, LLMProvider


@dataclass(kw_only=True)
class Dataset:
    """
    Manages a collection of movie data, including metadata, box office figures, and reviews.

    Provides methods to load, initialize, and collect data for a named dataset.
    It handles interactions with data collectors and the underlying repository.

    :ivar name: The unique name of the dataset.
    :ivar repository: The repository used for data access.
    :ivar __movies_data_cache: An internal cache for the fully loaded list of MovieData objects.
                              It is initialized to None and populated on first access to `movie_data` property.
                              The cache is invalidated when data collection methods are called.
    :ivar __logger: A logger instance for logging messages.
    """
    name: str
    repository: MovieRepository = field(init=False)
    __movies_data_cache: Optional[list[MovieData]] = field(default=None, init=False, repr=False)
    __logger: Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Performs post-initialization setup.

        Initializes the logger and sets up the default YamlMovieRepository.
        """
        self.__logger = LoggingManager().get_logger('root')
        # Default to YAML repository for backward compatibility and file-based operations
        dataset_path = ProjectPaths.get_dataset_path(dataset_name=self.name, dataset_type=ProjectDatasetType.STRUCTURED)
        self.repository = YamlMovieRepository(dataset_root_path=dataset_path)

    @property
    def dataset_path(self) -> Path:
        """
        The root path for this dataset's files.
        """
        return ProjectPaths.get_dataset_path(dataset_name=self.name, dataset_type=ProjectDatasetType.STRUCTURED)

    @property
    def index_file_path(self) -> Path:
        """
        The path to the index CSV file for this dataset.
        """
        return self.dataset_path / ProjectPaths.INDEX_FILE_NAME

    @property
    def box_office_folder_path(self) -> Path:
        """
        The path to the folder containing box office data files for this dataset.
        """
        return self.dataset_path / ProjectPaths.BOX_OFFICE_SUBFOLDER_NAME

    @property
    def public_review_folder_path(self) -> Path:
        """
        The path to the folder containing public review data files for this dataset.
        """
        return self.dataset_path / ProjectPaths.PUBLIC_REVIEWS_SUBFOLDER_NAME

    @property
    def expert_review_folder_path(self) -> Path:
        """
        The path to the folder containing expert review data files for this dataset.
        """
        return self.dataset_path / ProjectPaths.EXPERT_REVIEWS_SUBFOLDER_NAME

    @property
    def index_file(self) -> CsvFile:
        """
        A CsvFile instance for interacting with the dataset's index file.
        """
        return CsvFile(path=self.index_file_path)

    @property
    def movie_data(self) -> list[MovieData]:
        """
        A list of fully populated MovieData objects for the dataset.

        This property uses an internal cache (`_movies_data_cache`).
        On first access, it loads all movie data (metadata, box office, reviews)
        using the repository and caches the result.
        Subsequent accesses return the cached list.
        The cache is invalidated by data collection methods.

        :returns: A list of MovieData objects.
        """
        if self.__movies_data_cache is None:
            self.__logger.debug(f"Cache miss for 'movie_data' in dataset '{self.name}'. Loading all movie data.")
            self.__movies_data_cache = self.repository.fetch_movies(detail_level='ALL')
            self.__logger.debug(
                f"Populated 'movie_data' cache for dataset '{self.name}' with {len(self.__movies_data_cache)} items.")
        else:
            self.__logger.debug(
                f"Returning cached 'movie_data' for dataset '{self.name}' with {len(self.__movies_data_cache)} items.")
        return self.__movies_data_cache

    def initialize_index_file(self, source_csv: CsvFile) -> None:
        """
        Initializes or overwrites the dataset's index file from a source CSV file.

        It reads movie names from the 'movie_name' column of the source CSV,
        assigns a sequential ID (starting from 0), and saves this new
        index data (id, name) to the dataset's `index.csv` file.

        :param source_csv: A CsvFile instance representing the source CSV file
                           containing at least a 'movie_name' column.
        :raises (FileNotFoundError, PermissionError, IOError): If an I/O error occurs
                                                                during file operations.
        :raises Exception: For any other unexpected errors during initialization.
        """
        self.__logger.info(
            f"Initializing index file '{self.index_file_path}' for dataset '{self.name}' from source '{source_csv.path}'.")
        try:
            source_data: list[dict[str, str]] = source_csv.load()
            if not source_data:
                self.__logger.warning(
                    f"Source CSV file '{source_csv.path}' is empty. Index file will not be initialized with data.")
                self.index_file.save(data=[])
                return

            index_data: list[dict[str, str]] = []
            for index, movie_row in enumerate(source_data):
                movie_name: Optional[str] = movie_row.get('movie_name')
                if movie_name is None:
                    self.__logger.warning(
                        f"Row {index + 1} in source CSV '{source_csv.path}' is missing 'movie_name'. Skipping.")
                    continue
                index_data.append({'id': str(index), 'name': movie_name})

            self.index_file.save(data=index_data)
            self.__logger.info(
                f"Successfully initialized index file '{self.index_file_path}' with {len(index_data)} entries for dataset '{self.name}'.")
        except (FileNotFoundError, PermissionError, IOError) as e:
            self.__logger.error(
                f"An I/O error occurred during index initialization for dataset '{self.name}' from '{source_csv.path}': {e}",
                exc_info=True
            )
            raise
        except Exception as e:
            self.__logger.error(
                f"An unexpected error occurred during index initialization for dataset '{self.name}': {e}",
                exc_info=True
            )
            raise
        return

    def load_movie_data(self, mode: Literal['ALL', 'META']) -> list[MovieData]:
        """
        Loads MovieData objects based on the specified mode using the repository.

        :param mode: The loading mode, either 'ALL' or 'META'.
        :returns: A list of MovieData objects.
        """
        self.__logger.debug(f"Loading all movie data for dataset '{self.name}' in mode '{mode}'.")
        return self.repository.fetch_movies(detail_level=mode)

    def load_movie_sessions(self, number_of_weeks: int) -> list[MovieSessionData]:
        """
        Creates fixed-length movie session data from all movies in the dataset.

        This method leverages the `movie_data` property to get a list of all
        fully-loaded `MovieData` objects. It then delegates to
        `MovieSessionData.create_sessions_from_movie_data_list` to segment
        the data into sessions of the specified length.

        :param number_of_weeks: The number of weeks each movie session should span.
        :returns: A flattened list of all `MovieSessionData` objects created from the dataset.
        """
        self.__logger.debug(f"Creating {number_of_weeks}-week sessions for all movies in dataset '{self.name}'.")

        # Use the existing property to get all fully-loaded MovieData objects
        all_movie_data: list[MovieData] = self.movie_data

        if not all_movie_data:
            self.__logger.warning(f"No movie data available in dataset '{self.name}' to create sessions from.")
            return []

        # Delegate to the existing class method in MovieSessionData that works on an in-memory list
        all_sessions: list[MovieSessionData] = MovieSessionData.create_sessions_from_movie_data_list(
            movie_data_list=all_movie_data,
            number_of_weeks=number_of_weeks
        )
        return all_sessions

    def collect_box_office(self) -> None:
        """
        Collects and saves box office data for all movies in the dataset.

        This method initiates the box office data collection process. It invalidates
        the internal `movie_data` cache, loads movie metadata, and then uses the
        `BoxOfficeCollector` to download and save the data to the filesystem.
        Subsequent access to the `movie_data` property will reload the updated data.
        """
        self.__logger.info(f"Starting box office collection for dataset '{self.name}'.")
        if self.__movies_data_cache is not None:
            self.__logger.info(
                f"Invalidating `movie_data` cache for dataset '{self.name}' before box office collection.")
            self.__movies_data_cache = None
        else:
            self.__logger.debug(
                f"`movie_data` cache for dataset '{self.name}' was already empty before box office collection.")

        movies_to_collect_for: list[MovieData] = self.load_movie_data(mode='META')

        if not movies_to_collect_for:
            self.__logger.warning(
                f"No movie metadata available for dataset '{self.name}'. Skipping box office collection.")
            return

        self.__logger.info(
            f"Collecting box office data for {len(movies_to_collect_for)} movies in dataset '{self.name}'.")
        try:

            with BoxOfficeCollector(download_mode='WEEK') as collector:
                # Note: Collector still expects MovieData objects, but now they are simpler.
                # The collector might need to know WHERE to save.
                # Currently, collector takes 'data_folder'.
                collector.download_box_office_data_for_movies(multiple_movie_data=movies_to_collect_for,
                                                              data_folder=self.box_office_folder_path)

            self.__logger.info(f"Box office collection process finished for dataset '{self.name}'. "
                               f"The `movie_data` cache remains invalidated; reload to see updates.")
        except Exception as e:
            self.__logger.error(f"An error occurred during box office collection for dataset '{self.name}': {e}",
                                exc_info=True)
        return

    def collect_public_review(self, target_website: Literal['PTT', 'DCARD']) -> None:
        """
        Collects and saves public reviews for all movies in the dataset.

        This method initiates the public review collection process from a specified
        website. It invalidates the internal `movie_data` cache, loads movie
        metadata, and then uses the `ReviewCollector` to fetch and save the reviews
        to the filesystem. Subsequent access to the `movie_data` property will
        reload the updated data.

        :param target_website: The name of the website from which to collect reviews ("PTT" or "DCARD").
        """
        try:
            target_website_enum: TargetWebsite = TargetWebsite[target_website.upper()]
        except KeyError:
            self.__logger.error(
                f"Invalid target_website_str: '{target_website}'. Available: {[e.name for e in TargetWebsite]}")
            return

        self.__logger.info(
            f"Starting public review collection for dataset '{self.name}' from {target_website_enum.name}.")
        if self.__movies_data_cache is not None:
            self.__logger.info(
                f"Invalidating `movie_data` cache for dataset '{self.name}' before public review collection.")
            self.__movies_data_cache = None
        else:
            self.__logger.debug(
                f"`movie_data` cache for dataset '{self.name}' was already empty before public review collection.")

        movies_to_collect_for: list[MovieData] = self.load_movie_data(mode='META')

        if not movies_to_collect_for:
            self.__logger.warning(
                f"No movie metadata available for dataset '{self.name}' (index might be empty or missing). "
                f"Skipping public review collection."
            )
            return

        self.__logger.info(
            f"Proceeding with public review collection for {len(movies_to_collect_for)} movies in dataset '{self.name}' from {target_website_enum.name}.")
        try:
            collector: ReviewCollector = ReviewCollector(target_website=target_website_enum)
            collector.collect_reviews_for_movies(movie_list=movies_to_collect_for,
                                                 data_folder=self.public_review_folder_path)

            self.__logger.info(
                f"Public review collection process finished for dataset '{self.name}' from {target_website_enum.name}. "
                f"The `movie_data` cache remains invalidated; reload to see updates.")
        except Exception as e:
            self.__logger.error(
                f"An error occurred during public review collection for dataset '{self.name}' from {target_website_enum.name}: {e}",
                exc_info=True)
        return

    def collect_expert_review(self) -> None:
        """
        Collects expert review data for all movies in this dataset.

        This method is not yet implemented.
        """
        pass

    def compute_sentiment(self, model_id: str) -> None:
        """
        Computes sentiment scores for all public reviews in the dataset and updates them.

        This method using a specified large language model,
        iterates through each movie's public reviews, calculates a sentiment score,
        and then saves the updated reviews back to their respective files.

        :param model_id: The ID of the sentiment analysis model to use.
        """
        self.__logger.info(f"Starting sentiment computation for dataset '{self.name}' using model '{model_id}'.")

        try:
            # Initialize the client once for the entire process
            if LLMProvider.is_local_from_string(model_id=model_id):
                # For local models, provide connection details. These should be configurable in the future.
                llm_client: LLMClient = LLMClient(
                    target_model_id=model_id, # 'ollama/gemma3'
                    local_host='llm-service',
                    local_port=11434
                )
            else:
                # For remote models, the client will handle API key retrieval from environment variables.
                llm_client: LLMClient = LLMClient(target_model_id=model_id)
        except (ValueError, FileNotFoundError) as e:
            self.__logger.error(f"LLM Client Initialization failed: {e}")
            return  # Exit if client can't be created

        try:
            rule_text: Final[str] = """
            你是一個專業的影評情感分析引擎。你的任務是根據使用者提供的**單一電影評論**，判斷其整體情感傾向及強度。

            請遵循以下評分規則，給出一個 1 到 5 之間的整數：

            - **5 (極度正面)**：強烈推薦、神作、完美、非常感動、無可挑剔。評論者表現出極大的熱情或喜愛。
            - **4 (正面)**：好看、值得一看、優點多於缺點、滿意。評論者整體持肯定態度，但可能有些許小遺憾。
            - **3 (中性/普通)**：普通、還行、無感、平庸、殺時間可看。或者評論包含等量的優缺點，難以區分好壞。也包括純粹的劇情討論或提問，沒有明顯情感傾向。
            - **2 (負面)**：不好看、失望、不如預期、缺點多於優點。評論者整體持否定態度，但還沒到憤怒的程度。
            - **1 (極度負面)**：爛片、浪費時間、憤怒、一無是處、極度反推。評論者表現出強烈的厭惡或不滿。

            **重要約束：**
            1. 你將會收到一個完整的電影評論，該評論可能包含多個段落。請你**綜合考量評論的全部內容**，給出一個最終的判斷。
            2. 你的回覆**只能**包含一個數字（"1", "2", "3", "4", 或 "5"），絕對不能包含任何其他文字、符號、解釋，或多個數字。

            ---

            範例 1：
            評論：這部片絕對是年度最佳，看完直接二刷！劇情緊湊，演員表現也都很到位，非常值得一看。
            你的回覆：5

            範例 2：
            評論：特效做得不錯，畫面很美。但劇情有點老套，中間一度想睡覺，整體來說算是一部合格的爆米花電影。
            你的回覆：3

            範例 3：
            評論：劇情真的不行，浪費了這麼好的演員陣容。雖然畫面還不錯，但整體來說還是很失望。
            你的回覆：2

            範例 4：
            評論：請問這部片有彩蛋嗎？我打算週末去看，聽說評價兩極。
            你的回覆：3

            範例 5：
            評論：真的是爛到笑，完全不知道在演什麼，千萬不要浪費錢進戲院！
            你的回覆：1
            ---

            現在，請針對以下評論內容進行判斷：
            """
            total_review_count: int = sum(movie.public_review_count for movie in self.movie_data)
            max_response_retries: Final[int] = 3

            if total_review_count == 0:
                self.__logger.info("No public reviews found in the dataset to process.")
                return

            with tqdm(total=total_review_count, desc="Computing Sentiments") as pbar:
                for movie in self.movie_data:
                    if not movie.public_reviews:
                        continue

                    self.__logger.info(
                        f"Processing {len(movie.public_reviews)} reviews for movie ID {movie.id} ('{movie.name}')..."
                    )

                    updated_reviews: list[PublicReview] = []
                    for review in movie.public_reviews:
                        last_response: str = ""
                        for attempt in range(max_response_retries):
                            self.__logger.debug(
                                f"Attempt {attempt + 1}/{max_response_retries} for review: '{review.title}'")
                            current_temperature: float = 0.1 + (attempt * 0.4)
                            try:
                                response_text: str = llm_client.generate_response(
                                    prompt_texts=review.content,
                                    rule_message=rule_text,
                                    temperature=current_temperature
                                )
                                last_response = response_text

                                # Strict validation for the expected response
                                if response_text in ('1', '2', '3', '4', '5'):
                                    score_val: int = int(response_text)

                                    sentiment_score: Optional[float] = (score_val - 1) / 4.0
                                    self.__logger.debug(
                                        f"Validated response '{response_text}' for review '{review.title}', "
                                        "mapped to sentiment score: {sentiment_score:.2f}")
                                    break  # Exit the retry loop on success
                                else:
                                    self.__logger.warning(
                                        f"Received invalid sentiment response: '{response_text}'. Expected '0' or '1'. Retrying..."
                                    )

                            except RuntimeError as e:
                                # This is a non-daily-limit unrecoverable error from the client for this specific review
                                self.__logger.error(
                                    f"Unrecoverable error from LLM client for review '{review.title}': {e}")
                                sentiment_score: Optional[float] = None
                                break  # Exit the retry loop immediately
                            except Exception as e:
                                self.__logger.error(
                                    f"Unexpected error during sentiment generation for '{review.title}': {e}",
                                    exc_info=True)

                            # Wait a moment before the next retry if the response was invalid
                            sleep(2)
                        else:
                            # This block executes ONLY if the for loop completes without a 'break'.
                            self.__logger.critical(
                                f"Failed to get a valid sentiment for review '{review.title}' after {max_response_retries} attempts. "
                                f"Last invalid response was: '{last_response}'."
                            )
                            sentiment_score: Optional[float] = None

                        if sentiment_score is not None:
                            updated_reviews.append(replace(review, sentiment_score=sentiment_score))
                        else:
                            updated_reviews.append(review)

                        pbar.update(1)

                    # Update the movie object and save the results to disk
                    movie.public_reviews = updated_reviews
                    self.repository.save_movie(movie)

        except DailyRateLimitExceededError as e:
            self.__logger.critical(f"Terminating sentiment computation due to daily rate limit: {e}")
            # No further action needed, the function will now exit gracefully.

        self.__logger.info(f"Sentiment computation for dataset '{self.name}' is complete.")

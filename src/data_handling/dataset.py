from abc import ABC, abstractmethod
from dataclasses import replace
from logging import Logger
from pathlib import Path
from time import sleep
from typing import Any, Callable, cast, Final, Literal, Optional, TypedDict

from openai import RateLimitError
from tqdm import tqdm
from typing_extensions import override

from src.core.constants import Constants
from src.core.logging_manager import LoggingManager
from src.core.project_config import ProjectConfig, ProjectPaths
from src.core.types import ProjectDatasetType
from src.data_collection.box_office_collector import BoxOfficeCollector
from src.data_collection.review_collector import ReviewCollector, TargetWebsite
from src.data_handling.database_client import DatabaseClient, DatabaseConfig
from src.data_handling.file_io import CsvFile
from src.data_handling.movie_collections import MovieData, MovieSessionData, WeekData
from src.data_handling.repositories.db_repository import DbMovieRepository
from src.data_handling.repositories.repository import MovieRepository
from src.data_handling.repositories.yaml_repository import YamlMovieRepository
from src.data_handling.reviews import PublicReview
from src.sentiment_analysis.llm_client import DailyRateLimitExceededError, LLMClient, LLMProvider


class BoxOfficeProgressEntry(TypedDict):
    """
    Represents the structure of a single entry in the box office download progress file.

    :ivar id: The unique identifier for the movie.
    :ivar url: The URL of the movie's box office data page.
    :ivar processed: The status of collection of the movie.
    """
    id: int
    url: str
    processed: bool


class BoxOfficeProgressFile(CsvFile):
    """
    Handles read/write operations for the box office download progress CSV file.

    This class extends CsvFile to manage a progress file that tracks the download
    status (URL, processed) for each movie.

    :ivar HEADER: A tuple defining the CSV header fields: ('id', 'url', 'processed').
    """

    HEADER: Final[tuple[str, str, str]] = ('id', 'url', 'processed')

    def __init__(self, path: Path, encoding: str = Constants.DEFAULT_ENCODING):
        """
        Initializes the BoxOfficeProgressFile handler.

        :param path: The path to the progress CSV file.
        :param encoding: The encoding of the file, defaults to 'utf-8'.
        """
        super().__init__(path=path, encoding=encoding, header=self.HEADER)
        self._logger: Logger = LoggingManager().get_logger('root')

    def save(self, data: list[BoxOfficeProgressEntry]) -> None:
        """
        Saves a list of progress entries to the CSV file.

        This method writes the provided data, ensuring the parent directory exists.
        It uses the class's predefined HEADER for the CSV field names.

        :param data: A list of ``BoxOfficeProgressEntry`` dictionaries to save.
        """
        super().save(data=cast(list[dict[Any, Any]], cast(object, data)))
        self._logger.info(f"Successfully saved {len(data)} progress entries to '{self.path}'.")
        return

    def load(self, row_factory: Optional[Callable[[dict[str, str]], any]] = None) -> list[BoxOfficeProgressEntry]:
        """
        Loads and parses data from the progress CSV file.

        This method overrides the parent ``CsvFile.load`` to use a specific internal
        row factory (``_progress_entry_factory``) for converting rows into
        ``BoxOfficeProgressEntry`` dictionaries. A warning is logged if an external
        ``row_factory`` is provided, as it will be ignored.

        :param row_factory: This parameter is ignored. A warning will be logged if it is provided.
        :returns: A list of parsed ``BoxOfficeProgressEntry`` objects. Returns an
                  empty list if the file does not exist or is empty.
        :raises Exception: Propagates exceptions from the underlying CSV loading
                           process, except for ``FileNotFoundError``.
        """
        if not self.path.exists():
            self._logger.info(f"Progress file not found at '{self.path}'. Returning empty list.")
            return []

        if row_factory is not None:
            self._logger.warning(
                "BoxOfficeProgressFile.load was called with a 'row_factory' argument, "
                "but it will use its internal '_progress_entry_factory' for conversion."
            )

        try:
            loaded_entries: list[Optional[BoxOfficeProgressEntry]] = super().load(
                row_factory=self._progress_entry_factory)
            processed_data: list[BoxOfficeProgressEntry] = [entry for entry in loaded_entries if entry is not None]
            return processed_data
        except FileNotFoundError:
            self._logger.error(f"FileNotFoundError during load after exists() check for '{self.path}'.")
            return []
        except Exception as e:
            self._logger.error(f"Error loading progress file '{self.path}': {e}", exc_info=True)
            raise

    @staticmethod
    def _progress_entry_factory(row: dict[str, str]) -> Optional[BoxOfficeProgressEntry]:
        """
        Converts a raw CSV row into a structured ``BoxOfficeProgressEntry``.

        This factory function validates the input row, ensuring the 'id' field
        exists and is a valid integer. If the row is invalid, it logs a warning
        and returns ``None``.

        :param row: A dictionary representing a single row from the CSV file.
        :returns: A ``BoxOfficeProgressEntry`` instance if the row is valid,
                  otherwise ``None``.
        """
        logger: Logger = LoggingManager().get_logger(
            BoxOfficeProgressFile.__name__)
        movie_id_str: Optional[str] = row.get('id')
        url_str: str = row.get('url', '')
        process_str: str = row.get('processed', '')

        if movie_id_str is None:
            logger.warning(f"Skipping progress entry due to missing 'id': {row}")
            return None

        try:
            movie_id: int = int(movie_id_str)
        except ValueError:
            logger.warning(
                f"Skipping progress entry due to invalid 'id' format: '{movie_id_str}' in {row}")
            return None

        processed_value: bool = process_str.strip().lower() == 'true'

        return BoxOfficeProgressEntry(id=movie_id, url=url_str, processed=processed_value)

    def initialize_from_movies(self, movies: list[MovieData]) -> None:
        """
        Creates and initializes the progress file from a list of movies.

        This method generates an initial progress entry for each movie, setting the
        'id' from the movie data and leaving 'url' and 'file_path' empty.
        It will overwrite the progress file if it already exists.

        :param movies: A list of ``MovieData`` objects to use for initialization.
        """
        initial_data: list[BoxOfficeProgressEntry] = [
            BoxOfficeProgressEntry(id=movie.id, url='', processed=False) for movie in movies
        ]
        self.save(data=initial_data)
        self._logger.info(f"Initialized progress file '{self.path}' with {len(initial_data)} entries.")
        return

    def update_entry(self, movie_id: int, update_field: Literal['url', 'file_path'], new_value: str) -> None:
        """
        Updates a single field for a specific movie entry in the progress file.

        This method reads the entire progress file, finds the entry matching the
        ``movie_id``, modifies the specified ``update_field`` with the ``new_value``,
        and then writes the entire dataset back to the file.

        :param movie_id: The ID of the movie entry to update.
        :param update_field: The name of the field to update (either 'url' or 'file_path').
        :param new_value: The new value to set for the field.
        :raises ValueError: If the ``movie_id`` is not found in the progress file or
                            if ``update_field`` is not a valid field name.
        :raises FileNotFoundError: If the progress file does not exist when an update
                                   is attempted.
        """
        current_progress: list[BoxOfficeProgressEntry] = self.load()
        if not current_progress and not self.exists:
            raise FileNotFoundError(
                f"Progress file '{self.path}' not found. Cannot update entry for movie ID {movie_id}.")

        target_entry: Optional[BoxOfficeProgressEntry] = None
        entry_index: int = -1
        for i, entry in enumerate(current_progress):
            if entry.get('id') == movie_id:
                target_entry = entry
                entry_index = i
                break

        if target_entry is None:
            msg: str = f"Movie ID {movie_id} not found in progress file '{self.path}'. Cannot update."
            self._logger.error(msg)
            raise ValueError(msg)

        if update_field == 'url':
            current_progress[entry_index]['url'] = new_value
        elif update_field == 'processed':
            current_progress[entry_index]['processed'] = bool(new_value)
        else:
            invalid_field_msg: str = f"Invalid update_field: '{update_field}'. Must be 'url' or 'processed'."
            self._logger.error(invalid_field_msg)
            raise ValueError(invalid_field_msg)

        self.save(data=current_progress)
        self._logger.debug(f"Updated {update_field} for movie ID {movie_id} in progress file.")
        return


class BaseDataset(ABC):
    _repository: MovieRepository

    def __init__(self, name: str) -> None:
        self.name = name
        self._logger: Logger = LoggingManager().get_logger('root')
        self._movies_data_cache: Optional[list[MovieData]] = None


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
        if self._movies_data_cache is None:
            self._logger.debug(f"Cache miss for 'movie_data' in dataset '{self.name}'. Loading all movie data.")
            # Convert Iterator to list for caching
            self._movies_data_cache = list(self._repository.fetch_movies(detail_level='ALL'))
            self._logger.debug(
                f"Populated 'movie_data' cache for dataset '{self.name}' with {len(self._movies_data_cache)} items.")
        else:
            self._logger.debug(
                f"Returning cached 'movie_data' for dataset '{self.name}' with {len(self._movies_data_cache)} items.")
        return self._movies_data_cache

    @property
    def movie_metadata(self) -> list[MovieData]:
        self._logger.debug(f"Loading all movie metadata for dataset '{self.name}'.")
        # Convert Iterator to list for backward compatibility with callers expecting a list
        return list(self._repository.fetch_movies(detail_level='META'))

    @property
    def movie_week_data(self) -> list[WeekData]:
        """
        Retrieves a comprehensive list of weekly data for all movies in the dataset.

        This property acts as a direct interface to the underlying repository's
        `fetch_week_data` method. It returns a flat list of `WeekData` objects,
        representing the performance and reviews of movies on a weekly basis.
        This is particularly optimized for database-backed datasets where it leverages
        pre-aggregated views.

        :returns: A list of `WeekData` objects. Returns an empty list if no data is found.
        """
        week_data_list = list(self._repository.fetch_week_data())

        if not week_data_list:
            self._logger.warning(f"No week data found for dataset '{self.name}'.")
            return []
        return week_data_list

    def initialize_from_csv(self, source_csv: CsvFile, **kwargs) -> None:
        if not source_csv.exists:
            raise FileNotFoundError("Source csv file not found.")

        raw_data: list[dict[str, str]] = source_csv.load()
        movies: list[MovieData] = []
        for index, movie_row in enumerate(raw_data):
            movie_name: Optional[str] = movie_row.get('movie_name')
            if movie_name:
                movies.append(MovieData(id=index, name=movie_name))

        self._initialize_and_save(movies, **kwargs)

    def initialize_from_memory(self, movies: list[MovieData], **kwargs) -> None:
        self._initialize_and_save(movies, **kwargs)

    def _initialize_and_save(self, movies: list[MovieData], **kwargs) -> None:

        self._logger.info(f"Initializing dataset '{self.name}'...")

        self._prepare_environment(**kwargs)

        if self._repository.is_storage_occupied():
            raise ValueError(f"Storage for '{self.name}' is already occupied.")

        self._repository.setup_storage()

        if movies:
            self._repository.save_movies(movies)

        self._logger.info(f"Dataset '{self.name}' initialized with {len(movies)} movies.")

    @abstractmethod
    def _prepare_environment(self, **kwargs) -> None:

        pass

    def save_as_yaml(self, new_name: str) -> 'YamlDataset':
        """
        Saves the current dataset as a new YAML-based dataset.
        """
        self._logger.info(f"Saving dataset '{self.name}' as YAML dataset '{new_name}'...")
        new_dataset = YamlDataset(name=new_name)
        new_dataset.initialize_from_memory(movies=self.movie_data)
        return new_dataset

    def save_as_database(
        self,
        new_name: str,
        db_config: Optional[DatabaseConfig] = None,
        root_config: Optional[DatabaseConfig] = None
    ) -> 'DatabaseDataset':
        """
        Saves the current dataset as a new Database-based dataset.
        """
        self._logger.info(f"Saving dataset '{self.name}' as Database dataset '{new_name}'...")
        new_dataset = DatabaseDataset(name=new_name, database_config=db_config)
        new_dataset.initialize_from_memory(movies=self.movie_data, root_config=root_config)
        return new_dataset

    def get_movie_sessions(self, number_of_weeks: int) -> list[MovieSessionData]:
        """
        Creates fixed-length movie session data from all movies in the dataset.
        This method uses a unified strategy by calling `fetch_week_data` on the
        repository, which provides an optimized implementation based on its
        underlying storage (DB view vs. in-memory computation).

        :param number_of_weeks: The number of weeks each movie session should span.
        :returns: A flattened list of all `MovieSessionData` objects created from the dataset.
        """
        self._logger.info(f"Creating {number_of_weeks}-week sessions for all movies in dataset '{self.name}'.")
        self._logger.info("Using unified session creation via repository's `fetch_week_data`.")

        try:
            # Unified call to the repository's interface
            week_data_list = self.movie_week_data

            # The session creation logic is now the same regardless of the source
            return MovieSessionData.create_sessions_from_week_data_list(
                week_data_list=week_data_list,
                number_of_weeks=number_of_weeks
            )
        except Exception as e:
            self._logger.error(f"Failed to load movie sessions for dataset '{self.name}': {e}", exc_info=True)
            return []


    def collect_box_office(self) -> None:
        """
        Collects and saves box office data for all movies in the dataset.
        This method is robust against single-movie failures and supports
        resuming from the last progress if interrupted by a critical error.
        """
        self._logger.info(f"Starting box office collection for dataset '{self.name}'.")
        if self._movies_data_cache is not None:
            self._logger.info(
                f"Invalidating `movie_data` cache for dataset '{self.name}' before box office collection.")
            self._movies_data_cache = None
        else:
            self._logger.debug(
                f"`movie_data` cache for dataset '{self.name}' was already empty before box office collection.")

        movies_to_collect_for: list[MovieData] = self.movie_metadata

        if not movies_to_collect_for:
            self._logger.warning("No movie metadata found. Skipping box office collection.")
            return

        # Define and prepare the temporary progress file
        progress_file_path = ProjectPaths.temp_dir / f"bo_download_progress_{self.name}.csv"
        progress_file = BoxOfficeProgressFile(path=progress_file_path)

        if not progress_file.exists:
            self._logger.info(f"Progress file not found. Initializing at '{progress_file_path}'.")
            progress_file.initialize_from_movies(movies=movies_to_collect_for)
        else:
            self._logger.info(f"Resuming from existing progress file at '{progress_file_path}'.")

        progress_map: dict[int, BoxOfficeProgressEntry] = {entry['id']: entry for entry in progress_file.load()}

        # Main collection logic with nested error handling
        try:
            with BoxOfficeCollector(download_mode='WEEK') as collector:
                for movie in tqdm(
                    movies_to_collect_for, desc="Collecting Box Office", bar_format=Constants.STATUS_BAR_FORMAT
                ):
                    progress = progress_map.get(movie.id)
                    if progress and progress.get('processed'):
                        continue

                    try:
                        # Inner try for single movie processing
                        known_url = progress.get('url') if progress else None
                        box_office_data, movie_url = collector.fetch_single_movie_data(
                            movie_name=movie.name, movie_id=movie.id, known_url=known_url
                        )

                        if box_office_data and movie_url:
                            self._repository.save_box_office(movie_id=movie.id, data=box_office_data)
                            if progress:
                                progress['url'] = movie_url
                                progress['processed'] = True
                            self._logger.debug(f"Successfully processed movie ID {movie.id}.")
                        else:
                            # Handle cases where data is not found (not a critical error)
                            if progress:
                                progress['url'] = movie_url or ''
                                progress['processed'] = False
                            self._logger.warning(f"No box office data found for movie ID {movie.id}.")

                    except Exception as movie_error:
                        # Handle non-critical errors (e.g., network timeout for one movie)
                        self._logger.warning(f"Skipping movie ID {movie.id} due to an error: {movie_error}")
                        if progress:
                            progress['processed'] = False  # Mark as not processed
                        continue  # Continue to the next movie

            # Success path: All movies in the loop have been processed
            self._logger.info("Collection loop completed. Saving final state before cleanup.")
            progress_file.save(data=list(progress_map.values()))

            self._logger.info("Process finished successfully. Deleting temporary progress file.")
            progress_file.delete(missing_ok=True)

        except (KeyboardInterrupt, Exception) as critical_error:
            # Failure path: A critical error interrupted the entire process
            self._logger.critical(
                f"Collection process CRITICALLY interrupted: {critical_error}", exc_info=True
            )
            self._logger.info("Preserving progress file for future resumption...")
            progress_file.save(data=list(progress_map.values()))
            raise critical_error  # Re-raise the exception to signal failure

        self._logger.info(f"Box office collection for dataset '{self.name}' has concluded.")


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
            self._logger.error(
                f"Invalid target_website_str: '{target_website}'. Available: {[e.name for e in TargetWebsite]}")
            return

        self._logger.info(
            f"Starting public review collection for dataset '{self.name}' from {target_website_enum.name}.")
        if self._movies_data_cache is not None:
            self._logger.info(
                f"Invalidating `movie_data` cache for dataset '{self.name}' before public review collection.")
            self._movies_data_cache = None
        else:
            self._logger.debug(
                f"`movie_data` cache for dataset '{self.name}' was already empty before public review collection.")

        movies_to_collect_for: list[MovieData] = self.movie_metadata

        if not movies_to_collect_for:
            self._logger.warning(
                f"No movie metadata available for dataset '{self.name}' (index might be empty or missing). "
                f"Skipping public review collection."
            )
            return

        self._logger.info(
            f"Proceeding with public review collection for {len(movies_to_collect_for)} movies in dataset '{self.name}' from {target_website_enum.name}.")
        try:
            collector: ReviewCollector = ReviewCollector(target_website=target_website_enum)

            self._logger.debug(f"Starting batch review collection for {len(movies_to_collect_for)} movies.")

            # noinspection PyArgumentList
            with collector.managed_browser_session() as browser:
                for movie in tqdm(movies_to_collect_for, desc='Collecting Reviews', bar_format=Constants.STATUS_BAR_FORMAT):
                    self._logger.debug(f"Processing reviews for movie ID {movie.id} ('{movie.name}').")
                    try:
                        # Accessing private method via name mangling as requested to reproduce functionality without other changes
                        newly_fetched_reviews: list[PublicReview] = collector.get_reviews(
                            movie_name=movie.name, browser=browser
                        )

                        movie.update_public_reviews(update_method='EXTEND', data=newly_fetched_reviews)

                        self._repository.save_reviews(movie_id=movie.id, data=newly_fetched_reviews)

                        if not newly_fetched_reviews:
                            self._logger.debug(
                                f"No new reviews found for movie ID {movie.id}. (Repository handles empty storage)."
                            )
                        else:
                            self._logger.debug(
                                f"Successfully collected and saved {len(newly_fetched_reviews)} reviews for movie ID {movie.id}."
                            )

                    except Exception as e:
                        self._logger.error(
                            f"Failed to collect reviews for movie ID {movie.id} ('{movie.name}'): {e}",
                            exc_info=True
                        )

            self._logger.info(
                f"Public review collection process finished for dataset '{self.name}' from {target_website_enum.name}. "
                f"The `movie_data` cache remains invalidated; reload to see updates.")
        except Exception as e:
            self._logger.error(
                f"An error occurred during public review collection for dataset '{self.name}' from {target_website_enum.name}: {e}",
                exc_info=True)
        return

    def collect_expert_review(self) -> None:
        """
        Collects expert review data for all movies in this dataset.

        This method is not yet implemented.
        """
        pass

    def compute_sentiment(
        self, model_id: str, llm_address: str = 'llm-service', llm_port: int = 11434
    ) -> None:
        """
        Computes sentiment scores for all public reviews in the dataset and updates them.

        This method using a specified large language model,
        iterates through each movie's public reviews, calculates a sentiment score,
        and then saves the updated reviews back to their respective files.

        :param model_id: The ID of the sentiment analysis model to use.
        :param llm_address: The address (host) of the local LLM service (e.g., 'localhost' or 'llm-service').
                            Defaults to 'llm-service'.
        :param llm_port: The port number of the local LLM service. Defaults to 11434.
        """
        self._logger.info(f"Starting sentiment computation for dataset '{self.name}' using model '{model_id}'.")

        try:
            # Initialize the client once for the entire process
            if LLMProvider.is_local_from_string(model_id=model_id):
                # For local models, provide connection details. These should be configurable in the future.
                llm_client: LLMClient = LLMClient(
                    target_model_id=model_id, # 'ollama/gemma3'
                    local_host=llm_address,
                    local_port=llm_port
                )
            else:
                # For remote models, the client will handle API key retrieval from environment variables.
                llm_client: LLMClient = LLMClient(target_model_id=model_id)
        except (ValueError, FileNotFoundError) as e:
            self._logger.error(f"LLM Client Initialization failed: {e}")
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

            # Unified Strategy: Iterate movies -> Process reviews -> Save
            # This works for both YAML (file by file) and DB (batch by batch per movie)
            # It avoids loading ALL data into memory.

            movies: list[MovieData] = self.movie_metadata

            with tqdm(total=len(movies), desc="Computing Sentiments") as pbar:
                for movie in movies:
                    # Fetch only reviews for this movie (Iterator)
                    reviews_iter = self._repository.fetch_reviews(movie.id)

                    # Filter for PublicReview and materialize to list for processing
                    public_reviews = [r for r in reviews_iter if isinstance(r, PublicReview)]

                    if not public_reviews:
                        pbar.update(1)
                        continue

                    updated_reviews = []
                    # Process reviews (could be parallelized here)
                    for review in public_reviews:
                        updated_review = self._process_single_review_sentiment(llm_client, review, rule_text)
                        updated_reviews.append(updated_review)

                    # Save back (Batch update for this movie)
                    self._repository.save_reviews(movie.id, updated_reviews)
                    pbar.update(1)

        except DailyRateLimitExceededError as e:
            self._logger.critical(f"Terminating sentiment computation due to daily rate limit: {e}")

        self._logger.info(f"Sentiment computation for dataset '{self.name}' is complete.")

    @staticmethod
    def _process_single_review_sentiment(client: LLMClient, review: PublicReview, rule_text: str) -> PublicReview:
        max_response_retries = 3
        sentiment_score: Optional[float] = None

        for attempt in range(max_response_retries):
            current_temperature = 0.1 + (attempt * 0.4)
            try:
                response_text = client.generate_response(
                    prompt_texts=review.content,
                    rule_message=rule_text,
                    temperature=current_temperature
                )
                if response_text in ('1', '2', '3', '4', '5'):
                    score_val = int(response_text)
                    sentiment_score = (score_val - 1) / 4.0
                    break
            except (AttributeError, IndexError, TypeError, ValueError, RateLimitError):
                sleep(2)

        if sentiment_score is not None:
            return replace(review, sentiment_score=sentiment_score)
        return review


class YamlDataset(BaseDataset):
    @override
    def __init__(self, name: str):
        super().__init__(name=name)
        self._repository = YamlMovieRepository(
            dataset_root_path=ProjectPaths.get_yaml_dataset_path(
                dataset_name=self.name, dataset_type=ProjectDatasetType.STRUCTURED
            )
        )

    @property
    def dataset_path(self) -> Path:
        """
        The root path for this dataset's files.
        """
        return ProjectPaths.get_yaml_dataset_path(dataset_name=self.name, dataset_type=ProjectDatasetType.STRUCTURED)

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

    @override
    def _prepare_environment(self, **kwargs) -> None:
        pass


class DatabaseDataset(BaseDataset):
    @override
    def __init__(self, name: str, database_config: Optional[DatabaseConfig] = None):
        super().__init__(name=name)
        self.__database_config: DatabaseConfig
        if database_config:
            self.__database_config = DatabaseConfig(**database_config, database=name)
        else:
            self.__database_config = DatabaseConfig(**ProjectConfig.DEFAULT_DATABASE_CONFIG, database=name)
        self._repository: DbMovieRepository = DbMovieRepository(
            server_address=self.__database_config['address'],
            server_port=self.__database_config['port'],
            user_name=self.__database_config['user'],
            user_password=self.__database_config['password'],
            database_name=self.__database_config['database']
        )

    @override
    def initialize_from_csv(
        self,
        source_csv: CsvFile,
        root_user_name: Optional[str] = None,
        root_password: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Initializes the database dataset from a CSV file.

        :param source_csv: The source CSV file.
        :param root_user_name: Username for the root/admin database user (required for DB creation).
        :param root_password: Password for the root/admin database user.
        """
        super().initialize_from_csv(
            source_csv=source_csv, root_user_name=root_user_name, root_password=root_password, **kwargs
        )

    @override
    def initialize_from_memory(
        self,
        movies: list[MovieData],
        root_user_name: Optional[str] = None,
        root_password: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Initializes the database dataset from in-memory movie data.

        :param movies: The list of movies to initialize.
        :param root_user_name: Username for the root/admin database user (required for DB creation).
        :param root_password: Password for the root/admin database user.
        """
        super().initialize_from_memory(
            movies=movies, root_user_name=root_user_name, root_password=root_password, **kwargs
        )

    @override
    def _prepare_environment(self, **kwargs) -> None:
        root_user_name: Optional[str] = kwargs.get('root_user_name')
        root_password: Optional[str] = kwargs.get('root_password')

        # This part requires admin privileges
        admin_client = DatabaseClient(
            config=DatabaseConfig(
                address=self.__database_config['address'],
                port=self.__database_config['port'],
                user=root_user_name,
                password=root_password
            )
        )
        with admin_client.connection() as db:
            # Check for database existence without connecting to it
            result = db.execute_statement(f"SHOW DATABASES LIKE '{self.name}'")
            db_exists = bool(result)

        if db_exists:
            # Skip throwing error here, as BaseDataset._initialize_and_save handles occupied storage check
            # via self._repository.is_storage_occupied()
            pass
        else:
            # DB does not exist, create it and grant permissions
            with admin_client.connection() as db:
                db.execute_statement(f"CREATE DATABASE IF NOT EXISTS {self.name}")
                db.execute_statement(f"GRANT ALL ON {self.name}.* TO '{self.__database_config['user']}'@'%'")

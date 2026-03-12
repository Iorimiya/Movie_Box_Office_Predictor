from dataclasses import dataclass, field, replace, InitVar
from logging import Logger
from pathlib import Path
from time import sleep
from typing import Any, Callable, cast, Final, Literal, Optional, TypedDict

from tqdm import tqdm
from yaml import YAMLError

from data_handling.database_client import DatabaseClient
from src.core.constants import Constants
from src.core.logging_manager import LoggingManager
from src.core.project_config import DatabaseConfig, ProjectConfig, ProjectDatasetType, ProjectPaths
from src.data_collection.box_office_collector import BoxOfficeCollector
from src.data_collection.review_collector import ReviewCollector, TargetWebsite
from src.data_handling.file_io import CsvFile
from src.data_handling.movie_collections import MovieData, MovieSessionData
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
        self.__logger: Logger = LoggingManager().get_logger('root')

    def save(self, data: list[BoxOfficeProgressEntry]) -> None:
        """
        Saves a list of progress entries to the CSV file.

        This method writes the provided data, ensuring the parent directory exists.
        It uses the class's predefined HEADER for the CSV field names.

        :param data: A list of ``BoxOfficeProgressEntry`` dictionaries to save.
        """
        super().save(data=cast(list[dict[Any, Any]], cast(object, data)))
        self.__logger.info(f"Successfully saved {len(data)} progress entries to '{self.path}'.")
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
            self.__logger.info(f"Progress file not found at '{self.path}'. Returning empty list.")
            return []

        if row_factory is not None:
            self.__logger.warning(
                "BoxOfficeProgressFile.load was called with a 'row_factory' argument, "
                "but it will use its internal '_progress_entry_factory' for conversion."
            )

        try:
            loaded_entries: list[Optional[BoxOfficeProgressEntry]] = super().load(
                row_factory=self._progress_entry_factory)
            processed_data: list[BoxOfficeProgressEntry] = [entry for entry in loaded_entries if entry is not None]
            return processed_data
        except FileNotFoundError:
            self.__logger.error(f"FileNotFoundError during load after exists() check for '{self.path}'.")
            return []
        except Exception as e:
            self.__logger.error(f"Error loading progress file '{self.path}': {e}", exc_info=True)
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
        self.__logger.info(f"Initialized progress file '{self.path}' with {len(initial_data)} entries.")
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
            self.__logger.error(msg)
            raise ValueError(msg)

        if update_field == 'url':
            current_progress[entry_index]['url'] = new_value
        elif update_field == 'processed':
            current_progress[entry_index]['processed'] = bool(new_value)
        else:
            invalid_field_msg: str = f"Invalid update_field: '{update_field}'. Must be 'url' or 'processed'."
            self.__logger.error(invalid_field_msg)
            raise ValueError(invalid_field_msg)

        self.save(data=current_progress)
        self.__logger.debug(f"Updated {update_field} for movie ID {movie_id} in progress file.")
        return


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
    mode: Literal['DATABASE', 'YAML_FILE'] = 'YAML_FILE'
    override_database_config: InitVar[Optional[DatabaseConfig]] = None

    repository: MovieRepository = field(init=False, repr=False)
    _database_config: Optional[DatabaseConfig] = field(default=None, init=False, repr=False)
    __movies_data_cache: Optional[list[MovieData]] = field(default=None, init=False, repr=False)
    __logger: Logger = field(init=False, repr=False)

    def __post_init__(self, override_database_config: Optional[DatabaseConfig]) -> None:
        """
        Performs post-initialization setup.

        Initializes the logger and sets up the repository based on the specified mode.
        """
        self.__logger = LoggingManager().get_logger('root')

        if self.mode == 'YAML_FILE':
            if override_database_config is not None:
                self.__logger.warning(
                    "`override_database_config` is provided but mode is 'YAML_FILE'. The config will be ignored.")
            dataset_path = ProjectPaths.get_dataset_path(dataset_name=self.name,
                                                         dataset_type=ProjectDatasetType.STRUCTURED)
            self.repository = YamlMovieRepository(dataset_root_path=dataset_path)

        elif self.mode == 'DATABASE':
            self._database_config = ProjectConfig.DEFAULT_DATABASE_CONFIG.copy()
            if override_database_config:
                self._database_config.update(override_database_config)

            self.repository = DbMovieRepository(
                server_address=self._database_config['address'],
                server_port=self._database_config['port'],
                user_name=self._database_config['user'],
                user_password=self._database_config['password'],
                database_name=self.name
            )
        else:
            raise ValueError(f"Invalid mode specified: {self.mode}")

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
            # Convert Iterator to list for caching
            self.__movies_data_cache = list(self.repository.fetch_movies(detail_level='ALL'))
            self.__logger.debug(
                f"Populated 'movie_data' cache for dataset '{self.name}' with {len(self.__movies_data_cache)} items.")
        else:
            self.__logger.debug(
                f"Returning cached 'movie_data' for dataset '{self.name}' with {len(self.__movies_data_cache)} items.")
        return self.__movies_data_cache

    def initialize_dataset(self, source_csv: CsvFile, root_config: DatabaseConfig) -> None:
        """
        Initializes the dataset storage from a source CSV file.

        This method orchestrates the initialization process:
        1. Checks if storage is already occupied.
        2. Sets up the storage structure (folders or DB schema).
        3. Loads and prepares the initial movie list from CSV.
        4. Saves the initial movie data to the repository.

        :param root_config: Database configuration for admin tasks (creating DB).
        :param source_csv: A CsvFile instance representing the source CSV file
                           containing at least a 'movie_name' column.
        """
        self.__logger.info(f"Initializing dataset '{self.name}' from source '{source_csv.path}'.")

        if self.mode == 'DATABASE':
            admin_client = DatabaseClient(config=root_config)
            with admin_client.connection() as db:
                # Check for database existence without connecting to it
                result = db.execute_statement(f"SHOW DATABASES LIKE '{self.name}'")
                db_exists = bool(result)

            if db_exists:
                if self.repository.is_storage_occupied():
                    self.__logger.warning(
                        f"Storage for dataset '{self.name}' is already occupied. Skipping initialization.")
                    return
            else:
                # DB does not exist, create it and grant permissions
                with admin_client.connection() as db:
                    db.execute_statement(f"CREATE DATABASE IF NOT EXISTS {self.name}")
                    db.execute_statement(f"GRANT ALL ON {self.name}.* TO '{self._database_config['user']}'@'%'")

        elif self.mode == 'YAML_FILE':
            if self.repository.is_storage_occupied():
                self.__logger.warning(
                    f"Storage for dataset '{self.name}' is already occupied. Skipping initialization.")
                return

        # Delegate schema/folder creation to repository
        self.repository.setup_storage()

        # Load and Prepare Data
        try:
            source_data: list[dict[str, str]] = source_csv.load()
            if not source_data:
                self.__logger.warning(f"Source CSV file '{source_csv.path}' is empty.")
                return

            movies: list[MovieData] = []
            for index, movie_row in enumerate(source_data):
                movie_name: Optional[str] = movie_row.get('movie_name')
                if movie_name:
                    movies.append(MovieData(id=index, name=movie_name))

            # Save Data
            if movies:
                self.repository.save_movies(movies)
                self.__logger.info(f"Successfully initialized dataset '{self.name}' with {len(movies)} movies.")

        except Exception as e:
            self.__logger.error(f"Failed to initialize dataset '{self.name}': {e}", exc_info=True)
            raise

    @classmethod
    def create_from_data(
        cls,
        new_dataset_name: str,
        movies: list[MovieData],
        mode: Literal['DATABASE', 'YAML_FILE'] = 'YAML_FILE',
        override_database_config: Optional[DatabaseConfig] = None
    ) -> 'Dataset':
        """
        Creates a new dataset from a list of MovieData objects.

        This method handles the entire process of creating a new dataset instance,
        setting up its storage infrastructure, and populating it with the provided data.

        :param new_dataset_name: The name for the new dataset.
        :param movies: A list of MovieData objects to populate the dataset with.
        :param mode: The storage mode for the new dataset.
        :param override_database_config: Optional database configuration.
        :return: The newly created Dataset instance.
        """
        logger = LoggingManager().get_logger('root')
        logger.info(f"Creating new dataset '{new_dataset_name}' with {len(movies)} movies in mode '{mode}'.")

        # 1. Create Dataset Instance
        new_dataset = cls(
            name=new_dataset_name,
            mode=mode,
            override_database_config=override_database_config
        )

        # 2. Check if occupied
        if new_dataset.repository.is_storage_occupied():
            raise ValueError(f"Storage for dataset '{new_dataset_name}' is already occupied.")

        try:
            # 3. Setup Storage Infrastructure
            if mode == 'DATABASE':
                # Ensure DB exists (Admin Task)
                # Use root config from ProjectConfig (assuming it's set correctly for admin tasks)
                admin_client = DatabaseClient(config=ProjectConfig.DEFAULT_DATABASE_CONFIG)
                with admin_client.connection() as db:
                    db.execute_statement(f"CREATE DATABASE IF NOT EXISTS {new_dataset_name}")
                    target_user = new_dataset._database_config['user']
                    db.execute_statement(f"GRANT ALL ON {new_dataset_name}.* TO '{target_user}'@'%'")

            new_dataset.repository.setup_storage()

            # 4. Save Data
            new_dataset.repository.save_movies(movies)
            logger.info(f"Successfully created and populated dataset '{new_dataset_name}'.")

        except Exception as e:
            logger.error(f"Failed to create dataset '{new_dataset_name}': {e}", exc_info=True)
            raise

        return new_dataset

    def load_movie_data(self, mode: Literal['ALL', 'META']) -> list[MovieData]:
        """
        Loads MovieData objects based on the specified mode using the repository.

        :param mode: The loading mode, either 'ALL' or 'META'.
        :returns: A list of MovieData objects.
        """
        self.__logger.debug(f"Loading all movie data for dataset '{self.name}' in mode '{mode}'.")
        # Convert Iterator to list for backward compatibility with callers expecting a list
        return list(self.repository.fetch_movies(detail_level=mode))

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

        self.box_office_folder_path.mkdir(parents=True, exist_ok=True)

        progress_file: BoxOfficeProgressFile = BoxOfficeProgressFile(
            path=self.box_office_folder_path / "download_progress.csv")

        if not progress_file.exists:
            self.__logger.info(f"Progress file '{progress_file.path}' not found. Initializing.")
            progress_file.initialize_from_movies(movies=movies_to_collect_for)

        self.__logger.info(f"Loading progress file '{progress_file.path}' into memory...")
        all_progress_entries: list[BoxOfficeProgressEntry] = progress_file.load()
        progress_map: dict[int, BoxOfficeProgressEntry] = {entry['id']: entry for entry in all_progress_entries}
        self.__logger.info(f"Loaded {len(progress_map)} entries into progress map.")

        try:
            with BoxOfficeCollector(download_mode='WEEK') as collector:
                with tqdm(
                    total=len(movies_to_collect_for), bar_format=Constants.STATUS_BAR_FORMAT, desc="Collecting Box Office"
                ) as pbar:
                    for movie in movies_to_collect_for:
                        pbar.set_postfix_str(f"Movie: {movie.name[:30]}...", refresh=True)

                        # Check progress to see if we can skip based on file_path
                        progress: Optional[BoxOfficeProgressEntry] = progress_map.get(movie.id)
                        if progress and progress.get('processed') is True:
                            self.__logger.info(f"Data for movie ID {movie.id} already exists. Skipping.")
                            pbar.update(1)
                            continue

                        known_url: Optional[str] = progress.get('url') if progress else None

                        # Fetch data using the refactored collector
                        box_office_data, movie_url = collector.fetch_single_movie_data(
                            movie_name=movie.name, movie_id=movie.id, known_url=known_url)

                        # Persist data and update progress if fetch was successful
                        if box_office_data and movie_url:
                            movie.update_box_office(data=box_office_data, update_method='REPLACE')
                            self.repository.save_box_office(movie_id=movie.id, data=box_office_data)
                            # Update progress file
                            if progress:
                                progress['url'] = movie_url
                                progress['processed'] = True
                            self.__logger.info(f"Box office data for movie ID {movie.id} processed and saved.")
                        else:
                            # Handle failure by creating an empty file to prevent re-attempts
                            self.__logger.warning(
                                f"No box office data found for movie ID {movie.id}."
                            )
                            try:
                                # Update in-memory progress map
                                if progress:
                                    progress['url'] = movie_url if movie_url else ''
                                    progress['processed'] = False
                            except (OSError, YAMLError) as e:
                                self.__logger.error(
                                    f"Error creating empty file for movie ID {movie.id}: {e}", exc_info=True)

                        pbar.update(1)
        finally:
            self.__logger.info("Collection loop finished. Saving all progress updates to disk...")
            # Convert map back to list, preserving order if necessary (though order isn't critical here)
            updated_progress_list: list[BoxOfficeProgressEntry] = list(progress_map.values())
            try:
                progress_file.save(data=updated_progress_list)
            except Exception as e:
                self.__logger.critical(
                    f"Failed to save final progress to '{progress_file.path}': {e}", exc_info=True)

        self.__logger.info(f"Box office collection process finished for dataset '{self.name}'. "
                           f"The `movie_data` cache remains invalidated; reload to see updates.")
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

            # Unified Strategy: Iterate movies -> Process reviews -> Save
            # This works for both YAML (file by file) and DB (batch by batch per movie)
            # It avoids loading ALL data into memory.

            movies = self.load_movie_data(mode='META')

            with tqdm(total=len(movies), desc="Computing Sentiments") as pbar:
                for movie in movies:
                    # Fetch only reviews for this movie (Iterator)
                    reviews_iter = self.repository.fetch_reviews(movie.id)

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
                    self.repository.save_reviews(movie.id, updated_reviews)
                    pbar.update(1)

        except DailyRateLimitExceededError as e:
            self.__logger.critical(f"Terminating sentiment computation due to daily rate limit: {e}")

        self.__logger.info(f"Sentiment computation for dataset '{self.name}' is complete.")

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
            except Exception:
                sleep(2)

        if sentiment_score is not None:
            return replace(review, sentiment_score=sentiment_score)
        return review

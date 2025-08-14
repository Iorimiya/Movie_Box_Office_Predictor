from dataclasses import dataclass, field, replace
from functools import cached_property
from logging import Logger
from pathlib import Path
from typing import cast, Literal, Optional

from src.core.logging_manager import LoggingManager
from src.core.project_config import ProjectPaths, ProjectDatasetType, ProjectModelType
from src.data_collection.box_office_collector import BoxOfficeCollector
from src.data_collection.review_collector import ReviewCollector, TargetWebsite
from src.data_handling.box_office import BoxOffice
from src.data_handling.file_io import CsvFile
from src.data_handling.movie_collections import MovieData, MovieSessionData
from src.data_handling.movie_metadata import MovieMetadata, MovieMetadataRawData, MoviePathMetadata
from src.data_handling.reviews import PublicReview, ExpertReview
from src.models.sentiment.components.data_processor import SentimentDataProcessor
from src.models.sentiment.components.model_core import SentimentModelCore, SentimentPredictConfig


@dataclass(kw_only=True)
class Dataset:
    """
    Manages a collection of movie data, including metadata, box office figures, and reviews.

    Provides methods to load, initialize, and collect data for a named dataset.
    It handles file paths and interactions with data collectors.

    :ivar name: The unique name of the dataset.
    :ivar __movies_data_cache: An internal cache for the fully loaded list of MovieData objects.
                              It is initialized to None and populated on first access to `movie_data` property.
                              The cache is invalidated when data collection methods are called.
    """
    name: str
    __movies_data_cache: Optional[list[MovieData]] = field(default=None, init=False, repr=False)
    __logger: Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Performs post-initialization setup, primarily for acquiring the logger.

        This ensures the logger is acquired only when a `Dataset` object is
        instantiated, preventing premature initialization of the LoggingManager
        during module import.
        """
        self.__logger = LoggingManager().get_logger('root')

    @property
    def dataset_path(self) -> Path:
        """The root path for this dataset's files."""
        return ProjectPaths.get_dataset_path(dataset_name=self.name, dataset_type=ProjectDatasetType.STRUCTURED)

    @property
    def index_file_path(self) -> Path:
        """The path to the index CSV file for this dataset."""
        return self.dataset_path / 'index.csv'

    @property
    def box_office_folder_path(self) -> Path:
        """The path to the folder containing box office data files for this dataset."""
        return self.dataset_path / 'box_office'

    @property
    def public_review_folder_path(self) -> Path:
        """The path to the folder containing public review data files for this dataset."""
        return self.dataset_path / 'public_review'

    @property
    def expert_review_folder_path(self) -> Path:
        """The path to the folder containing expert review data files for this dataset."""
        return self.dataset_path / 'expert_review'

    @property
    def index_file(self) -> CsvFile:
        """A CsvFile instance for interacting with the dataset's index file."""
        return CsvFile(path=self.index_file_path)

    @cached_property
    def movies_metadata(self) -> list[MovieMetadata]:
        """
        A cached list of MovieMetadata objects loaded from the dataset's index file.

        The list is generated once upon first access and then cached.
        If the index file is not found, empty, or an error occurs during loading,
        an empty list is returned and a log message is generated.

        :returns: A list of MovieMetadata objects.
        """
        self.__logger.info(
            f"Attempting to create MovieSourceInfo objects for dataset '{self.name}' from index file: '{self.index_file_path}'.")

        raw_movie_data_from_csv: list[dict[str, str]]
        try:
            if not self.index_file_path.exists():
                self.__logger.error(f"Index file not found: '{self.index_file_path}' for dataset '{self.name}'.")
                return []

            raw_movie_data_from_csv = self.index_file.load()

            if not raw_movie_data_from_csv:
                self.__logger.info(
                    f"No movie data found or index file is empty: '{self.index_file_path}' for dataset '{self.name}'.")
                return []
        except FileNotFoundError:
            return []
        except Exception as e:
            self.__logger.error(f"Error loading index file '{self.index_file_path}' for dataset '{self.name}': {e}")
            return []

        return [
            movie_metadata for raw_movie_data in raw_movie_data_from_csv
            if (movie_metadata := MovieMetadata.from_csv_raw_data(
                source=cast(MovieMetadataRawData, raw_movie_data)
            )) is not None
        ]

    @property
    def movie_data(self) -> list[MovieData]:
        """
        A list of fully populated MovieData objects for the dataset.

        This property uses an internal cache (`_movies_data_cache`).
        On first access, it loads all movie data (metadata, box office, reviews)
        using `load_all_movie_data(mode='ALL')` and caches the result.
        Subsequent accesses return the cached list.
        The cache is invalidated by data collection methods like `collect_box_office`.

        :returns: A list of MovieData objects.
        """
        if self.__movies_data_cache is None:
            self.__logger.info(f"Cache miss for 'movie_data' in dataset '{self.name}'. Loading all movie data.")
            self.__movies_data_cache = self.load_movie_data(mode= 'ALL')
            self.__logger.info(f"Populated 'movie_data' cache for dataset '{self.name}' with {len(self.__movies_data_cache)} items.")
        else:
            self.__logger.debug(f"Returning cached 'movie_data' for dataset '{self.name}' with {len(self.__movies_data_cache)} items.")
        return self.__movies_data_cache

    def initialize_index_file(self, source_csv: CsvFile) -> None:
        """
        Initializes or overwrites the dataset's index file from a source CSV file.

        It reads movie names from the 'movie_name' column of the source CSV,
        assigns a sequential ID (starting from 0), and saves this new
        index data (id, name) to the dataset's `index.csv` file.

        :param source_csv: A CsvFile instance representing the source CSV file
                           containing at least a 'movie_name' column.
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

    def load_movie_source_info(self) -> list[MoviePathMetadata]:
        """
        Loads movie metadata and constructs paths to their associated data files.

        This method leverages the `movies_metadata` cached property to get basic
        movie metadata (ID, name) and then, for each movie, creates a
        `MoviePathMetadata` object. This object includes the original metadata
        plus fully resolved paths to where that movie's box office, public review,
        and expert review data files are expected to be located within the dataset's
        directory structure.

        :returns: A list of `MoviePathMetadata` objects. If no base metadata is found,
                  an empty list is returned.
        """
        self.__logger.debug(f"Loading movie source info (paths) for dataset '{self.name}'.")
        current_movies_metadata: list[MovieMetadata] = self.movies_metadata
        if not current_movies_metadata:
            self.__logger.info(f"No base movie metadata found for dataset '{self.name}'. Cannot load source info.")
            return []

        path_metadata_list: list[MoviePathMetadata] = [
            MoviePathMetadata.from_metadata(source=movie_metadata, dataset_root_path=self.dataset_path)
            for movie_metadata in current_movies_metadata
        ]
        self.__logger.debug(f"Generated {len(path_metadata_list)} MoviePathMetadata objects for dataset '{self.name}'.")
        return path_metadata_list


    def load_movie_data(self, mode: Literal['ALL', 'META']) -> list[MovieData]:
        """
        Loads MovieData objects based on the specified mode.

        'ALL': Loads complete MovieData objects, including metadata, box office data,
               public reviews, and expert reviews by reading from their respective files.
        'META': Loads MovieData objects with only metadata (ID, name). Box office
                and review lists will be empty. This mode is typically used to prepare
                a list of movies for data collection processes.

        :param mode: The loading mode, either 'ALL' or 'META'.
        :returns: A list of MovieData objects.
        :raises ValueError: If an invalid mode is provided.
        """

        self.__logger.info(f"Loading all movie data for dataset '{self.name}' in mode '{mode}'.")

        match mode:
            case 'ALL':
                source_infos: list[MoviePathMetadata] = self.load_movie_source_info()
                if not source_infos:
                    self.__logger.info(
                        f"No processable movie metadata after initial validation from '{self.index_file_path}'."
                    )
                    return []

                loaded_data: list[MovieData] = [
                    MovieData(
                        metadata=movie_meta_info,
                        box_office=BoxOffice.create_multiple(source=movie_meta_info.box_office_file_path),
                        public_reviews=PublicReview.create_multiple(source=movie_meta_info.public_reviews_file_path),
                        expert_reviews=ExpertReview.create_multiple(source=movie_meta_info.expert_reviews_file_path)
                    ) for movie_meta_info in source_infos
                ]
                self.__logger.info(
                    f"Loaded {len(loaded_data)} full MovieData objects for dataset '{self.name}' in 'ALL' mode."
                )
                return loaded_data
            case 'META':
                movies_meta: list[MovieMetadata] = self.movies_metadata
                if not movies_meta:
                    self.__logger.info(
                        f"No processable movie metadata after initial validation from '{self.index_file_path}'."
                    )
                    return []

                meta_data_list: list[MovieData] = [
                    MovieData(
                        metadata=movie_meta, box_office=[], public_reviews=[], expert_reviews=[]
                    )
                    for movie_meta in movies_meta
                ]
                self.__logger.info(
                    f"Loaded {len(meta_data_list)} MovieData objects (metadata only) for dataset '{self.name}' in 'META' mode."
                )
                return meta_data_list
            case _:
                err_msg: str = f"Invalid mode '{mode}' specified for load_all_movie_data. Must be 'ALL' or 'META'."
                self.__logger.error(err_msg)
                raise ValueError(err_msg)

    def load_movie_sessions(self, number_of_weeks: int) -> list[MovieSessionData]:
        """
        Creates MovieSessionData objects for all movies in this dataset.

        It reads the index file to get movie metadata, then for each movie,
        it loads its full data and segments it into valid, fixed-length sessions.

        :param number_of_weeks: The number of weeks each movie session should span.
        :return: A flattened list of all MovieSessionData objects created from the dataset.
        :raises FileNotFoundError: If the dataset's index file is not found (propagated).
        :raises ValueError: If the index file is found but contains no processable movie metadata (propagated).
        """
        self.__logger.info(f"Creating {number_of_weeks}-week sessions for all movies in dataset '{self.name}'.")

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
        Collects box office data for all movies in this dataset.

        This method initiates the box office data collection process.
        It first invalidates any cached `movie_data`. Then, it loads
        movie metadata, passes it to the `BoxOfficeCollector`, which
        downloads and saves the box office data to the filesystem.
        The internal `_movies_data_cache` is set to None, so subsequent
        access to `self.movie_data` will reload the (potentially updated) data.
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
                collector.download_box_office_data_for_movies(multiple_movie_data=movies_to_collect_for,
                                                              data_folder=self.box_office_folder_path)

            self.__logger.info(f"Box office collection process finished for dataset '{self.name}'. "
                               f"The `movie_data` cache remains invalidated; reload to see updates.")
        except Exception as e:
            self.__logger.error(f"An error occurred during box office collection for dataset '{self.name}': {e}",
                                exc_info=True)
        return

    def collect_public_review(self, target_website:Literal['PTT','DCARD']) -> None:
        """
        Collects public review data for all movies in this dataset from the specified target website.

        This method initiates the public review data collection process.
        It first invalidates any cached `movie_data`. Then, it loads
        movie metadata (as MovieData objects), passes this list to the
        `ReviewCollector`, which fetches and saves public reviews.
        The internal `__movies_data_cache` is set to None, so subsequent
        access to `self.movie_data` will reload the (potentially updated) data.

        :param target_website: The name of the website from which to collect reviews (e.g., "PTT", "DCARD").
                                   Defaults to "PTT".
        """

        try:
            target_website_enum: TargetWebsite = cast(TargetWebsite, TargetWebsite[target_website.upper()])
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
            collector.collect_reviews_for_movies(movie_list=movies_to_collect_for,data_folder=self.public_review_folder_path)

            self.__logger.info(
                f"Public review collection process finished for dataset '{self.name}' from {target_website_enum.name}. "
                f"The `movie_data` cache remains invalidated; reload to see updates.")
        except Exception as e:
            self.__logger.error(
                f"An error occurred during public review collection for dataset '{self.name}' from {target_website_enum.name}: {e}",
                exc_info=True)
        return

    def collect_expert_review(self) -> None:
        pass

    def compute_sentiment(self, model_id: str, model_epoch: int) -> None:
        """
        Computes sentiment scores for all public reviews in the dataset and updates them.

        This method loads a specified sentiment analysis model and its data processor,
        iterates through each movie's public reviews, calculates a sentiment score,
        and then saves the updated reviews back to their respective files.

        :param model_id: The ID of the sentiment analysis model to use.
        :param model_epoch: The specific epoch of the model to use.
        :raises FileNotFoundError: If the required model or processor artifacts are not found.
        :raises ValueError: If the loaded data processor is missing required components (tokenizer, max_sequence_length).
        """
        model_artifacts_path: Path = ProjectPaths.get_model_root_path(
            model_id=model_id, model_type=ProjectModelType.SENTIMENT
        )
        model_file_path: Path = model_artifacts_path / f"{model_id}_{model_epoch:04d}.keras"

        try:
            self.__logger.info(f"Loading sentiment model components for model '{model_id}' (epoch: {model_epoch}).")
            data_processor = SentimentDataProcessor(model_artifacts_path=model_artifacts_path)
            model_core = SentimentModelCore(model_path=model_file_path)
        except Exception as e:
            self.__logger.error(f"Failed to initialize model components: {e}", exc_info=True)
            raise

        if not data_processor.tokenizer or data_processor.max_sequence_length is None:
            raise ValueError(
                "Data processor is missing required artifacts (tokenizer or max_sequence_length). "
                "Ensure the model was trained correctly and artifacts were saved."
            )

        for movie in self.movie_data:
            if not movie.public_reviews:
                self.__logger.info(f"No public reviews to process for movie ID {movie.id} ('{movie.name}').")
                continue

            self.__logger.info(
                f"Computing sentiment for {len(movie.public_reviews)} reviews for movie ID {movie.id}...")

            updated_reviews: list[PublicReview] = []
            for review in movie.public_reviews:
                pred_config = SentimentPredictConfig(verbose=0)

                processed_input = data_processor.process_for_prediction(single_input=review.content)
                prediction = model_core.predict(data=processed_input, config=pred_config)
                sentiment_score = float(prediction[0][0])

                updated_reviews.append(replace(review, sentiment_score=sentiment_score))

            movie.public_reviews = updated_reviews
            movie.save_public_reviews(target_directory=self.public_review_folder_path)

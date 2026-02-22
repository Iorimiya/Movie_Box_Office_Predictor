from logging import Logger
from pathlib import Path
from typing import cast, Iterator, Literal, Optional, Type

from typing_extensions import override

from data_handling.repositories.repository import MovieRepository
from data_handling.reviews import ReviewSerializableData
from src.core.logging_manager import LoggingManager
from src.core.project_config import ProjectPaths
from src.data_handling.box_office import BoxOffice
from src.data_handling.file_io import CsvFile, YamlFile
from src.data_handling.movie_collections import MovieData, WeekData
from src.data_handling.reviews import ExpertReview, PublicReview, Review


class YamlMovieRepository(MovieRepository):
    """
    A repository implementation that stores movie data in YAML files.

    It manages the directory structure for a dataset, including index files
    and subfolders for different data components.
    """

    def __init__(self, dataset_root_path: Path) -> None:
        """
        Initializes the repository with the root path of the dataset.

        :param dataset_root_path: The root directory of the dataset.
        """
        self.dataset_root_path = dataset_root_path
        self.logger: Logger = LoggingManager().get_logger('root')

    @property
    def index_file_path(self) -> Path:
        return self.dataset_root_path / ProjectPaths.INDEX_FILE_NAME

    @property
    def box_office_folder_path(self) -> Path:
        return self.dataset_root_path / ProjectPaths.BOX_OFFICE_SUBFOLDER_NAME

    @property
    def public_reviews_folder_path(self) -> Path:
        return self.dataset_root_path / ProjectPaths.PUBLIC_REVIEWS_SUBFOLDER_NAME

    @property
    def expert_reviews_folder_path(self) -> Path:
        return self.dataset_root_path / ProjectPaths.EXPERT_REVIEWS_SUBFOLDER_NAME

    @override
    def setup_storage(self) -> None:
        """
        Creates necessary directories and empty index files.
        """
        self.logger.info(f"Setting up YAML storage at '{self.dataset_root_path}'.")

        # Create directories
        self.dataset_root_path.mkdir(parents=True, exist_ok=True)
        self.box_office_folder_path.mkdir(parents=True, exist_ok=True)
        self.public_reviews_folder_path.mkdir(parents=True, exist_ok=True)
        self.expert_reviews_folder_path.mkdir(parents=True, exist_ok=True)

        # Create empty index file if not exists
        if not self.index_file_path.exists():
            self.logger.info(f"Creating empty index file at '{self.index_file_path}'.")
            CsvFile(path=self.index_file_path).save(data=[])

    @override
    def is_storage_occupied(self) -> bool:
        """
        Checks if the index file exists and is not empty.
        """
        if not self.index_file_path.exists():
            return False

        try:
            data = CsvFile(path=self.index_file_path).load()
            return len(data) > 0
        except Exception:
            # If file exists but can't be read, assume occupied/corrupted
            return True

    def _load_metadata_from_index(self) -> list[MovieData]:
        """Loads movie metadata from the index CSV file and returns basic MovieData objects."""
        if not self.index_file_path.exists():
            self.logger.warning(f"Index file not found: {self.index_file_path}")
            return []

        try:
            raw_data = CsvFile(path=self.index_file_path).load()
            movies = []
            for item in raw_data:
                raw_id = item.get('id')
                raw_name = item.get('name')
                if raw_id and raw_name:
                    try:
                        movies.append(MovieData(id=int(raw_id), name=raw_name))
                    except ValueError:
                        self.logger.warning(f"Invalid ID format in index file: {raw_id}")
            return movies
        except Exception as e:
            self.logger.error(f"Error loading index file: {e}")
            return []

    @override
    def fetch_movies(
        self,
        filters: Optional[dict] = None,
        detail_level: Literal['META', 'ALL'] = 'META'
    ) -> Iterator[MovieData]:
        """
        Fetches movies from YAML files.

        :param filters: Currently only supports filtering by 'name' (exact match).
        :param detail_level: 'META' loads only metadata; 'ALL' loads full data.
        """
        movies = self._load_metadata_from_index()

        # Apply filters (simple implementation for now)
        if filters and 'name' in filters:
            target_name = filters['name']
            movies = [m for m in movies if m.name == target_name]

        if detail_level == 'META':
            yield from movies
            return

        # Load detailed data
        for movie in movies:
            movie.box_office = list(self.fetch_box_office(movie.id))

            reviews = list(self.fetch_reviews(movie.id))
            movie.public_reviews = [r for r in reviews if isinstance(r, PublicReview)]
            movie.expert_reviews = [r for r in reviews if isinstance(r, ExpertReview)]

            yield movie

    @override
    def save_movies(self, movies: list[MovieData]) -> None:
        """Saves a list of movies' data to YAML files."""
        # Update index file
        # Note: This overwrites the index with the provided movies.
        # If appending is needed, logic should be different.
        # But for initialization/cloning, overwriting is expected.
        index_data = [{'id': str(m.id), 'name': m.name} for m in movies]
        CsvFile(path=self.index_file_path).save(data=index_data)

        for movie in movies:
            if movie.box_office:
                self.save_box_office(movie.id, movie.box_office)

            all_reviews = movie.public_reviews + movie.expert_reviews
            if all_reviews:
                self.save_reviews(movie.id, all_reviews)

    @override
    def fetch_movie_name_to_id_map(self) -> dict[str, int]:
        """Fetches a mapping of movie names to IDs from the index file."""
        movies = self._load_metadata_from_index()
        return {m.name: m.id for m in movies}

    @override
    def fetch_box_office(
        self, movie_id: Optional[int] = None, week_number: Optional[int] = None
    ) -> Iterator[BoxOffice]:
        """
        Fetches box office data.

        :param movie_id: The ID of the movie. If None, fetches box office data for ALL movies.
        :param week_number: The specific week number to fetch (1-based).
                            If provided, movie_id must also be provided.
        :raises ValueError: If week_number is <= 0 or if movie_id is None when week_number is provided.
        """
        if week_number is not None:
            if week_number <= 0:
                raise ValueError("week_number must be greater than 0.")
            if movie_id is None:
                raise ValueError("movie_id must be provided when querying for a specific week_number.")

        if movie_id is not None:
            # Single movie fetch
            path = self.box_office_folder_path / f"{movie_id}.yaml"
            if not path.exists():
                if week_number is not None:
                    raise ValueError(f"No box office data found for movie {movie_id} (file missing).")
                return
            try:
                data = YamlFile(path=path).load() or []
                box_office_list = BoxOffice.create_multiple(source=data, schema_type='NESTED')

                if week_number is not None:
                    # Sort by start_date to ensure consistent ordering
                    box_office_list.sort(key=lambda x: x.start_date)
                    if week_number > len(box_office_list):
                        raise ValueError(
                            f"Week number {week_number} out of range for movie {movie_id} (total {len(box_office_list)} weeks).")
                    yield box_office_list[week_number - 1]
                else:
                    yield from box_office_list

            except ValueError as ve:
                raise ve  # Re-raise validation errors
            except Exception as e:
                self.logger.error(f"Error loading box office for movie {movie_id}: {e}")
                if week_number is not None:
                    raise ValueError(f"Error loading data for movie {movie_id}: {e}")
        else:
            # Fetch all (iterate over all files in the folder)
            if not self.box_office_folder_path.exists():
                return
            for path in self.box_office_folder_path.glob("*.yaml"):
                try:
                    data = YamlFile(path=path).load() or []
                    yield from BoxOffice.create_multiple(source=data, schema_type='NESTED')
                except Exception as e:
                    self.logger.error(f"Error loading box office from {path}: {e}")

    @override
    def save_box_office(self, movie_id: int, data: list[BoxOffice]) -> None:
        self.box_office_folder_path.mkdir(parents=True, exist_ok=True)
        path = self.box_office_folder_path / f"{movie_id}.yaml"
        serializable_data = [item.as_serializable_dict() for item in data]
        try:
            # noinspection PyTypeChecker
            YamlFile(path=path).save(data=serializable_data)
        except Exception as e:
            self.logger.error(f"Error saving box office for movie {movie_id}: {e}")

    @override
    def fetch_reviews(self, movie_id: Optional[int] = None) -> Iterator[Review]:
        def fetch_all_reviews_from_folder(folder_path: Path, review_class: Type[Review]) -> Iterator[Review]:
            if folder_path.exists():
                for path in folder_path.glob("*.yaml"):
                    try:
                        data = YamlFile(path=path).load() or []
                        # noinspection PyTypeChecker
                        yield from review_class.create_multiple(source=data, schema_type='NESTED')
                    except Exception as e:
                        self.logger.error(f"Error loading reviews from {path}: {e}")

        if movie_id is not None:
            # Single movie fetch
            yield from self._fetch_reviews_for_single_movie(movie_id)
        else:
            # Fetch all
            yield from fetch_all_reviews_from_folder(self.public_reviews_folder_path, PublicReview)
            yield from fetch_all_reviews_from_folder(self.expert_reviews_folder_path, ExpertReview)

    def _fetch_reviews_for_single_movie(self, movie_id: int) -> Iterator[Review]:
        public_path = self.public_reviews_folder_path / f"{movie_id}.yaml"
        expert_path = self.expert_reviews_folder_path / f"{movie_id}.yaml"

        if public_path.exists():
            try:
                data = YamlFile(path=public_path).load() or []
                yield from PublicReview.create_multiple(source=data, schema_type='NESTED')
            except Exception as e:
                self.logger.error(f"Error loading public reviews for movie {movie_id}: {e}")

        if expert_path.exists():
            try:
                data = YamlFile(path=expert_path).load() or []
                yield from ExpertReview.create_multiple(source=data, schema_type='NESTED')
            except Exception as e:
                self.logger.error(f"Error loading expert reviews for movie {movie_id}: {e}")

    @override
    def save_reviews(self, movie_id: int, data: list[Review]) -> None:
        self.public_reviews_folder_path.mkdir(parents=True, exist_ok=True)
        self.expert_reviews_folder_path.mkdir(parents=True, exist_ok=True)

        public_reviews = [r for r in data if isinstance(r, PublicReview)]
        expert_reviews = [r for r in data if isinstance(r, ExpertReview)]

        def save_review_to_file(
            reviews: list[Review], review_folder: Path, save_movie_id: int, review_type_string: str
        ) -> None:
            if reviews:
                path = review_folder / f"{save_movie_id}.yaml"
                serializable: list[ReviewSerializableData] = [r.as_serializable_dict() for r in reviews]
                try:
                    YamlFile(path=path).save(data=cast(list[dict], cast(object, serializable)))
                except Exception as e:
                    self.logger.error(f"Error saving {review_type_string} for movie {save_movie_id}: {e}")

        save_review_to_file(
            reviews=public_reviews,
            review_folder=self.public_reviews_folder_path,
            save_movie_id=movie_id,
            review_type_string='public reviews'
        )
        save_review_to_file(
            reviews=expert_reviews,
            review_folder=self.expert_reviews_folder_path,
            save_movie_id=movie_id,
            review_type_string='expert reviews')

    @override
    def fetch_week_data(self, movie_id: Optional[int] = None) -> Iterator[WeekData]:
        """
        Fetches aggregated weekly data for analysis.
        For YAML, this involves loading full movie data and computing in memory.
        """
        # Reuse fetch_movies(ALL) to get full data, then compute WeekData
        filters = {'id': movie_id} if movie_id is not None else None

        # Note: fetch_movies filters by 'name', not 'id'. We need to adapt or filter manually.
        # Since fetch_movies iterates all for ALL mode anyway, we can filter the iterator.

        movies_iter = self.fetch_movies(filters=filters, detail_level='ALL')

        for movie in movies_iter:
            if movie_id is not None and movie.id != movie_id:
                continue

            # Compute WeekData in memory
            week_data_list = WeekData.create_multiple_from_source_variable(
                movie_id=movie.id,
                weeks_data_source=movie.box_office,
                public_reviews_master_source=movie.public_reviews
            )
            yield from week_data_list

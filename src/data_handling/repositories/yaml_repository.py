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
from src.data_handling.movie_collections import MovieData
from data_handling.repositories.repository import MovieRepository
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
            movie.box_office = self.fetch_box_office(movie.id)

            reviews = self.fetch_reviews(movie.id)
            movie.public_reviews = [r for r in reviews if isinstance(r, PublicReview)]
            movie.expert_reviews = [r for r in reviews if isinstance(r, ExpertReview)]

        return movies

    def save_movie(self, movie: MovieData) -> None:
        """Saves a movie's data to YAML files."""
        if movie.box_office:
            self.save_box_office(movie.id, movie.box_office)

        all_reviews = []
        if movie.public_reviews:
            all_reviews.extend(movie.public_reviews)
        if movie.expert_reviews:
            all_reviews.extend(movie.expert_reviews)

        if all_reviews:
            self.save_reviews(movie.id, all_reviews)

    def fetch_movie_name_to_id_map(self) -> dict[str, int]:
        """Fetches a mapping of movie names to IDs from the index file."""
        movies = self._load_metadata_from_index()
        return {m.name: m.id for m in movies}

    def fetch_box_office(self, movie_id: int) -> list[BoxOffice]:
        path = self.box_office_folder_path / f"{movie_id}.yaml"
        if not path.exists():
            return []
        try:
            data = YamlFile(path=path).load() or []
            return BoxOffice.create_multiple(source=data, schema_type='NESTED')
        except Exception as e:
            self.logger.error(f"Error loading box office for movie {movie_id}: {e}")
            return []

    def save_box_office(self, movie_id: int, data: list[BoxOffice]) -> None:
        self.box_office_folder_path.mkdir(parents=True, exist_ok=True)
        path = self.box_office_folder_path / f"{movie_id}.yaml"
        serializable_data = [item.as_serializable_dict() for item in data]
        try:
            YamlFile(path=path).save(data=serializable_data)
        except Exception as e:
            self.logger.error(f"Error saving box office for movie {movie_id}: {e}")

    def fetch_reviews(self, movie_id: int) -> list[Review]:
        public_path = self.public_reviews_folder_path / f"{movie_id}.yaml"
        expert_path = self.expert_reviews_folder_path / f"{movie_id}.yaml"

        reviews: list[Review] = []

        if public_path.exists():
            try:
                data = YamlFile(path=public_path).load() or []
                reviews.extend(PublicReview.create_multiple(source=data, schema_type='NESTED'))
            except Exception as e:
                self.logger.error(f"Error loading public reviews for movie {movie_id}: {e}")

        if expert_path.exists():
            try:
                data = YamlFile(path=expert_path).load() or []
                reviews.extend(ExpertReview.create_multiple(source=data, schema_type='NESTED'))
            except Exception as e:
                self.logger.error(f"Error loading expert reviews for movie {movie_id}: {e}")

        return reviews

    def save_reviews(self, movie_id: int, data: list[Review]) -> None:
        self.public_reviews_folder_path.mkdir(parents=True, exist_ok=True)
        self.expert_reviews_folder_path.mkdir(parents=True, exist_ok=True)

        public_reviews = [r for r in data if isinstance(r, PublicReview)]
        expert_reviews = [r for r in data if isinstance(r, ExpertReview)]

        if public_reviews:
            path = self.public_reviews_folder_path / f"{movie_id}.yaml"
            serializable = [r.as_serializable_dict() for r in public_reviews]
            try:
                YamlFile(path=path).save(data=serializable)
            except Exception as e:
                self.logger.error(f"Error saving public reviews for movie {movie_id}: {e}")

        if expert_reviews:
            path = self.expert_reviews_folder_path / f"{movie_id}.yaml"
            serializable = [r.as_serializable_dict() for r in expert_reviews]
            try:
                YamlFile(path=path).save(data=serializable)
            except Exception as e:
                self.logger.error(f"Error saving expert reviews for movie {movie_id}: {e}")

from abc import ABC, abstractmethod
from typing import Iterator, Literal, Optional

from src.data_handling.box_office import BoxOffice
from src.data_handling.movie_collections import MovieData
from src.data_handling.reviews import Review


class MovieRepository(ABC):
    """
    Abstract base class for movie data access.

    Defines the contract for fetching and saving movie data, allowing for
    different storage backends (e.g., YAML files, Database).
    """

    @abstractmethod
    def setup_storage(self) -> None:
        """
        Prepares the storage backend for use.

        For YAML: Creates necessary directories and empty index files.
        For DB: Creates the database schema (tables, views, etc.) if not exists.
        """
        pass

    @abstractmethod
    def is_storage_occupied(self) -> bool:
        """
        Checks if the storage backend already contains data or structure.

        :return: True if storage is occupied/initialized, False otherwise.
        """
        pass

    @abstractmethod
    def fetch_movies(
        self,
        filters: Optional[dict] = None,
        detail_level: Literal['META', 'ALL'] = 'META'
    ) -> Iterator[MovieData]:
        """
        Fetches movies based on filters and detail level.

        :param filters: A dictionary of filters to apply (e.g., {'name': 'Avatar'}).
        :param detail_level: The level of detail to fetch.
                             'META': Only basic metadata (ID, Name).
                             'ALL': Full movie data including reviews and box office.
        :return: An iterator of MovieData objects.
        """
        pass

    @abstractmethod
    def save_movies(self, movies: list[MovieData]) -> None:
        """
        Saves a list of movies' data to the storage backend.

        This should persist all components of the movies (metadata, box office, reviews).
        Implementations should optimize for batch processing where possible.

        :param movies: A list of MovieData objects to save.
        """
        pass

    @abstractmethod
    def fetch_movie_name_to_id_map(self) -> dict[str, int]:
        """
        Fetches a mapping of movie names to their IDs.

        Useful for quick lookups and caching.

        :return: A dictionary where keys are movie names and values are movie IDs.
        """
        pass

    @abstractmethod
    def fetch_box_office(
        self, movie_id: Optional[int] = None, week_number: Optional[int] = None
    ) -> Iterator[BoxOffice]:
        """
        Fetches box office data.

        :param movie_id: The ID of the movie. If None, fetches box office data for ALL movies.
        :param week_number: The specific week number to fetch (1-based).
                            If provided, movie_id must also be provided.
        :return: An iterator of BoxOffice objects.
        :raises ValueError: If week_number is <= 0 or if movie_id is None when week_number is provided.
        """
        pass

    @abstractmethod
    def save_box_office(self, movie_id: int, data: list[BoxOffice]) -> None:
        """
        Batch saves box office data for a specific movie.

        :param movie_id: The ID of the movie.
        :param data: A list of BoxOffice objects to save.
        """
        pass

    @abstractmethod
    def fetch_reviews(self, movie_id: Optional[int] = None) -> Iterator[Review]:
        """
        Fetches reviews (including replies for public reviews).

        :param movie_id: The ID of the movie. If None, fetches reviews for ALL movies.
        :return: An iterator of Review objects (PublicReview and ExpertReview).
        """
        pass

    @abstractmethod
    def save_reviews(self, movie_id: int, data: list[Review]) -> None:
        """
        Batch saves reviews and their replies for a specific movie.

        :param movie_id: The ID of the movie.
        :param data: A list of Review objects to save.
        """
        pass

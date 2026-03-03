from logging import Logger

from mysql.connector.cursor import MySQLCursor

from src.core.logging_manager import LoggingManager
from src.data_handling.movie_collections import MovieData


class MovieRepository:
    """
    Handles database operations for the 'movies' table.
    """

    def __init__(self, cursor: MySQLCursor):
        self.cursor = cursor
        self.logger: Logger = LoggingManager().get_logger('root')

    def save(self, movie: MovieData) -> None:
        """
        Inserts or updates a movie record and verifies the insertion.

        :param movie: The MovieData object containing movie information.
        :raises RuntimeError: If the insertion cannot be verified.
        """
        self.cursor.execute(
            """
            INSERT INTO `movies` (`movie_id`, `movie_name`) VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE `movie_name` = VALUES(`movie_name`)
            """,
            (movie.id, movie.name)
        )

        if not self._exists(movie.id):
            raise RuntimeError(f"Verification SELECT failed for movie ID {movie.id}")

    def _exists(self, movie_id: int) -> bool:
        """Checks if a movie exists by ID."""
        self.cursor.execute("SELECT movie_id FROM `movies` WHERE `movie_id` = %s", (movie_id,))
        return self.cursor.fetchone() is not None
from logging import Logger

from mysql.connector.cursor import MySQLCursor

from src.core.logging_manager import LoggingManager
from src.data_handling.box_office import BoxOffice


class BoxOfficeRepository:
    """
    Handles database operations for 'on_air_weeks' and 'box_office' tables.
    """

    def __init__(self, cursor: MySQLCursor):
        self.cursor = cursor
        self.logger: Logger = LoggingManager().get_logger('root')

    def save_all(self, movie_id: int, box_office_list: list[BoxOffice]) -> None:
        """
        Saves a list of box office records for a specific movie.

        :param movie_id: The ID of the movie.
        :param box_office_list: A list of BoxOffice objects.
        :raises RuntimeError: If any insertion cannot be verified.
        """
        if not box_office_list:
            return

        for bo in box_office_list:
            # Insert into on_air_weeks
            self.cursor.execute("""
                INSERT INTO `on_air_weeks` (`movie_id`, `start_date`)
                VALUES (%s, %s)
                ON DUPLICATE KEY UPDATE `on_air_weeks_id` = LAST_INSERT_ID(`on_air_weeks_id`)
            """, (movie_id, bo.start_date))

            on_air_week_id = self.cursor.lastrowid

            # Verify on_air_weeks
            if not self._on_air_week_exists(on_air_week_id):
                raise RuntimeError(f"Verification SELECT failed for on_air_weeks ID {on_air_week_id}")

            # Insert amount into box_office
            self.cursor.execute("""
                INSERT INTO `box_office` (`on_air_week_id`, `amount`)
                VALUES (%s, %s)
                ON DUPLICATE KEY UPDATE `amount` = VALUES(`amount`)
            """, (on_air_week_id, bo.box_office))

            # Verify box_office
            if not self._box_office_exists(on_air_week_id):
                raise RuntimeError(f"Verification SELECT failed for box_office with on_air_week_id {on_air_week_id}")

    def _on_air_week_exists(self, on_air_week_id: int) -> bool:
        self.cursor.execute("SELECT on_air_weeks_id FROM `on_air_weeks` WHERE `on_air_weeks_id` = %s", (on_air_week_id,))
        return self.cursor.fetchone() is not None

    def _box_office_exists(self, on_air_week_id: int) -> bool:
        self.cursor.execute("SELECT on_air_week_id FROM `box_office` WHERE `on_air_week_id` = %s", (on_air_week_id,))
        return self.cursor.fetchone() is not None
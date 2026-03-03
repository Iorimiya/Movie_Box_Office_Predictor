from pathlib import Path
from typing import Optional, TypedDict

from data_handling.movie_collections import MovieData, WeekData
from data_handling.reply import Reply
from data_handling.reviews import Review


class DatabaseConfig(TypedDict):
    address: str
    port: str
    user: str
    password: str


class DatabaseClient:

    def __init__(self, config: DatabaseConfig) -> None:
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def verify_admin_privileges(self) -> bool:
        pass

    def execute_script_from_file(self, path: Path):
        pass

    def execute_statement(self, statement: str):
        pass


class BoxOfficeRepository:
    def __init__(
        self,
        server_address: str,
        server_port: str,
        user_name: str,
        user_password: str,
        root_name: Optional[str],
        root_password: Optional[str]
    ) -> None:
        self.__user_client: DatabaseClient = DatabaseClient(config=DatabaseConfig(
            address=server_address,
            port=server_port,
            user=user_name,
            password=user_password
        ))
        self.__root_client: Optional[DatabaseClient] = None
        if (root_name and root_password) is not None:
            self.__root_client: DatabaseClient = DatabaseClient(config=DatabaseConfig(
                address=server_address,
                port=server_port,
                user=root_name,
                password=root_password
            ))

    def __build_where_clause(self, filters: dict, allowed_columns: set) -> tuple[str, list]:
        pass
        # return where_sql_string, parameters_list

    def have_root_privilege(self) -> bool:
        return self.__root_client is not None

    def fetch_reply(self, dataset_name: str, reply_id: str) -> Reply:
        pass

    def fetch_replies_by_dataset(self, dataset_name: str) -> list[Reply]:
        pass

    def fetch_replies_by_review(self, dataset_name: str, review_id: str) -> list[Reply]:
        pass

    def fetch_replies_by_movie(self, dataset_name: str, movie_id: str) -> list[Reply]:
        pass

    def save_replies(self, dataset_name: str, review_id: str, replies: list[Reply]) -> bool:
        pass

    def fetch_review(self, dataset_name: str, review_id: str) -> Review:
        pass

    def fetch_reviews_by_dataset(self, dataset_name: str) -> list[Review]:
        pass

    def fetch_reviews_by_movie(self, dataset_name: str, movie_id: str) -> list[Review]:
        pass

    def save_reviews(self, dataset_name: str, movie_id: str, reviews: list[Review]) -> bool:
        pass

    def fetch_movie(self, dataset_name: str, movie_id: str) -> MovieData:
        pass

    def fetch_movies_by_dataset(self, dataset_name: str, filters: Optional[dict]) -> list[MovieData]:
        pass

    def save_movies(self, datasets_name: str, movies: MovieData) -> bool:
        pass

    def fetch_week_data(self, dataset_name: str) -> list[WeekData]:
        pass

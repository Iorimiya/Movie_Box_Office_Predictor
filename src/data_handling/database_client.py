from pathlib import Path
from typing import cast, Optional, TypedDict, Type

from mysql.connector import connect, Error as DBError, MySQLConnection
from mysql.connector.cursor import MySQLCursor
from mysql.connector.errorcode import ER_ACCESS_DENIED_ERROR, ER_BAD_DB_ERROR



class DatabaseConfig(TypedDict, total=False):
    """
    Represents the configuration for a database connection.
    All fields are optional to allow for partial overrides.
    """
    address: str
    port: str
    user: str
    password: str



class DatabaseClient:
    """
    A client for connecting to a database, handling connection and session management.
    Uses mysql-connector-python.
    """

    def __init__(self, config: DatabaseConfig) -> None:
        """
        Initializes the DatabaseClient with connection configuration.
        Does not establish a connection immediately.

        :param config: A dictionary containing database connection details
                       (address, port, user, password, database_name).
        """
        self.__config = config
        self.__connection: Optional[MySQLConnection] = None
        self.__cursor: Optional[MySQLCursor] = None
        self.__database_name: Optional[str] = None

    @property
    def database_name(self) -> Optional[str]:
        """Gets the current database name."""
        return self.__database_name

    @database_name.setter
    def database_name(self, value: Optional[str]) -> None:
        """Sets the database name to be used for the connection."""
        self.__database_name = value

    def __enter__(self):
        """
        Establishes a database connection and creates a cursor.
        This is called when entering a 'with' statement.

        :return: The DatabaseClient instance itself.
        :raises DBError: If the connection to the database fails.
        """
        try:
            self.__connection = cast(MySQLConnection, connect(
                user=self.__config["user"],
                password=self.__config["password"],
                host=self.__config["address"],
                port=self.__config["port"],
                database=self.__database_name
            ))
            self.__cursor = self.__connection.cursor(dictionary=True)
            return self
        except DBError as err:
            if err.errno == ER_ACCESS_DENIED_ERROR:
                print("Something is wrong with your user name or password")
            elif err.errno == ER_BAD_DB_ERROR:
                print("Database does not exist")
            else:
                print(f"Error connecting to MySQL/MariaDB: {err}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Closes the cursor and connection, committing or rolling back the transaction.
        This is called when exiting a 'with' statement.
        """
        if self.__connection:
            try:
                if exc_type:
                    print(f"An exception occurred. Rolling back transaction.")
                    self.__connection.rollback()
                else:
                    self.__connection.commit()
            finally:
                if self.__cursor:
                    self.__cursor.close()
                self.__connection.close()
                self.__connection = None
                self.__cursor = None

    @classmethod
    def build_where_clause(
        cls: Type['DatabaseClient'], filters: dict, allowed_columns: Optional[set] = None
    ) -> tuple[str, list]:
        if not filters:
            return "", []
        conditions = []
        parameters = []
        allowed_operators = {'=', '<', '>', '<=', '>=', '!=', 'LIKE', 'IN'}
        for column, value in filters.items():
            if allowed_columns is not None and column not in allowed_columns:
                raise ValueError(f"Filtering by column '{column}' is not allowed.")
            operator = '='
            param_value = value
            if isinstance(value, tuple) and len(value) == 2:
                op, val = value
                if op.upper() in allowed_operators:
                    operator = op.upper()
                    param_value = val
                else:
                    raise ValueError(f"Operator '{op}' is not allowed.")
            if operator == 'IN':
                if not isinstance(param_value, (list, tuple)):
                    raise ValueError("Value for 'IN' operator must be a list or tuple.")
                placeholders = ', '.join(['%s'] * len(param_value))
                conditions.append(f"{column} IN ({placeholders})")
                parameters.extend(param_value)
            elif param_value is None:
                conditions.append(f"{column} IS NULL")
            else:
                conditions.append(f"{column} {operator} %s")
                parameters.append(param_value)
        where_clause = " AND ".join(conditions)
        return where_clause, parameters

    def verify_admin_privileges(self) -> bool:
        if not self.__connection or not self.__cursor:
            raise RuntimeError("Database operations must be performed within a 'with' block.")
        try:
            self.__cursor.execute("SHOW GRANTS FOR CURRENT_USER()")
            grants = self.__cursor.fetchall()
            has_create_db = False
            has_grant_option = False
            for grant_row in grants:
                grant_string = list(grant_row.values())[0].upper()
                if "ALL PRIVILEGES ON *.*" in grant_string or "CREATE" in grant_string:
                    has_create_db = True
                if "WITH GRANT OPTION" in grant_string:
                    has_grant_option = True
            return has_create_db and has_grant_option
        except DBError as e:
            print(f"Error checking privileges: {e}")
            return False

    def execute_script_from_file(self, path: Path) -> bool:
        if not self.__connection or not self.__cursor:
            raise RuntimeError("Database operations must be performed within a 'with' block.")
        if not path.exists():
            raise FileNotFoundError(f"SQL file not found: {path}")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                script_content = f.read()
            # mysql-connector handles multiple statements if split by semicolon
            for _ in self.__cursor.execute(script_content, multi=True):
                pass
            return True
        except (DBError, IOError) as e:
            print(f"Error executing script from {path}: {e}")
            return False

    def execute_statement(
        self,
        statement: str,
        parameters: Optional[tuple | list] = None,
        get_last_id: bool = False
    ) -> Optional[list[dict] | bool | int]:
        if not self.__connection or not self.__cursor:
            raise RuntimeError("Database operations must be performed within a 'with' block.")
        is_select_query = statement.strip().upper().startswith("SELECT")
        try:
            self.__cursor.execute(statement, parameters)
            if is_select_query:
                return self.__cursor.fetchall()
            elif get_last_id:
                return self.__cursor.lastrowid
            else:
                return True
        except DBError as e:
            print(f"Error executing statement: {e}")
            return None if is_select_query else False

    def execute_many(
        self,
        statement: str,
        parameters: list[tuple | list]
    ) -> bool:
        """
        Executes a SQL statement multiple times with different parameters.

        Should be called within a 'with' block.

        :param statement: The SQL statement to execute.
        :param parameters: A list of parameter tuples/lists.
        :return: True on success, False on error.
        :raises RuntimeError: If called outside of a 'with' block.
        """
        if not self.__connection or not self.__cursor:
            raise RuntimeError("Database operations must be performed within a 'with' block.")

        try:
            self.__cursor.executemany(statement, parameters)
            # 注意：executemany 通常用於 INSERT/UPDATE，不需要 fetchall
            return True
        except DBError as e:
            print(f"Error executing batch statement: {e}")
            return False

    def select(
        self,
        table_name: str,
        filters: Optional[dict] = None,
        allowed_columns: Optional[set] = None,
        columns: str = "*"
    ) -> list[dict]:
        """
        通用的 SELECT 方法。
        必須在 with block 中呼叫。
        """
        if not self.__connection or not self.__cursor:
            raise RuntimeError("Database operations must be performed within a 'with' block.")

        where_clause, parameters = self.build_where_clause(filters or {}, allowed_columns)
        query = f"SELECT {columns} FROM {table_name}"
        if where_clause:
            query += f" WHERE {where_clause}"

        return self.execute_statement(query, parameters) or []


# # TODO: 將結果化為物件改為create_multiple
#
# class BoxOfficeRepository:
#     ALLOWED_REVIEW_FILTER_COLUMNS = {'created_at', 'type', 'movie_id'}
#     ALLOWED_MOVIE_FILTER_COLUMNS = {'name'}
#     ALLOWED_REPLY_FILTER_COLUMNS = {'id', 'type', 'created_at', 'review_id'}
#
#     #     ALLOWED_WEEK_FILTER_COLUMNS = {'week_number', 'amount'}
#
#     def __init__(
#         self,
#         server_address: str,
#         server_port: str,
#         user_name: str,
#         user_password: str,
#         root_name: Optional[str],
#         root_password: Optional[str]
#     ) -> None:
#         self.__user_client: DatabaseClient = DatabaseClient(config=DatabaseConfig(
#             address=server_address,
#             port=server_port,
#             user=user_name,
#             password=user_password
#         ))
#         self.__root_client: Optional[DatabaseClient] = None
#         if root_name and root_password:
#             self.__root_client = DatabaseClient(config=DatabaseConfig(
#                 address=server_address,
#                 port=server_port,
#                 user=root_name,
#                 password=root_password
#             ))
#
#     def have_root_privilege(self) -> bool:
#         return self.__root_client is not None
#
#     def fetch_replies(self, dataset_name: str, filters: Optional[dict] = None) -> list[Reply]:
#         self.__user_client.database_name = dataset_name
#         with self.__user_client as db:
#             results = db.select(
#                 table_name="replies",
#                 filters=filters,
#                 allowed_columns=self.ALLOWED_REPLY_FILTER_COLUMNS
#             )
#             # noinspection PyTypeChecker
#             return Reply.create_multiple(source=results, schema_type='FLAT')
#
#     def fetch_reply(self, dataset_name: str, reply_id: int) -> Optional[Reply]:
#         replies: list[Reply] = self.fetch_replies(dataset_name=dataset_name, filters={'id': reply_id})
#         return replies[0] if replies else None
#
#     def fetch_replies_by_review_id(
#         self, dataset_name: str, review_id: int, filters: Optional[dict] = None
#     ) -> list[Reply]:
#         return self.fetch_replies(
#             dataset_name=dataset_name, filters={**(filters or {}), 'review_id': review_id}
#         )
#
#     def fetch_replies_by_movie_id(
#         self, dataset_name: str, movie_id: int, filters: Optional[dict] = None
#     ) -> list[Reply]:
#
#
#         self.__user_client.database_name = dataset_name
#         with self.__user_client as db:
#
#             results = db.select(
#                 table_name="movie_replies_view",
#                 filters={**(filters or{}), 'movie_id':movie_id},
#                 allowed_columns=self.ALLOWED_REPLY_FILTER_COLUMNS
#             )
#             # noinspection PyTypeChecker
#             return Reply.create_multiple(source=results, schema_type='FLAT')
#
#     def fetch_replies_by_movie_name(self, dataset_name: str, movie_name: str, filters: Optional[dict] = None) -> list[
#         Reply]:
#         movies = self.fetch_movie_id_name_map(dataset_name, filters={'name': movie_name})
#         if not movies:
#             return []
#         movie_id = movies[0]['id']
#         return self.fetch_replies_by_movie_id(dataset_name=dataset_name, movie_id=movie_id, filters=filters)
#
#     #     def save_replies(self, dataset_name: str, review_id: int, replies: list[Reply]) -> bool:
#     #         self.__user_client.database_name = dataset_name
#     #         with self.__user_client as db:
#     #             query = "INSERT INTO replies (review_id, author, content) VALUES (%s, %s, %s)"
#     #             params = [(review_id, r.author, r.content) for r in replies]
#     #             try:
#     #                 db.execute_many(statement=query, parameters=params)
#     #                 return True
#     #             except DBError as e:
#     #                 print(f"Failed to save replies for review {review_id}: {e}")
#     #                 return False
#
#     def fetch_public_reviews(
#         self, dataset_name: str, filters: Optional[dict] = None, detail_mode:Literal['META', 'ALL']='ALL'
#     ) -> list[PublicReview]:
#         self.__user_client.database_name = dataset_name
#         filters = filters or {}
#         with self.__user_client as db:
#             public_reviews_results = db.select(
#                 table_name="reviews",
#                 filters={**filters, "type": "public"},
#                 allowed_columns=self.ALLOWED_REVIEW_FILTER_COLUMNS
#             )
#
#             if not public_reviews_results:
#                 return []
#             if detail_mode == 'ALL':
#                 review_ids = [row['id'] for row in public_reviews_results]
#
#                 replies_results = []
#                 if review_ids:
#                     replies_results = db.select(
#                         table_name="replies",
#                         filters={'review_id': ('IN', review_ids)},
#                         allowed_columns=self.ALLOWED_REPLY_FILTER_COLUMNS
#                     )
#
#                 replies_by_review_id: dict[int, list[dict]] = {rid: [] for rid in review_ids}
#                 for reply_row in replies_results:
#                     r_id = reply_row['review_id']
#                     if r_id in replies_by_review_id:
#                         replies_by_review_id[r_id].append(reply_row)
#
#                 for review_row in public_reviews_results:
#                     review_row['replies'] = replies_by_review_id.get(review_row['id'], [])
#
#             # noinspection PyTypeChecker
#             return PublicReview.create_multiple(source=public_reviews_results, schema_type='FLAT')
#
#     def fetch_public_reviews_by_movie_id(
#         self,
#         dataset_name: str,
#         movie_id: int,
#         filters: Optional[dict] = None,
#         detail_mode:Literal['META', 'ALL']='ALL'
#     ) -> list[PublicReview]:
#         return self.fetch_public_reviews(
#             dataset_name=dataset_name,
#             filters={**(filters or {}), 'movie_id': movie_id},
#             detail_mode=detail_mode
#         )
#
#     def fetch_public_reviews_by_movie_name(
#         self,
#         dataset_name: str,
#         movie_name: str,
#         filters: Optional[dict] = None,
#         detail_mode:Literal['META', 'ALL']='ALL'
#     ) -> list[PublicReview]:
#         movies = self.fetch_movie_id_name_map(dataset_name, filters={'movie_title': movie_name})
#         if not movies:
#             return []
#         movie_id = movies[movie_name]
#         return self.fetch_public_reviews_by_movie_id(
#             dataset_name=dataset_name,
#             movie_id=movie_id,
#             filters=filters,
#             detail_mode=detail_mode
#         )
#
#     def fetch_movie_id_name_map(self, dataset_name: str, filters: Optional[dict] = None) -> dict[str,int]:
#         self.__user_client.database_name = dataset_name
#         with self.__user_client as db:
#             result = db.select(
#                 table_name="movies",
#                 filters=filters,
#                 allowed_columns=self.ALLOWED_MOVIE_FILTER_COLUMNS
#             )
#
#             return  {row['name']: row['id'] for row in result}
#
# #     def save_reviews(self, dataset_name: str, movie_id: int, reviews: list[Review]) -> bool:
# #         self.__user_client.database_name = dataset_name
# #         with self.__user_client as db:
# #             for review in reviews:
# #                 review_query = "INSERT INTO reviews (movie_id, author, content, rating) VALUES (%s, %s, %s, %s)"
# #                 review_params = (movie_id, review.author, review.content, review.rating)
# #                 review_id = db.execute_statement(review_query, review_params, get_last_id=True)
# #                 if not review_id:
# #                     print(f"Failed to save review by {review.author}. Rolling back.")
# #                     return False
# #                 if review.replies:
# #                     if not self.save_replies(dataset_name, review_id, review.replies):
# #                         print(f"Failed to save replies for review {review_id}. Rolling back.")
# #                         return False
# #         return True
# #
# #     def fetch_movie(self, dataset_name: str, movie_id: int) -> Optional[MovieData]:
# #         self.__user_client.database_name = dataset_name
# #         with self.__user_client as db:
# #             query = "SELECT * FROM movies WHERE id = %s"
# #             results = db.execute_statement(query, (movie_id,))
# #             if not results:
# #                 return None
# #             return MovieData(**results[0])
#     # TODO: 處利玩meta模式，處理Box office在來處理ALL
#
#
#     # def fetch_movies_by_dataset_name(
#     #     self,
#     #     dataset_name: str,
#     #     filters: Optional[dict] = None,
#     #     detail_mode:Literal['META', 'ALL']='ALL'
#     # ) -> list[MovieData]:
#     #     self.__user_client.database_name = dataset_name
#     #     with self.__user_client as db:
#     #         result = db.select(
#     #             table_name="movies",
#     #             filters=filters,
#     #             allowed_columns=self.ALLOWED_MOVIE_FILTER_COLUMNS
#     #         )
#     #         if detail_mode == 'ALL':
#     #             pass
#     #
#     #     return
# #
# #     def save_movies(self, dataset_name: str, movies: list[MovieData]) -> bool:
# #         self.__user_client.database_name = dataset_name
# #         with self.__user_client as db:
# #             query = "INSERT INTO movies (movie_title, year, genre, box_office) VALUES (%s, %s, %s, %s)"
# #             params = [(m.movie_title, m.year, m.genre, m.box_office) for m in movies]
# #             try:
# #
# #                 db.execute_many(statement=query, parameters=params)
# #                 return True
# #             except DBError as e:
# #                 print(f"Failed to save movies: {e}")
# #                 return False
# #
# #     def fetch_week_data_by_dataset_name(self, dataset_name: str, filters: Optional[dict] = None) -> list[WeekData]:
# #         self.__user_client.database_name = dataset_name
# #         weeks = []
# #         with self.__user_client as db:
# #             where_clause, parameters = self.__build_where_clause(filters or {}, self.ALLOWED_WEEK_FILTER_COLUMNS)
# #             query = "SELECT * FROM on_air_weeks"
# #             if where_clause:
# #                 query += f" WHERE {where_clause}"
# #             results = db.execute_statement(query, parameters)
# #             if results and isinstance(results, list):
# #                 for row in results:
# #                     weeks.append(WeekData(**row))
# #         return weeks
# #
# #     def fetch_week_data_by_movie_id(self, dataset_name: str, movie_id: int) -> list[WeekData]:
# #         self.__user_client.database_name = dataset_name
# #         weeks = []
# #         with self.__user_client as db:
# #             query = "SELECT * FROM on_air_weeks WHERE movie_id = %s"
# #             results = db.execute_statement(query, (movie_id,))
# #             if results and isinstance(results, list):
# #                 for row in results:
# #                     weeks.append(WeekData(**row))
# #         return weeks
# #
# #     def fetch_week_data_by_movie_name(self, dataset_name: str, movie_name: str) -> list[WeekData]:
# #         movies = self.fetch_movies_by_dataset_name(dataset_name, filters={'movie_title': movie_name})
# #         if not movies:
# #             return []
# #         movie_id = movies[0].id
# #         return self.fetch_week_data_by_movie_id(dataset_name, movie_id)

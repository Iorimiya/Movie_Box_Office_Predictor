from contextlib import contextmanager
from pathlib import Path
from typing import cast, Iterator, Optional, TypedDict, Type

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
                       (address, port, user, password).
        """
        self.__config = config
        self.__connection: Optional[MySQLConnection] = None
        self.__cursor: Optional[MySQLCursor] = None

    @contextmanager
    def connection(self, database_name: Optional[str] = None) -> Iterator['DatabaseClient']:
        """
        A context manager that provides a database connection.

        Establishes a connection upon entering the 'with' block and automatically
        handles commit, rollback, and closing of the connection.

        :param database_name: The name of the database to connect to.
        :yields: The DatabaseClient instance itself, ready for operations.
        :raises DBError: If the connection to the database fails.
        """
        try:
            self.__connection = cast(MySQLConnection, connect(
                user=self.__config["user"],
                password=self.__config["password"],
                host=self.__config["address"],
                port=self.__config["port"],
                database=database_name
            ))
            self.__cursor = self.__connection.cursor(dictionary=True)
            yield self
            self.__connection.commit()
        except DBError as err:
            if self.__connection:
                self.__connection.rollback()

            if err.errno == ER_ACCESS_DENIED_ERROR:
                print("Something is wrong with your user name or password")
            elif err.errno == ER_BAD_DB_ERROR:
                print(f"Database '{database_name}' does not exist")
            else:
                print(f"Error connecting to MySQL/MariaDB: {err}")
            raise
        finally:
            if self.__connection:
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
            raise

    def execute_statement(
        self,
        statement: str,
        parameters: Optional[tuple | list] = None,
        get_last_id: bool = False
    ) -> Optional[list[dict] | bool | int]:
        if not self.__connection or not self.__cursor:
            raise RuntimeError("Database operations must be performed within a 'with' block.")

        statement_upper = statement.strip().upper()
        # Support SELECT, SHOW, DESCRIBE, EXPLAIN as queries that return rows
        is_result_query = statement_upper.startswith(("SELECT", "SHOW", "DESCRIBE", "EXPLAIN"))

        try:
            self.__cursor.execute(statement, parameters)
            if is_result_query:
                return self.__cursor.fetchall()
            elif get_last_id:
                return self.__cursor.lastrowid
            else:
                return True
        except DBError as e:
            print(f"Error executing statement: {e}")
            return None if is_result_query else False

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

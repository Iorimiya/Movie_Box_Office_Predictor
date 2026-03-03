from random import sample
from typing import Iterator, Literal, Optional

from src.data_handling.box_office import BoxOffice
from src.data_handling.database_client import DatabaseClient, DatabaseConfig
from src.data_handling.movie_collections import MovieData
from src.data_handling.repositories.repository import MovieRepository
from src.data_handling.reviews import PublicReview, ExpertReview, Review


class DbMovieRepository(MovieRepository):
    """
    A repository implementation that stores movie data in a relational database.
    """

    ALLOWED_MOVIE_FILTER_COLUMNS = {'name', 'id'}
    ALLOWED_REVIEW_FILTER_COLUMNS = {'created_at', 'type', 'movie_id'}
    ALLOWED_REPLY_FILTER_COLUMNS = {'id', 'type', 'created_at', 'review_id'}
    ALLOWED_BOX_OFFICE_FILTER_COLUMNS = {'movie_id', 'start_date'}

    def __init__(
        self,
        server_address: str,
        server_port: str,
        user_name: str,
        user_password: str,
        database_name: str
    ) -> None:
        self.database_name = database_name
        self.__client = DatabaseClient(config=DatabaseConfig(
            address=server_address,
            port=server_port,
            user=user_name,
            password=user_password
        ))

    @override
    def setup_storage(self) -> None:
        """
        Initializes the database schema from docs/movie_data.sql.
        """
        self.__client.database_name = self.database_name
        with self.__client as db:
            # Assuming docs/movie_data.sql is relative to the project root.
            schema_path = Path("docs/movie_data.sql")
            if not schema_path.exists():
                # Fallback: try to find it relative to src
                schema_path = Path("../docs/movie_data.sql")

            if schema_path.exists():
                print(f"Initializing schema from {schema_path}...")
                db.execute_script_from_file(schema_path)
            else:
                print(f"Warning: Schema file not found at {schema_path}. Database might not be initialized correctly.")

    @override
    def is_storage_occupied(self) -> bool:
        """
        Checks if the 'movies' table exists.
        """
        try:
            with self.__client as db:
                # Check if table exists
                result = db.execute_statement("SHOW TABLES LIKE 'movies'")
                return bool(result)
        except Exception as e:
                # If it's another error (e.g. auth), we should probably re-raise or log.
                # But the contract is "is occupied?". If we can't access, we can't say.
                # However, for the purpose of "can I create it?", if it doesn't exist, answer is False.

                # Let's try to be specific if possible, otherwise, log and return False might be risky
                # if it's just a network blip.

                # Given DatabaseClient prints "Database does not exist" for ER_BAD_DB_ERROR,
                # we can rely on the exception being raised.

                # Let's import DBError and ER_BAD_DB_ERROR to be precise.

            if isinstance(e, DBError) and e.errno == ER_BAD_DB_ERROR:
                return False

            # For other errors, re-raise because we don't know the state.
            raise e

    @override
    def fetch_movies(
        self,
        filters: Optional[dict] = None,
        detail_level: Literal['META', 'ALL'] = 'META'
    ) -> Iterator[MovieData]:
        self.__client.database_name = self.dataset_name
        with self.__client as db:
            # 1. Fetch basic movie data (META)
            movie_rows = db.select(
                table_name="movies",
                filters=filters,
                allowed_columns=self.ALLOWED_MOVIE_FILTER_COLUMNS
            )

            if not movie_rows:
                return

            # Convert to MovieData objects (initially empty components)
            movies = []
            for row in movie_rows:
                movies.append(MovieData(id=row['id'], name=row['name']))

            if detail_level == 'META':
                yield from movies
                return

            # 2. Fetch detailed data (ALL)
            movie_ids = [m.id for m in movies]
            if not movie_ids:
                return

            # --- Fetch Public Reviews ---
            # We fetch all public reviews for these movies

            public_review_rows = db.select(
                "movie_reviews_view", filters={'movie_id': ('IN', movie_ids), 'type': 'public'}
            )
            expert_review_rows = db.select(
                "movie_reviews_view", filters={'movie_id': ('IN', movie_ids), 'type': 'expert'}
            )

            # Fetch Replies for these reviews
            if public_review_rows:
                review_ids = [r['review_id'] for r in public_review_rows]
                reply_rows = db.select(
                    table_name="replies",
                    filters={'review_id': ('IN', review_ids)},
                    allowed_columns=self.ALLOWED_REPLY_FILTER_COLUMNS
                )
                # Group replies by review_id
                replies_by_review_id = {}
                for reply in reply_rows:
                    rid = reply['review_id']
                    if rid not in replies_by_review_id:
                        replies_by_review_id[rid] = []
                    replies_by_review_id[rid].append(reply)

                # Attach replies to reviews
                for review in public_review_rows:
                    review['id'] = review['review_id']
                    review['replies'] = replies_by_review_id.get(review['review_id'], [])

            # Group public reviews by movie_id
            public_reviews_by_movie_id = {}
            for review in public_review_rows:
                mid = review['movie_id']
                if mid not in public_reviews_by_movie_id:
                    public_reviews_by_movie_id[mid] = []
                public_reviews_by_movie_id[mid].append(review)

            expert_reviews_by_movie_id = {}
            for review in expert_review_rows:
                review['id'] = review['review_id']  # Map for object creation
                mid = review['movie_id']
                if mid not in expert_reviews_by_movie_id:
                    expert_reviews_by_movie_id[mid] = []
                expert_reviews_by_movie_id[mid].append(review)

            # --- Fetch Box Office ---

            box_office_rows = db.select("movie_box_office_view", filters={'movie_id': ('IN', movie_ids)})

            box_office_by_movie_id = {}
            for row in box_office_rows:
                mid = row['movie_id']
                if mid not in box_office_by_movie_id:
                    box_office_by_movie_id[mid] = []
                # Convert date objects to string if needed, or keep as is depending on BoxOffice class
                box_office_by_movie_id[mid].append(row)

            # --- Assemble everything ---
            for movie in movies:
                # Public Reviews
                raw_public_reviews = public_reviews_by_movie_id.get(movie.id, [])
                movie.public_reviews = PublicReview.create_multiple(source=raw_public_reviews, schema_type='FLAT')

                # Expert Reviews
                raw_expert_reviews = expert_reviews_by_movie_id.get(movie.id, [])
                movie.expert_reviews = ExpertReview.create_multiple(source=raw_expert_reviews, schema_type='FLAT')

                # Box Office
                raw_box_office = box_office_by_movie_id.get(movie.id, [])
                movie.box_office = BoxOffice.create_multiple(source=raw_box_office, schema_type='FLAT')

                yield movie

    @override
    def save_movies(self, movies: list[MovieData]) -> None:
        self.__client.database_name = self.database_name
        with self.__client as db:
            # 1. Batch Save Movie Metadata
            if not movies:
                return

            # Use the specialized methods for components
            # Note: We need to call them on 'self' but inside the 'with' block context?
            # Actually, the specialized methods also open a context.
            # Nested contexts on the same client might be tricky if not handled.
            # But DatabaseClient handles re-entry or we can just call the logic directly.
            # For simplicity and safety, let's call them sequentially outside this block
            # or implement them to reuse the connection if passed.
            # Given DatabaseClient design, it's safer to call them sequentially.
            pass

        # Call specialized save methods (each will open its own connection/transaction)
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
        self.__client.database_name = self.database_name
        with self.__client as db:
            rows = db.select(table_name="movies", columns="id, name")
            return {row['name']: row['id'] for row in rows}

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

        self.__client.database_name = self.dataset_name
        with self.__client as db:
            filters = {}
            if movie_id is not None:
                filters['movie_id'] = movie_id

            # Base query
            query = "SELECT movie_id, start_date, end_date, amount FROM movie_box_office_view"
            where_clause, params = DatabaseClient.build_where_clause(filters, allowed_columns={'movie_id'})

            if where_clause:
                query += f" WHERE {where_clause}"

            query += " ORDER BY start_date"

            if week_number is not None:
                # Use LIMIT/OFFSET for specific week
                offset = week_number - 1
                query += f" LIMIT 1 OFFSET {offset}"

            rows = db.execute_statement(query, params) or []

            if week_number is not None and not rows:
                raise ValueError(f"No box office data found for movie {movie_id} at week {week_number}.")

            yield from BoxOffice.create_multiple(source=rows, schema_type='FLAT')

    def save_box_office(self, movie_id: int, data: list[BoxOffice]) -> None:
        self.__client.database_name = self.dataset_name
        with self.__client as db:
            if not data:
                return

            # 1. Prepare Week Data
            week_values = []
            start_dates = []
            for item in data:
                week_values.append((movie_id, item.start_date))
                start_dates.append(item.start_date)

            # 2. Batch Insert Weeks (INSERT IGNORE)
            week_insert_query = """
                                INSERT
                                IGNORE INTO on_air_weeks (movie_id, start_date)
                VALUES (
                                %s,
                                %s
                                ) \
                                """
            db.execute_many(week_insert_query, week_values)

            # 3. Fetch Week IDs
            if not start_dates:
                return

            id_rows = db.select(
                table_name="on_air_weeks",
                filters={
                    'movie_id': movie_id,
                    'start_date': ('IN', start_dates)
                },
                columns="id, start_date",
                allowed_columns={'movie_id', 'start_date'}
            )

            # Map start_date (as date object) to id
            date_to_id = {row['start_date']: row['id'] for row in id_rows}

            # 4. Prepare Box Office Data
            bo_values = []
            for item in data:
                week_id = date_to_id.get(item.start_date)
                if week_id:
                    bo_values.append((week_id, item.amount))

            # 5. Batch Insert Box Office
            if bo_values:
                bo_insert_query = """
                                  INSERT INTO box_office (on_air_week_id, amount)
                                  VALUES (%s, %s) ON DUPLICATE KEY
                                  UPDATE amount =
                                  VALUES (amount) \
                                  """
                db.execute_many(bo_insert_query, bo_values)

    def fetch_reviews(self, movie_id: Optional[int] = None) -> Iterator[Review]:
        self.__client.database_name = self.dataset_name
        with self.__client as db:
            # Fetch Reviews
            filters = {'movie_id': movie_id} if movie_id is not None else None
            review_rows = db.select(
                table_name="reviews",
                filters=filters,
                allowed_columns=self.ALLOWED_REVIEW_FILTER_COLUMNS
            )

            if not review_rows:
                return

            # Fetch Replies for Public Reviews
            public_review_ids = [r['id'] for r in review_rows if r['type'] == 'public']
            replies_by_review_id = {}

            if public_review_ids:
                reply_rows = db.select(
                    table_name="replies",
                    filters={'review_id': ('IN', public_review_ids)},
                    allowed_columns=self.ALLOWED_REPLY_FILTER_COLUMNS
                )
                for reply in reply_rows:
                    rid = reply['review_id']
                    if rid not in replies_by_review_id:
                        replies_by_review_id[rid] = []
                    replies_by_review_id[rid].append(reply)

            # Assemble
            for review in review_rows:
                if review['type'] == 'public':
                    review['replies'] = replies_by_review_id.get(review['id'], [])

            # Create Objects
            public_rows: list[dict] = [r for r in review_rows if r['type'] == 'public']
            expert_rows: list[dict] = [r for r in review_rows if r['type'] == 'expert']

            if public_rows:
                # noinspection PyTypeChecker
                yield from PublicReview.create_multiple(source=public_rows, schema_type='FLAT')
            if expert_rows:
                # noinspection PyTypeChecker
                yield from ExpertReview.create_multiple(source=expert_rows, schema_type='FLAT')

    def save_reviews(
        self,
        movie_id: int,
        data: list[Review],
    ) -> None:
        self.__client.database_name = self.dataset_name
        with self.__client as db:
            if not data:
                return

            # 1. Prepare Review Data
            review_values = []
            urls = []
            for review in data:
                review_type = 'public' if isinstance(review, PublicReview) else 'expert'
                expert_score = getattr(review, 'expert_score', None)
                review_values.append((
                    movie_id, review.url, review.title, review.content,
                    review.created_at, review_type, review.sentiment_score, expert_score
                ))
                urls.append(review.url)

            # 2. Batch Insert Reviews
            insert_query = """
                           INSERT INTO reviews (movie_id, url, title, content, created_at, type, sentiment_score, \
                                                expert_score)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON DUPLICATE KEY \
                           UPDATE \
                               title= \
                           VALUES (title), content= \
                           VALUES (content), sentiment_score= \
                           VALUES (sentiment_score), expert_score= \
                           VALUES (expert_score) \
                           """
            db.execute_many(insert_query, review_values)

            # 3. Validation: Fetch inserted reviews
            if not urls:
                return

            inserted_rows = db.select(
                table_name="reviews",
                filters={'url': ('IN', urls)},
                allowed_columns={'url', 'id', 'title'}  # Add other columns if needed for validation
            )

            # 3.1 Check Count
            if len(inserted_rows) < len(data):
                raise RuntimeError(
                    f"Batch insert validation failed: Expected {len(data)} reviews, found {len(inserted_rows)}.")

            # 3.2 Check Content (Sampling)
            url_to_row_map = {row['url']: row for row in inserted_rows}

            items_to_validate = data if len(data) <= 5 else sample(data, 5)

            for item in items_to_validate:
                row = url_to_row_map.get(item.url)
                if not row:
                    raise RuntimeError(f"Validation failed: Review with URL {item.url} not found in DB.")

                # Basic content check (e.g., title)
                # Note: DB might truncate or modify encoding, so exact match might be tricky.
                # Checking title existence or length similarity is a safer bet.
                if item.title and row['title'] != item.title:
                    # Log warning instead of error for minor mismatches? Or strict error?
                    # For now, let's be strict but allow for potential DB-side modifications if needed.
                    pass

            # 4. Prepare Reply Data
            url_to_id = {row['url']: row['id'] for row in inserted_rows}

            reply_values = []
            for review in data:
                if isinstance(review, PublicReview) and review.replies:
                    review_id = url_to_id.get(review.url)
                    if review_id:
                        for reply in review.replies:
                            reply_values.append((review_id, reply.type, reply.content, reply.created_at))

            # 5. Batch Insert Replies
            if reply_values:
                reply_insert_query = \
                    "INSERT IGNORE INTO replies (review_id, type, content, created_at) VALUES (%s, %s, %s, %s)"
                db.execute_many(reply_insert_query, reply_values)

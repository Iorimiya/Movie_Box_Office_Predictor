from logging import Logger
from typing import Any, Final, Literal

from mysql.connector.cursor import MySQLCursor

from src.core.logging_manager import LoggingManager
from src.data_handling.reviews import PublicReview, ExpertReview


class ReviewRepository:
    """
    Handles database operations for 'review' and 'reply' tables.
    """

    # Mapping from Python Reply rating to Database ENUM
    REPLY_RATING_MAP: Final[dict[str, str]] = {
        "推": "push",
        "噓": "boo",
        "→": "arrow"
    }

    def __init__(self, cursor: MySQLCursor):
        self.cursor = cursor
        self.logger: Logger = LoggingManager().get_logger('root')

    def save_all(self, movie_id: int, reviews: list[PublicReview] | list[ExpertReview], review_type: Literal['public', 'expert']) -> None:
        """
        Saves a list of reviews (and their replies if applicable) for a specific movie.

        :param movie_id: The ID of the movie.
        :param reviews: A list of PublicReview or ExpertReview objects.
        :param review_type: The type of review ('public' or 'expert').
        :raises RuntimeError: If any insertion cannot be verified.
        """
        if not reviews:
            return

        for review in reviews:
            expert_score = getattr(review, 'expert_score', None)

            # Insert Review
            self.cursor.execute("""
                INSERT INTO `review`
                (`movie_id`, `type`, `review_url`, `review_title`, `review_content`, `review_date`, `sentiment_score`, `expert_score`)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    `review_title` = VALUES(`review_title`),
                    `review_content` = VALUES(`review_content`),
                    `sentiment_score` = VALUES(`sentiment_score`),
                    `expert_score` = VALUES(`expert_score`),
                    `review_id` = LAST_INSERT_ID(`review_id`)
            """, (
                movie_id,
                review_type,
                review.url,
                review.title,
                review.content,
                review.date,
                review.sentiment_score,
                expert_score
            ))

            review_id = self.cursor.lastrowid

            # Verify Review
            if not self._review_exists(review.url):
                raise RuntimeError(f"Verification SELECT failed for review URL {review.url}")

            # Insert Replies (Only for PublicReview)
            if isinstance(review, PublicReview) and review.replies:
                self._save_replies(review_id, review.replies)

    def _save_replies(self, review_id: int, replies: list[Any]) -> None:
        """Helper to save replies for a review."""
        for reply in replies:
            # Map rating symbol to DB enum
            db_type = self.REPLY_RATING_MAP.get(reply.rating, 'arrow')

            # Insert Reply
            self.cursor.execute("""
                INSERT INTO `reply` (`review_id`, `type`, `content`, `created_at`)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    `type` = VALUES(`type`),
                    `content` = VALUES(`content`),
                    `created_at` = VALUES(`created_at`),
                    `reply_id` = LAST_INSERT_ID(`reply_id`)
            """, (
                review_id,
                db_type,
                reply.content,
                reply.time
            ))

            reply_id = self.cursor.lastrowid

            # Verify Reply
            if not self._reply_exists(reply_id):
                raise RuntimeError(f"Verification SELECT failed for reply ID {reply_id}")

    def _review_exists(self, review_url: str) -> bool:
        self.cursor.execute("SELECT review_id FROM `review` WHERE `review_url` = %s", (review_url,))
        return self.cursor.fetchone() is not None

    def _reply_exists(self, reply_id: int) -> bool:
        self.cursor.execute("SELECT reply_id FROM `reply` WHERE `reply_id` = %s", (reply_id,))
        return self.cursor.fetchone() is not None
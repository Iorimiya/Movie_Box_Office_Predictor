from dataclasses import dataclass, field
from datetime import date, timedelta
from itertools import chain
from logging import Logger
from typing import Final, Literal, Optional, Type, TypeAlias, TypeVar

from numpy import mean

from src.core.logging_manager import LoggingManager
from src.data_handling.box_office import BoxOffice, BoxOfficeRawData, BoxOfficeSerializableData
from src.data_handling.reviews import (
    ExpertReview, ExpertReviewSerializableData,
    PublicReview, PublicReviewRawData, PublicReviewSerializableData
)
from src.utilities.collection_utils import delete_duplicate

WeekDataReviewType = TypeVar('WeekDataReviewType', PublicReview, ExpertReview)

MovieComponentSerializableData: TypeAlias = \
    list[BoxOfficeSerializableData] | list[PublicReviewSerializableData] | list[ExpertReviewSerializableData]
MovieComponent: TypeAlias = list[BoxOffice] | list[PublicReview] | list[ExpertReview]
ComponentClassMapType: TypeAlias = dict[
    Literal['box_office', 'public_reviews', 'expert_reviews'],
    Type[BoxOffice] | Type[PublicReview] | Type[ExpertReview]
]
COMPONENT_CLASS_MAP: Final[ComponentClassMapType] = {
    'box_office': BoxOffice,
    'public_reviews': PublicReview,
    'expert_reviews': ExpertReview
}


@dataclass(kw_only=True)
class WeekData:
    movie_id: Optional[int]  # for recognize
    start_date: date  # for recognize
    end_date: date  # for recognize
    total_content_length: int
    total_title_length: int

    box_office: int
    box_office_last_week: int
    box_office_previous_week: int

    weekly_difference: int  # box office amount (current week - last week)
    previous_weekly_difference: int  # currently not used
    percentage_change_last_week: float  # currently not used
    weekly_box_office_trend: int  # currently not used

    average_sentiment_score: float

    total_public_review_count: int  # currently_not_used

    total_reply_count: int
    weekly_reply_count: int
    total_positive_reply_count: int
    weekly_positive_reply_count: int
    total_negative_reply_count: int
    weekly_negative_reply_count: int

    @staticmethod
    def _filter_review_by_week(
        reviews: list[WeekDataReviewType], start_date: date, end_date: date
    ) -> list[WeekDataReviewType]:
        """
        Filters a list of reviews to include only those within a specific date range.

        :param reviews: A list of reviews (PublicReview or ExpertReview) to filter.
        :param start_date: The start date of the filtering period (inclusive).
        :param end_date: The end date of the filtering period (inclusive).
        :return: A new list containing only the reviews that fall within the specified date range.
        """
        logger: Logger = LoggingManager().get_logger('root')
        filtered_review: list[WeekDataReviewType] = []
        try:
            filtered_review = [review for review in reviews if start_date <= review.created_at <= end_date]
        except Exception as e:
            logger.error(
                f"Error filtering reviews for week {start_date}-{end_date}: {e}"
            )
        return filtered_review

    @classmethod
    def create_multiple_from_source_variable(
        cls,
        weeks_data_source: list[BoxOfficeRawData] | list[BoxOffice],
        public_reviews_master_source: Optional[list[PublicReviewRawData] | list[PublicReview]] = None,
        movie_id: Optional[int] = None
    ) -> list['WeekData']:
        """
        Creates a list of WeekData objects from various sources.

        Each WeekData object represents a week's box office data and associated reviews.
        Reviews are filtered from the master sources to match each week's date range.

        :param movie_id: The source movie id, None if used in prediction.
        :param weeks_data_source: The source of week data (box office, start/end dates).
        :param public_reviews_master_source: An optional master source for all public reviews.
                                             If provided, reviews will be filtered and assigned to relevant weeks.
        :return: A list of created WeekData objects.
        :raises ValueError: If `weeks_data_source` is of an invalid type (propagated from underlying calls).
        """
        logger: Logger = LoggingManager().get_logger('root')

        # Ensure weeks_data_source is a list of BoxOffice objects
        all_box_office_instances: list[BoxOffice]
        if weeks_data_source and isinstance(weeks_data_source[0], BoxOffice):
             # noinspection PyTypeChecker
             all_box_office_instances = weeks_data_source
        else:
             # noinspection PyTypeChecker
             all_box_office_instances = BoxOffice.create_multiple(source=weeks_data_source, schema_type='NESTED')

        if not all_box_office_instances:
            logger.warning(
                f"No valid BoxOffice instances could be created from weeks_data_source. Cannot create WeekData.")
            return []

        # Ensure public_reviews_master_source is a list of PublicReview objects
        all_public_reviews: list[PublicReview] = []
        if public_reviews_master_source:
            if isinstance(public_reviews_master_source[0], PublicReview):
                 # noinspection PyTypeChecker
                 all_public_reviews = public_reviews_master_source
            else:
                 # noinspection PyTypeChecker
                 all_public_reviews = PublicReview.create_multiple(
                     source=public_reviews_master_source, schema_type='NESTED'
                 )

        box_office_map:dict[date, int] = {bo.start_date: bo.amount for bo in all_box_office_instances}

        return [cls(
            movie_id=movie_id,
            start_date=current_box_office_data.start_date,
            end_date=current_box_office_data.end_date,
            total_public_review_count=len(
                assembled_public_reviews := cls._filter_review_by_week(
                    reviews=all_public_reviews,
                    start_date=current_box_office_data.start_date,
                    end_date=current_box_office_data.end_date
                )
            ),
            total_content_length=sum(len(pr.content) for pr in assembled_public_reviews),
            total_title_length=sum(len(pr.title) for pr in assembled_public_reviews),
            box_office=(bo_now := current_box_office_data.amount),
            box_office_last_week=(
                bo_last := box_office_map.get(current_box_office_data.start_date - timedelta(days=7), 0)
            ),
            box_office_previous_week=(
                bo_prev := box_office_map.get(current_box_office_data.start_date - timedelta(days=14), 0)
            ),
            weekly_difference=(weekly_diff := bo_now - bo_last),
            previous_weekly_difference=bo_last - bo_prev,
            percentage_change_last_week=weekly_diff / bo_last if bo_last != 0 else 0,
            weekly_box_office_trend=1 if bo_now > bo_last else -1 if bo_now < bo_last else 0,

            average_sentiment_score=float(mean([
                pr.sentiment_score for pr in assembled_public_reviews
            ])) if assembled_public_reviews else 0.0,
            total_reply_count=sum([pr.reply_count for pr in assembled_public_reviews]),
            weekly_reply_count=len(
                weekly_reply := [
                    pr for rp in assembled_public_reviews for pr in rp.replies
                    if current_box_office_data.start_date <= pr.created_at <= current_box_office_data.end_date
                ]
            ),
            total_positive_reply_count=sum([pr.positive_reply_count for pr in assembled_public_reviews]),
            weekly_positive_reply_count=len([reply for reply in weekly_reply if reply.type == "boo"]),
            total_negative_reply_count=sum([pr.negative_reply_count for pr in assembled_public_reviews]),
            weekly_negative_reply_count=len([reply for reply in weekly_reply if reply.type == "push"]),
        ) for current_box_office_data in all_box_office_instances]


@dataclass(kw_only=True)
class MovieData:
    """
    Represents comprehensive data for a single movie.

    This includes its metadata (ID, name), and lists of all its
    box office records, public reviews, and expert reviews.

    :ivar id: The unique integer identifier of the movie.
    :ivar name: The name of the movie.
    :ivar box_office: A list of all box office records for the movie.
    :ivar public_reviews: A list of all public reviews for the movie.
    :ivar expert_reviews: A list of all expert reviews for the movie.
    """
    id: int
    name: str
    box_office: list[BoxOffice] = field(default_factory=list)
    public_reviews: list[PublicReview] = field(default_factory=list)
    expert_reviews: list[ExpertReview] = field(default_factory=list)

    @property
    def box_office_week_lens(self) -> int:
        """
        Returns the number of weeks for which box office data is available.

        :return: The count of ``BoxOffice`` records, or 0 if none exist.
        """
        return len(self.box_office) if self.box_office else 0

    @property
    def public_reply_count(self) -> int:
        """
        Returns the total number of replies across all available public reviews.

        :return: The sum of reply counts from all ``PublicReview`` objects,
                  or 0 if no public reviews exist.
        """
        return sum(public_review.reply_count for public_review in self.public_reviews)

    @property
    def public_review_count(self) -> int:
        """
        Returns the number of public reviews available for the movie.

        :return: The count of ``PublicReview`` objects, or 0 if none exist.
        """
        return len(self.public_reviews) if self.public_reviews else 0

    def update_box_office(self, update_method: Literal['REPLACE', 'EXTEND'], data: list[BoxOffice]) -> None:
        """
        Updates the movie's box office data.

        The existing box office data can be replaced or extended with the provided data.
        Duplicates are handled.

        :param update_method: How to update ('replace' or 'extend').
        :param data: A list of new BoxOffice instances.
        """
        self.__update_component(component_type='box_office', update_method=update_method, data=data)

    def update_public_reviews(self, update_method: Literal['REPLACE', 'EXTEND'], data: list[PublicReview]) -> None:
        """
        Updates the movie's public reviews data.

        The existing public reviews can be replaced or extended with the provided data.
        Duplicates are handled.

        :param update_method: How to update ('replace' or 'extend').
        :param data: A list of new PublicReview instances.
        """
        self.__update_component(component_type='public_reviews', update_method=update_method, data=data)

    def update_expert_reviews(self, update_method: Literal['REPLACE', 'EXTEND'], data: list[ExpertReview]) -> None:
        """
        Updates the movie's expert reviews data.

        The existing expert reviews can be replaced or extended with the provided data.
        Duplicates are handled.

        :param update_method: How to update ('replace' or 'extend').
        :param data: A list of new ExpertReview instances.
        """
        self.__update_component(component_type='expert_reviews', update_method=update_method, data=data)

    def __update_component(
        self,
        component_type: Literal['box_office', 'public_reviews', 'expert_reviews'],
        update_method: Literal['REPLACE', 'EXTEND'],
        data: MovieComponent
    ) -> None:
        """
        Internal helper to update a specific component's data list (box office, public reviews, or expert reviews).

        The component's data can be either replaced entirely or extended with new data.
        Duplicate items are removed after the operation.

        :param component_type: The type of component to update.
        :param update_method: The method of update ('REPLACE' or 'EXTEND').
        :param data: A list of new data items (BoxOffice, PublicReview, or ExpertReview instances).
        :raises ValueError: If an invalid `update_method` is provided.
        """
        logger: Logger = LoggingManager().get_logger("root")
        component_name_for_log: str = component_type.replace('_', ' ')
        incoming_data_count: int = len(data)

        logger.debug(
            f"Attempting to update {component_name_for_log} for movie ID {self.id} "
            f"using method '{update_method}' with {incoming_data_count} new items."
        )

        current_data_list: MovieComponent = getattr(self, component_type, [])
        original_data_count: int = len(current_data_list)

        match update_method:
            case 'REPLACE':
                deduplicated_new_data: MovieComponent = delete_duplicate(data)
                setattr(self, component_type, deduplicated_new_data)
                new_count: int = len(deduplicated_new_data)
                logger.debug(
                    f"Replaced {component_name_for_log} for movie ID {self.id}. "
                    f"Previous count: {original_data_count}, New count: {new_count}."
                )
            case 'EXTEND':
                logger.debug(
                    f"Extending {component_name_for_log} for movie ID {self.id}. "
                    f"Original count: {original_data_count}, Items to add: {incoming_data_count}."
                )
                combined_data: MovieComponent = current_data_list + data

                deduplicated_new_data: MovieComponent = delete_duplicate(combined_data)
                setattr(self, component_type, deduplicated_new_data)
                final_count: int = len(deduplicated_new_data)
                logger.debug(
                    f"Extended and deduplicated {component_name_for_log} for movie ID {self.id}. "
                    f"Final count: {final_count} (was {original_data_count}, added {incoming_data_count} before deduplication)."
                )


@dataclass(kw_only=True)
class MovieSessionData:
    """
    Represents session data for a movie over several weeks.

    This includes the movie's ID, name, and a list of WeekData objects
    representing its performance and reviews over consecutive weeks.

    :ivar id: The unique integer identifier of the movie.
    :ivar name: The name of the movie.
    :ivar weeks_data: A list of WeekData objects for the movie session.
    """
    id: int
    name: str
    weeks_data: list[WeekData]

    @staticmethod
    def _create_sliding_window_batches(items: list[BoxOffice], window_size: int) -> list[list[BoxOffice]]:
        """
        Creates sliding window batches from a list of items.

        :param items: The list of items to create batches from.
        :param window_size: The size of each batch (window).
        :return: A list of batches.
        """
        if not items or len(items) < window_size:
            return []

        return [items[i: i + window_size] for i in range(len(items) - window_size + 1)]

    @staticmethod
    def _filter_valid_batches(batches: list[list[BoxOffice]]) -> list[list[BoxOffice]]:
        """
        Filters a list of batches to keep only those where all weeks have non-zero box office.

        :param batches: A list of batches, where each batch is a list of BoxOffice objects.
        :return: A new list containing only the valid batches.
        """
        return [batch for batch in batches if all(map(lambda week: week.amount != 0, batch))]

    @classmethod
    def create_sessions_from_single_movie_data(
        cls, movie_data: 'MovieData', number_of_weeks: int
    ) -> list['MovieSessionData']:
        """
        Creates a list of MovieSessionData objects from a single MovieData instance.

        This is the core logic for segmenting a movie's full history into valid,
        fixed-length sessions.

        :param movie_data: A complete MovieData object for a single movie.
        :param number_of_weeks: The number of weeks each movie session should span.
        :return: A list of all valid MovieSessionData objects for the given movie.
        """
        logger: Logger = LoggingManager().get_logger("root")
        box_office_history: list[BoxOffice] = movie_data.box_office
        if len(box_office_history) < number_of_weeks:
            return []

        all_batches: list[list[BoxOffice]] = cls._create_sliding_window_batches(
            items=box_office_history,
            window_size=number_of_weeks
        )

        valid_batches: list[list[BoxOffice]] = cls._filter_valid_batches(batches=all_batches)

        if not valid_batches:
            logger.debug(
                f"No valid {number_of_weeks}-week sessions found after filtering for movie ID {movie_data.id}."
            )
            return []

        return [
            cls(
                id=movie_data.id,
                name=movie_data.name,
                weeks_data=WeekData.create_multiple_from_source_variable(
                    movie_id=movie_data.id,
                    weeks_data_source=single_batch,
                    public_reviews_master_source=movie_data.public_reviews
                )
            )
            for single_batch in valid_batches
        ]

    @classmethod
    def create_sessions_from_movie_data_list(
        cls, movie_data_list: list['MovieData'], number_of_weeks: int
    ) -> list['MovieSessionData']:
        """
        Creates MovieSessionData objects from a list of in-memory MovieData objects.

        This method iterates through each MovieData object and delegates the session
        creation to `create_sessions_from_single_movie_data`.

        :param movie_data_list: A list of MovieData objects to process.
        :param number_of_weeks: The number of weeks each movie session should span.
        :return: A flattened list of all valid MovieSessionData objects created.
        """
        logger: Logger = LoggingManager().get_logger("root")
        all_sessions: list['MovieSessionData'] = list(chain.from_iterable(
            cls.create_sessions_from_single_movie_data(
                movie_data=movie_data,
                number_of_weeks=number_of_weeks
            )
            for movie_data in movie_data_list
        ))
        logger.debug(
            f"Created a total of {len(all_sessions)} sessions from {len(movie_data_list)} movies."
        )
        return all_sessions

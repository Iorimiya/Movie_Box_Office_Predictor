from dataclasses import dataclass, field, fields, MISSING
from datetime import date
from itertools import chain
from logging import Logger
from pathlib import Path
from typing import get_args, Literal, Optional, Type, TypeVar, Union

import numpy as np

from src.core.logging_manager import LoggingManager
from src.core.project_config import ProjectConfig
from src.data_handling.file_io import CsvFile, YamlFile

SelfReview = TypeVar('SelfReview', bound='Review')
ReviewSubclass = TypeVar('ReviewSubclass', bound='Review')


@dataclass(kw_only=True)
class Review:
    """
    Represents a general review.

    :ivar url: The URL of the review.
    :ivar title: The title of the review.
    :ivar content: The content of the review.
    :ivar date: The date of the review.
    :ivar sentiment_score: The sentiment score of the review, if available.
    """
    url: str
    title: str
    content: str
    date: date
    sentiment_score: Optional[float] = None

    def _key(self) -> str:
        """
        Returns a unique key for the review.

        This key is used for hashing and equality checks. It defaults to the URL
        if available, otherwise it uses the review content.

        :returns: A unique string key for the review.
        """
        if self.url:
            return self.url
        else:
            return self.content

    def __hash__(self) -> int:
        """
        Returns the hash of the review based on its key.

        :returns: The hash value of the review.
        """
        return hash(self._key())

    def __eq__(self, other) -> bool:
        """
        Checks if this review is equal to another object.

        Equality is determined by comparing their unique keys, if the other object
        is also an instance of Review.

        :param other: The object to compare with.
        :returns: True if the objects are considered equal, False otherwise.
                  NotImplemented if the comparison is not supported for the given type.
        """
        if isinstance(other, Review):
            return self._key() == other._key()
        return NotImplemented

    @classmethod
    def _get_constructor_kwargs(cls: Type[SelfReview], data_item: dict) -> dict:
        """
        Prepares keyword arguments for the class constructor from a raw data dictionary.

        This method handles parsing for fields common to the Review class,
        such as date conversion and sentiment score handling.
        Subclasses should override this to add their specific fields and call
        `super()._get_constructor_kwargs(data_item)` to get the base arguments.

        :param data_item: The raw dictionary containing data for a review.
        :returns: A dictionary of keyword arguments suitable for instantiating the class.
        :raises ValueError: If required fields are missing or have invalid formats (e.g., date).
        """
        logger: Logger = LoggingManager().get_logger('root')
        cls_fields: set[str] = {f.name for f in fields(cls) if f.init}
        kwargs: dict[str, any] = {k: v for k, v in data_item.items() if k in cls_fields}

        date_val: str | date = data_item.get('date')
        if isinstance(date_val, str):
            try:
                kwargs['date'] = date.fromisoformat(date_val)
            except ValueError:
                logger.error(f"Invalid date format '{date_val}' in data: {data_item}.")
                raise ValueError(f"Invalid date format: {date_val}")
        elif isinstance(date_val, date):
            kwargs['date'] = date_val
        elif 'date' in cls_fields:
            logger.error(f"Required field 'date' is missing or has invalid type in data: {data_item}")
            raise ValueError(f"Missing or invalid type for 'date' in data: {data_item}")

        sentiment_val: float | str = data_item.get('sentiment_score')
        if sentiment_val is not None:
            try:
                kwargs['sentiment_score'] = float(sentiment_val)
            except (ValueError, TypeError):
                logger.warning(
                    f"Could not convert sentiment_score '{sentiment_val}' to int in data: {data_item}. Setting to None.")
                kwargs['sentiment_score'] = None
        elif 'sentiment_score' in cls_fields:
            kwargs['sentiment_score'] = None

        for f_name in ['url', 'title', 'content']:
            if f_name in cls_fields and f_name not in kwargs:
                logger.error(f"Required field '{f_name}' missing from prepared kwargs for data: {data_item}")
                raise ValueError(f"Required field '{f_name}' could not be prepared from data: {data_item}")
        return kwargs

    @classmethod
    def create_reviews(cls: Type[SelfReview], reviews_source: Path | YamlFile | list[dict]) -> list[SelfReview]:
        """
        Creates a list of Review (or subclass) objects from various sources.

        This method is generic and relies on `_get_constructor_kwargs` for field processing.
        It handles loading data from a file path, a YamlFile object, or a preloaded list of dictionaries.
        Invalid data items that cause errors during processing or instantiation are skipped and logged.

        :param reviews_source: The source of review data. Can be a Path to a YAML file,
                               a YamlFile object, or a list of dictionaries.
        :returns: A list of created Review (or subclass) instances.
        :raises ValueError: If `reviews_source` is of an invalid type.
        """
        logger: Logger = LoggingManager().get_logger('root')
        reviews_data_list: list[dict]
        if isinstance(reviews_source, Path):
            reviews_data_list = YamlFile(path=reviews_source).load()
            logger.info(f"Loading file \"{reviews_source}\" to create multiple {cls.__name__} objects.")
        elif isinstance(reviews_source, YamlFile):
            logger.info(f"Loading file \"{reviews_source.path}\" to create multiple {cls.__name__} objects.")
            reviews_data_list = reviews_source.load()
        elif isinstance(reviews_source, list) and all(
            isinstance(review_dict, dict) for review_dict in reviews_source):
            reviews_data_list = reviews_source
        else:
            raise ValueError(f"Invalid type for reviews_source: {type(reviews_source)}")

        required_raw_keys: set[str] = {
            f.name for f in fields(cls)
            if f.init and
               f.default is MISSING and
               f.default_factory is MISSING and
               not (getattr(f.type, '__origin__', None) is Union and type(None) in get_args(f.type))
        }

        def _process_single_item(data_dict: dict) -> Optional[SelfReview]:
            """
            Processes a single raw data dictionary to create a review instance.

            Validates required keys based on the class definition and handles exceptions
            during argument preparation and instance creation using `_get_constructor_kwargs`.

            :param data_dict: The raw dictionary for a single review.
            :returns: A review instance if successful, None otherwise.
            """
            current_missing_keys: set[str] = required_raw_keys - data_dict.keys()
            if current_missing_keys:
                logger.warning(
                    f"Skipping data for {cls.__name__} due to missing required raw keys: {current_missing_keys} in {data_dict}")
                return None

            try:
                constructor_kwargs: dict[str, any] = cls._get_constructor_kwargs(data_dict)
                instance: SelfReview = cls(**constructor_kwargs)
                return instance
            except (ValueError, TypeError) as e:
                logger.error(f"Error processing or creating {cls.__name__} instance from data {data_dict}: {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error creating {cls.__name__} instance from data {data_dict}: {e}")
                return None

        return list(filter(None.__ne__, map(_process_single_item, reviews_data_list)))


@dataclass(kw_only=True)
class PublicReview(Review):
    """
    Represents a public review, extending Review with a reply count.

    :ivar reply_count: The number of replies to this public review.
    """
    reply_count: int

    @classmethod
    def _get_constructor_kwargs(cls: Type[SelfReview], data_item: dict) -> dict:
        """
        Prepares keyword arguments for the PublicReview constructor.

        Extends the base Review's argument preparation by adding 'reply_count'.
        It expects 'reply_count' to be present in the `data_item`.

        :param data_item: The raw dictionary containing data for a public review.
        :returns: A dictionary of keyword arguments suitable for instantiating PublicReview.
        :raises ValueError: If 'reply_count' is missing or has an invalid format.
        """
        logger: Logger = LoggingManager().get_logger('root')
        kwargs: dict[str, any] = super()._get_constructor_kwargs(data_item)

        raw_reply_count: int | str = data_item.get('reply_count')
        if raw_reply_count is not None:
            try:
                kwargs['reply_count'] = int(raw_reply_count)
            except (ValueError, TypeError):
                logger.error(f"Invalid reply_count value '{raw_reply_count}' in data: {data_item}.")
                raise ValueError(f"Invalid reply_count value: {raw_reply_count}")
        else:
            logger.error(f"Required field 'reply_count' missing in data for PublicReview: {data_item}")
            raise ValueError(f"Missing 'reply_count' for PublicReview in data: {data_item}")
        return kwargs


@dataclass(kw_only=True)
class ExpertReview(Review):
    """
    Represents an expert review, extending Review with an expert score.

    :ivar expert_score: The score given by the expert.
    """
    expert_score: float

    @classmethod
    def _get_constructor_kwargs(cls: Type[SelfReview], data_item: dict) -> dict:
        """
        Prepares keyword arguments for the ExpertReview constructor.

        Extends the base Review's argument preparation by adding 'expert_score'.
        It expects 'expert_score' to be present in the `data_item`.

        :param data_item: The raw dictionary containing data for an expert review.
        :returns: A dictionary of keyword arguments suitable for instantiating ExpertReview.
        :raises ValueError: If 'expert_score' is missing or has an invalid format.
        """
        logger: Logger = LoggingManager().get_logger('root')
        kwargs: dict[str, any] = super()._get_constructor_kwargs(data_item)

        raw_expert_score: str | float = data_item.get('expert_score')
        if raw_expert_score is not None:
            try:
                kwargs['expert_score'] = float(raw_expert_score)
            except (ValueError, TypeError):
                logger.error(f"Invalid expert_score value '{raw_expert_score}' in data: {data_item}.")
                raise ValueError(f"Invalid expert_score value: {raw_expert_score}")
        else:
            logger.error(f"Required field 'expert_score' missing in data for ExpertReview: {data_item}")
            raise ValueError(f"Missing 'expert_score' for ExpertReview in data: {data_item}")
        return kwargs


@dataclass(kw_only=True)
class WeekData:
    """
    Represents data for a specific week of a movie's release.

    This includes the start and end dates of the week, the box office revenue
    for that week, and lists of public and expert reviews pertaining to that week.

    :ivar start_date: The start date of the week.
    :ivar end_date: The end date of the week.
    :ivar public_reviews: A list of public reviews for the week.
    :ivar expert_reviews: A list of expert reviews for the week.
    :ivar box_office: The box office revenue for the week.
    """
    start_date: date
    end_date: date
    public_reviews: list[PublicReview] = field(default_factory=list)
    expert_reviews: list[ExpertReview] = field(default_factory=list)
    box_office: int

    @property
    def public_review_count(self) -> int:
        """
        Number of public reviews for the week.

        :returns: The count of public reviews.
        """
        return len(self.public_reviews)

    @property
    def expert_review_count(self) -> int:
        """
        Number of expert reviews for the week.

        :returns: The count of expert reviews.
        """
        return len(self.expert_reviews)

    @property
    def review_count(self) -> int:
        """
        Total number of public and expert reviews for the week.

        :returns: The combined count of public and expert reviews.
        """
        return self.public_review_count + self.expert_review_count

    @property
    def average_sentiment_score(self) -> Optional[float]:
        """
        Average sentiment score of both public and expert reviews for the week.

        Calculates the mean of `sentiment_score` from all reviews
        where the score is not None.

        :returns: The average sentiment score as a float, or None if no scores are available.
        """
        scores: list[float] = [review.sentiment_score for review in chain(self.public_reviews, self.expert_reviews) if
                               review.sentiment_score is not None]
        return float(np.mean(scores)) if scores else None

    @property
    def average_expert_score(self) -> Optional[float]:
        """
        Average expert score from expert reviews for the week.

        Calculates the mean of `expert_score` from all expert reviews.

        :returns: The average expert score as a float, or None if no expert reviews are available.
        """
        scores: list[float] = [review.expert_score for review in self.expert_reviews]
        return float(np.mean(scores)) if scores else None

    @property
    def total_reply_count(self) -> int:
        """
        Total reply count from all public reviews for the week.

        :returns: The sum of reply counts from all public reviews.
        """
        return sum(review.reply_count for review in self.public_reviews)

    def _update_specific_reviews_list(
        self,
        reviews_source: Path | YamlFile | list[dict | ReviewSubclass],
        review_class: Type[ReviewSubclass],
        target_list_attribute_name: Literal['public_reviews', 'expert_reviews']
    ) -> None:
        """
        Internal helper to update a specific list of reviews (public or expert) for the week.

        Loads reviews from the given source, creates instances of the specified `review_class`,
        filters them to include only those within the week's date range, and updates
        the target attribute (either 'public_reviews' or 'expert_reviews') on the instance.

        :param reviews_source: The source of review data.
        :param review_class: The specific Review subclass (e.g., PublicReview, ExpertReview) to create.
        :param target_list_attribute_name: The name of the attribute on self to update.
        :raises TypeError: If `review_class` is the base `Review` class, or if `reviews_source`
                           is a list containing mixed or invalid types.
        """
        if review_class is Review:
            raise TypeError(
                "The 'review_class' parameter cannot be the base 'Review' class. "
                "Please provide a specific subclass like PublicReview or ExpertReview."
            )
        loaded_data: list[dict]
        processed_reviews_list: list[ReviewSubclass]

        if isinstance(reviews_source, Path):
            loaded_data = YamlFile(path=reviews_source).load()
            processed_reviews_list = review_class.create_reviews(reviews_source=loaded_data)
        elif isinstance(reviews_source, YamlFile):
            loaded_data = reviews_source.load()
            processed_reviews_list = review_class.create_reviews(reviews_source=loaded_data)
        elif isinstance(reviews_source, list):
            if not reviews_source:
                processed_reviews_list = []

            elif all(isinstance(review_item, dict) for review_item in reviews_source):
                processed_reviews_list = review_class.create_reviews(reviews_source=reviews_source)
            elif all(isinstance(review_item, review_class) for review_item in reviews_source):
                processed_reviews_list = reviews_source
            else:
                raise TypeError(
                    f"List for reviews_source must contain only dict or {review_class.__name__} objects."
                )
        else:
            raise TypeError(f"Invalid type {type(reviews_source)} for reviews_source.")

        filtered_list: list[ReviewSubclass] = [review for review in processed_reviews_list if
                                               self.start_date <= review.date <= self.end_date]

        setattr(self, target_list_attribute_name, filtered_list)

    def update_public_reviews(self, reviews_source: Path | YamlFile | list[dict | PublicReview]) -> None:
        """
        Updates the public reviews for this WeekData instance.

        Reviews are loaded from the source, filtered by the week's date range,
        and stored in the `public_reviews` attribute.

        :param reviews_source: The source of public review data.
        """
        self._update_specific_reviews_list(
            reviews_source=reviews_source,
            review_class=PublicReview,
            target_list_attribute_name="public_reviews"
        )

    def update_expert_reviews(self, reviews_source: Path | YamlFile | list[dict | ExpertReview]) -> None:
        """
        Updates the expert reviews for this WeekData instance.

        Reviews are loaded from the source, filtered by the week's date range,
        and stored in the `expert_reviews` attribute.

        :param reviews_source: The source of expert review data.
        """
        self._update_specific_reviews_list(
            reviews_source=reviews_source,
            review_class=ExpertReview,
            target_list_attribute_name="expert_reviews"
        )

    def with_public_reviews(self, reviews_source: Path | YamlFile | list[dict | PublicReview]) -> 'WeekData':
        """
        Updates public reviews and returns the WeekData instance for chaining.

        :param reviews_source: The source of public review data.
        :returns: The WeekData instance itself.
        """
        self.update_public_reviews(reviews_source=reviews_source)
        return self

    def with_expert_reviews(self, reviews_source: Path | YamlFile | list[dict | ExpertReview]) -> 'WeekData':
        """
        Updates expert reviews and returns the WeekData instance for chaining.

        :param reviews_source: The source of expert review data.
        :returns: The WeekData instance itself.
        """
        self.update_expert_reviews(reviews_source=reviews_source)
        return self

    @staticmethod
    def _preprocess_master_review_source(
        master_source: Optional[Path | YamlFile | list[dict | ReviewSubclass]],
        review_class: Type[ReviewSubclass],
        source_name_for_error_msg: str
    ) -> Optional[list[ReviewSubclass]]:
        """
        Helper to load and parse a master review source into a list of review instances.

        Handles different source types (Path, YamlFile, list of dicts, list of instances).
        If the source is already a list of correct review instances, it's returned directly.
        If it's a list of dicts or a file, `review_class.create_reviews` is used.

        :param master_source: The master source of review data. Can be None.
        :param review_class: The specific Review subclass to create.
        :param source_name_for_error_msg: A descriptive name for the source, used in error messages.
        :returns: A list of review instances if `master_source` is provided, otherwise None.
        :raises TypeError: If `master_source` is of an invalid type or a list with mixed/invalid content.
        """
        if master_source is None:
            return None

        if isinstance(master_source, (Path, YamlFile)):
            return review_class.create_reviews(master_source)
        elif isinstance(master_source, list):
            if not master_source:
                return []
            if all(isinstance(item, review_class) for item in master_source):
                return master_source
            elif all(isinstance(item, dict) for item in master_source):
                return review_class.create_reviews(master_source)
            else:
                raise TypeError(
                    f"{source_name_for_error_msg} list must contain only dicts or {review_class.__name__} objects."
                )
        else:
            raise TypeError(f"Invalid type for {source_name_for_error_msg}: {type(master_source)}")

    @classmethod
    def create_weeks_data(cls,
                          weeks_data_source: Path | YamlFile | list[dict],
                          public_reviews_master_source: Optional[Path | YamlFile | list[dict | PublicReview]] = None,
                          expert_reviews_master_source: Optional[Path | YamlFile | list[dict | ExpertReview]] = None
                          ) -> list['WeekData']:
        """
        Creates a list of WeekData objects from various sources.

        Each WeekData object represents a week's box office data and associated reviews.
        Reviews are filtered from the master sources to match each week's date range.

        :param weeks_data_source: The source of week data (box office, start/end dates).
                                  Can be a Path to a YAML file, a YamlFile object, or a list of dictionaries.
        :param public_reviews_master_source: An optional master source for all public reviews.
                                             If provided, reviews will be filtered and assigned to relevant weeks.
        :param expert_reviews_master_source: An optional master source for all expert reviews.
                                             If provided, reviews will be filtered and assigned to relevant weeks.
        :returns: A list of created WeekData objects.
        :raises ValueError: If `weeks_data_source` is of an invalid type.
        """

        def _process_single_week_item(week_data_dict: dict) -> Optional['WeekData']:
            """
            Processes a single raw week_data dictionary to create a WeekData object.

            Parses dates and box office, instantiates WeekData, and then updates it
            with relevant public and expert reviews filtered from the master sources.

            :param week_data_dict: The raw dictionary for a single week's data.
            :returns: A WeekData instance if successful, None otherwise.
            """
            logger: Logger = LoggingManager().get_logger('root')
            if not all(key in week_data_dict for key in required_week_fields):
                logger.warning(
                    f"Skipping week data due to missing required fields: {required_week_fields - week_data_dict.keys()} in {week_data_dict}")
                return None
            try:

                box_office_val: int = int(week_data_dict['box_office'])

                raw_start_date: date | str = week_data_dict.get('start_date')
                start_date_val: date = date.fromisoformat(raw_start_date) if isinstance(raw_start_date,
                                                                                        str) else raw_start_date

                raw_end_date: date | str = week_data_dict.get('end_date')
                end_date_val: date = date.fromisoformat(raw_end_date) if isinstance(raw_end_date, str) else raw_end_date

                if not (isinstance(start_date_val, date) and isinstance(end_date_val, date)):
                    logger.warning(
                        f"Invalid date types for week data. Start: {type(start_date_val)}, End: {type(end_date_val)}. Skipping: {week_data_dict}")
                    return None

            except (
                ValueError, TypeError,
                KeyError) as e:
                logger.warning(
                    f"Error processing basic week data (box_office/dates) '{e}'. Skipping week data: {week_data_dict}")
                return None

            try:
                week_data_obj: WeekData = cls(box_office=box_office_val, start_date=start_date_val,
                                              end_date=end_date_val)
            except Exception as e:
                logger.error(f"Unexpected error instantiating WeekData from {week_data_dict}: {e}")
                return None

            if all_potential_public_reviews is not None:
                try:

                    week_data_obj.update_public_reviews(reviews_source=all_potential_public_reviews)
                except Exception as e:
                    logger.error(
                        f"Error updating public reviews for week {week_data_obj.start_date}-{week_data_obj.end_date}: {e}")

            if all_potential_expert_reviews is not None:
                try:
                    week_data_obj.update_expert_reviews(reviews_source=all_potential_expert_reviews)
                except Exception as e:
                    logger.error(
                        f"Error updating expert reviews for week {week_data_obj.start_date}-{week_data_obj.end_date}: {e}")

            return week_data_obj

        weeks_data_list_from_source: list[dict]
        if isinstance(weeks_data_source, Path):
            weeks_data_list_from_source = YamlFile(path=weeks_data_source).load()
        elif isinstance(weeks_data_source, YamlFile):
            weeks_data_list_from_source = weeks_data_source.load()
        elif isinstance(weeks_data_source, list) and all(
            isinstance(week_data_dict, dict) for week_data_dict in weeks_data_source):
            weeks_data_list_from_source = weeks_data_source
        else:
            raise ValueError(f"Invalid type for weeks_data_source: {type(weeks_data_source)}")

        all_potential_public_reviews: Optional[list[PublicReview]] = cls._preprocess_master_review_source(
            master_source=public_reviews_master_source,
            review_class=PublicReview,
            source_name_for_error_msg="public_reviews_master_source"
        )

        all_potential_expert_reviews: Optional[list[ExpertReview]] = cls._preprocess_master_review_source(
            master_source=expert_reviews_master_source,
            review_class=ExpertReview,
            source_name_for_error_msg="expert_reviews_master_source"
        )

        required_week_fields: set[str] = {f.name for f in fields(WeekData)
                                          if f.init and f.name not in ['public_reviews', 'expert_reviews'] and
                                          not (getattr(f.type, '__origin__', None) is Union
                                               and type(None) in get_args(f.type))}

        return list(filter(None.__ne__, map(_process_single_week_item, weeks_data_list_from_source)))


@dataclass(kw_only=True)
class MovieSessionData:
    """
    Represents session data for a movie over several weeks.

    This includes the movie's ID, name, and a list of WeekData objects
    representing its performance and reviews over consecutive weeks.

    :ivar movie_id: The unique identifier for the movie.
    :ivar movie_name: The name of the movie.
    :ivar weeks_data: A list of WeekData objects for the movie session.
    """
    movie_id: int
    movie_name: str
    weeks_data: list[WeekData]

    def update_weeks_data(self,
                          weeks_data_source: Path | YamlFile | list[dict | WeekData],
                          public_reviews_master_source: Optional[Path | YamlFile | list[dict | PublicReview]] = None,
                          expert_reviews_master_source: Optional[Path | YamlFile | list[dict | ExpertReview]] = None
                          ) -> None:
        """
        Updates the weekly data (WeekData list) for this movie session.

        The method processes various `weeks_data_source` types. If the source contains
        raw dictionaries, it uses `WeekData.create_weeks_data` to build the list,
        optionally incorporating reviews from master sources. If the source is already
        a list of `WeekData` objects, it's used directly.

        :param weeks_data_source: The source of weekly data. Can be a Path, a YamlFile object,
                                  a list of dictionaries, or a list of WeekData objects.
        :param public_reviews_master_source: Optional master source for public reviews,
                                             passed to `WeekData.create_weeks_data` if needed.
        :param expert_reviews_master_source: Optional master source for expert reviews,
                                             passed to `WeekData.create_weeks_data` if needed.
        :raises TypeError: If `weeks_data_source` is of an invalid type or a list with mixed/invalid content.
        """

        processed_weeks_data_list: Optional[list[WeekData]] = None
        data_to_create_from: Optional[list[dict]]

        if isinstance(weeks_data_source, list):
            if not weeks_data_source:
                processed_weeks_data_list = []
                data_to_create_from = None
            elif all(isinstance(item, WeekData) for item in weeks_data_source):
                processed_weeks_data_list = weeks_data_source
                data_to_create_from = None
            elif all(isinstance(item, dict) for item in weeks_data_source):
                data_to_create_from = weeks_data_source
            else:
                raise TypeError("If weeks_data_source is a list, it must contain only dict or WeekData objects.")
        elif isinstance(weeks_data_source, YamlFile):
            data_to_create_from = weeks_data_source.load()
        elif isinstance(weeks_data_source, Path):
            data_to_create_from = YamlFile(path=weeks_data_source).load()
        else:
            raise TypeError(
                f"Invalid type for weeks_data_source: {type(weeks_data_source)}. "
                f"Expected Path, YamlFile, list[dict], or list[WeekData]."
            )

        if processed_weeks_data_list is None:
            processed_weeks_data_list = WeekData.create_weeks_data(
                weeks_data_source=data_to_create_from,
                public_reviews_master_source=public_reviews_master_source,
                expert_reviews_master_source=expert_reviews_master_source
            )

        self.weeks_data = processed_weeks_data_list
        return

    def with_weeks_data(self,
                        weeks_data_source: Path | YamlFile | list[dict | WeekData],
                        public_reviews_master_source: Optional[Path | YamlFile | list[dict | PublicReview]] = None,
                        expert_reviews_master_source: Optional[Path | YamlFile | list[dict | ExpertReview]] = None
                        ) -> 'MovieSessionData':
        """
        Updates weekly data and returns the MovieSessionData instance for chaining.

        :param weeks_data_source: The source of weekly data.
        :param public_reviews_master_source: Optional master source for public reviews.
        :param expert_reviews_master_source: Optional master source for expert reviews.
        :returns: The MovieSessionData instance itself.
        """
        self.update_weeks_data(weeks_data_source=weeks_data_source,
                               public_reviews_master_source=public_reviews_master_source,
                               expert_reviews_master_source=expert_reviews_master_source)
        return self

    @classmethod
    def _create_sessions_for_single_movie(cls,
                                          movie_meta_item: dict,
                                          number_of_weeks: int) -> list['MovieSessionData']:
        """
        Helper method to process a single movie's metadata and create all its weekly sessions.

        Loads box office and review data for a movie, then segments the box office data
        into batches of `number_of_weeks`. For each valid batch, a MovieSessionData
        instance is created with corresponding WeekData.

        :param movie_meta_item: A dictionary containing metadata for the movie,
                                including 'id', 'name', and paths to data files
                                ('box_office_path', 'public_reviews_path', 'expert_reviews_path').
        :param number_of_weeks: The number of weeks each movie session should span.
        :returns: A list of MovieSessionData objects for the given movie,
                  or an empty list if processing fails or no valid sessions are found.
        """

        def _is_valid_week_for_session(week_dict: dict) -> bool:
            """
            Checks if the box office data for the given week dictionary is valid.

            A week is considered valid if its 'box_office' value can be converted
            to an integer and is not zero.

            :param week_dict: A dictionary representing a single week's box office data.
            :returns: True if the week's box office data is valid, False otherwise.
            """
            try:
                if int(week_dict.get('box_office', 0)) == 0:
                    return False
                return True
            except (ValueError, TypeError):
                logger.warning(f"Invalid box office value in batch for movie ID {movie_id}. Week data: {week_dict}")
                return False
        movie_id_val: Optional[int | str] = movie_meta_item.get('id')
        movie_name_val: Optional[str] = movie_meta_item.get('name')
        box_office_file_path: Optional[Path] = movie_meta_item.get('box_office_path')
        public_reviews_file_path: Optional[Path] = movie_meta_item.get('public_reviews_path')
        expert_reviews_file_path: Optional[Path] = movie_meta_item.get('expert_reviews_path')
        logger: Logger = LoggingManager().get_logger('root')
        if movie_id_val is None or movie_name_val is None:
            logger.warning(f"Skipping movie due to missing ID or name in index: {movie_meta_item}")
            return []
        try:
            movie_id: int = int(movie_id_val) if isinstance(movie_id_val, str) else movie_id_val
        except ValueError:
            logger.warning(f"Skipping movie due to invalid ID '{movie_id_val}' in index: {movie_meta_item}")
            return []
        if not box_office_file_path or not box_office_file_path.exists():
            logger.warning(f"Box office file not found for movie ID {movie_id}. Skipping movie '{movie_name_val}'.")
            return []
        loaded_box_office: list[dict] = YamlFile(path=box_office_file_path).load()
        loaded_public_reviews: list[PublicReview] = []
        if public_reviews_file_path and public_reviews_file_path.exists():
            loaded_public_reviews = PublicReview.create_reviews(reviews_source=public_reviews_file_path)
        else:
            logger.info(f"Review file not found for movie ID {movie_id} ('{movie_name_val}')."
                        f"Proceeding without reviews for this movie.")
        loaded_expert_reviews: list[ExpertReview] = []
        if expert_reviews_file_path and expert_reviews_file_path.exists():
            loaded_expert_reviews = ExpertReview.create_reviews(reviews_source=expert_reviews_file_path)
        else:
            logger.info(f"Review file not found for movie ID {movie_id} ('{movie_name_val}')."
                        f"Proceeding without reviews for this movie.")
        multiple_batches: list[list[dict]] = [loaded_box_office[i: i + number_of_weeks]
                                              for i in range(len(loaded_box_office) - number_of_weeks + 1)]
        filtered_batches: list[list[dict]] = [batch for batch in multiple_batches
                                              if all(_is_valid_week_for_session(week) for week in batch)]
        return [cls(movie_id=movie_id,
                    movie_name=movie_name_val,
                    weeks_data=WeekData.create_weeks_data(
                        weeks_data_source=single_batch_week_dicts,
                        public_reviews_master_source=loaded_public_reviews,
                        expert_reviews_master_source=loaded_expert_reviews))
                for single_batch_week_dicts in filtered_batches]

    @classmethod
    def create_movie_sessions_from_dataset(cls, dataset_name: str, number_of_weeks: int) -> list['MovieSessionData']:
        """
        Creates MovieSessionData objects for all movies in a specified dataset.

        It reads an index file from the dataset to get movie metadata, then for each movie,
        it constructs paths to its box office and review files. These are then passed to
        `_create_sessions_for_single_movie` to generate the sessions.

        :param dataset_name: The name of the dataset to process.
        :param number_of_weeks: The number of weeks each movie session should span.
        :returns: A flattened list of all MovieSessionData objects created from the dataset.
        :raises FileNotFoundError: If the dataset's index file is not found.
        :raises ValueError: If the index file is found but contains no movie metadata.
        """

        logger: Logger = LoggingManager().get_logger("root")

        try:
            dataset_path: Path = ProjectConfig().get_processed_box_office_dataset_path(dataset_name=dataset_name)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Failed to get dataset path for '{dataset_name}': {e}")
            return []
        index_file_path: Path = dataset_path / 'index.csv'
        if not index_file_path.exists():
            logger.error(f"Index file not found at '{index_file_path}' for dataset '{dataset_name}'.")
            raise FileNotFoundError
        raw_movies_metadata: list[dict[str, str]] = CsvFile(path=index_file_path).load()
        if not raw_movies_metadata:
            logger.info(f"No movie metadata found in index file: '{index_file_path}'.")
            raise ValueError
        movies_metadata_with_paths: list[dict[str, Optional[str | Path]]] = [
            {**movie_meta_dict,
             'box_office_path': dataset_path / 'box_office' / f"{movie_id_str}.yaml",
             'public_reviews_path': dataset_path / 'public_reviews' / f"{movie_id_str}.yaml",
             'expert_reviews_path': dataset_path / 'expert_reviews' / f"{movie_id_str}.yaml"}
            for movie_meta_dict in raw_movies_metadata if (movie_id_str := movie_meta_dict.get('id'))
        ]
        if not movies_metadata_with_paths:
            logger.info(f"No processable movie metadata after initial validation from '{index_file_path}'.")
            return []
        return chain.from_iterable([cls._create_sessions_for_single_movie(
            movie_meta_item=movie_meta_with_paths_item,
            number_of_weeks=number_of_weeks)
            for movie_meta_with_paths_item in movies_metadata_with_paths
        ])


if __name__ == '__main__':
    pass

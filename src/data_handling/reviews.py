from dataclasses import dataclass
from datetime import date
from logging import Logger
from typing import Optional, Type, TypedDict, TypeVar

from src.core.logging_manager import LoggingManager
from src.data_handling.loader_mixin import MovieAuxiliaryDataMixin

SelfReview = TypeVar('SelfReview', bound='Review')


class ReviewRawData(TypedDict, total=False):
    """
    Represents the raw data structure for a review before processing.

    All fields in this TypedDict are optional due to `total=False`, meaning
    they may not be present in the raw data.

    :ivar url: The URL source from which the review was obtained.
    :ivar title: The title of the review, if one exists.
    :ivar content: The main textual content of the review.
    :ivar date: The original publication date of the review. This can be
                an ISO format string or a date object.
    :ivar sentiment_score: An unprocessed sentiment score associated with the
                           review. Its type can vary (e.g., string, number)
                           or it might be absent.
    """
    url: str
    title: str
    content: str
    date: str | date
    sentiment_score: Optional[str | float | int]


class ReviewPreparedArgs(TypedDict):
    """
    Prepared arguments structure for creating a Review instance.

    :ivar url: The URL of the review.
    :ivar title: The title of the review.
    :ivar content: The content of the review.
    :ivar sentiment_score: The sentiment score of the review. Optional.
    :ivar date: The date of the review.
    """
    url: str
    title: str
    content: str
    date: date
    sentiment_score: Optional[float]


class ReviewSerializableData(TypedDict):
    """
    Serializable data structure for a Review instance.

    :ivar url: The URL of the review.
    :ivar title: The title of the review.
    :ivar content: The content of the review.
    :ivar sentiment_score: The sentiment score of the review. Optional.
    :ivar date: The date of the review.
    """
    url: str
    title: str
    content: str
    date: date
    sentiment_score: Optional[float]


class PublicReviewRawData(ReviewRawData, total=False):
    """
    Raw data structure for a public review, extending ReviewRawData.

    All fields in this TypedDict are optional due to `total=False`.
    Inherits fields from :class:`~ReviewRawData`.

    :ivar reply_count: The reply count.
    """
    reply_count: str | int


class PublicReviewPreparedArgs(ReviewPreparedArgs):
    """
    Represents the prepared arguments structure for creating a PublicReview instance.

    Inherits fields from :class:`~ReviewPreparedArgs`.

    :ivar reply_count: The reply count.
    """
    reply_count: int


class PublicReviewSerializableData(ReviewSerializableData):
    """
    Serializable data structure for a PublicReview instance.

    Inherits fields from :class:`~ReviewSerializableData`.

    :ivar reply_count: The reply count.
    """
    reply_count: int


class ExpertReviewRawData(ReviewRawData, total=False):
    """
    Raw data structure for an expert review, extending ReviewRawData.

    All fields in this TypedDict are optional due to `total=False`.
    Inherits fields from :class:`~ReviewRawData`.

    :ivar expert_score: The expert score.
    """
    expert_score: str | float | int


class ExpertReviewPreparedArgs(ReviewPreparedArgs):
    """
    Represents the prepared arguments structure for creating an ExpertReview instance.

    Inherits fields from :class:`~ReviewPreparedArgs`.

    :ivar expert_score: The expert score.
    """
    expert_score: float


class ExpertReviewSerializableData(ReviewSerializableData):
    """
    Serializable data structure for an ExpertReview instance.

    Inherits fields from :class:`~ReviewSerializableData`.

    :ivar expert_score: The expert score.
    """
    expert_score: float


@dataclass(kw_only=True, frozen=True)
class Review(MovieAuxiliaryDataMixin[SelfReview, ReviewRawData, ReviewPreparedArgs, ReviewSerializableData]):
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

        :return: A unique string key for the review.
        """
        if self.url:
            return self.url
        else:
            return self.content

    def __hash__(self) -> int:
        """
        Returns the hash of the review based on its key.

        :return: The hash value of the review.
        """
        return hash(self._key())

    def __eq__(self, other) -> bool:
        """
        Checks if this review is equal to another object.

        Equality is determined by comparing their unique keys, if the other object
        is also an instance of Review. Not implemented for other types.

        :param other: The object to compare with this review.
        :return: True if the objects are considered equal, False otherwise.
                 NotImplemented if the comparison is not supported for the given type.
        """
        if isinstance(other, Review):
            return self._key() == other._key()
        return NotImplemented

    @property
    def sentiment_score_v1(self) -> bool:
        """
        Provides a boolean sentiment score based on the sentiment_score.

        :return: True if sentiment_score is greater than 0.5, False otherwise.
        :raises ValueError: If sentiment_score is None.
        """
        if self.sentiment_score:
            return True if self.sentiment_score > 0.5 else False
        else:
            raise ValueError(
                "Cannot access v1 boolean sentiment score: sentiment_score has not been analyzed yet and is None.")

    @classmethod
    def _prepare_constructor_args(cls: Type[SelfReview], raw_data: ReviewRawData) -> ReviewPreparedArgs:
        """
        Prepares keyword arguments for the class constructor from a raw data dictionary.

        This method handles parsing for fields common to the Review class,
        such as date conversion and sentiment score handling.
        Subclasses should override this to add their specific fields and call
        `super()._prepare_constructor_args(raw_data)` to get the base arguments.

        :param cls: The class itself.
        :param raw_data: The raw dictionary containing data for a review.
        :return: A dictionary of keyword arguments suitable for instantiating the class.
        :raises ValueError: If required fields are missing, have incorrect types,
                            or if date parsing fails.
        """
        logger: Logger = LoggingManager().get_logger('root')

        processed_text_fields: dict[str, str] = {}
        for field_name in ('url', 'title', 'content'):
            # noinspection PyTypedDict
            raw_value: Optional[str] = raw_data.get(field_name)
            if not isinstance(raw_value, str):
                logger.error(f"Required field '{field_name}' is missing or not a string in data: {raw_data}")
                raise ValueError(f"Field '{field_name}' must be a string and present in data.")
            processed_text_fields[field_name] = raw_value

        raw_date: Optional[str | date] = raw_data.get('date')
        processed_date: date
        if isinstance(raw_date, str):
            try:
                processed_date = date.fromisoformat(raw_date)
            except ValueError:
                logger.error(f"Invalid date format '{raw_date}' in data: {raw_data}.")
                raise ValueError(f"Invalid date format: {raw_date}")
        elif isinstance(raw_date, date):
            processed_date = raw_date
        else:
            logger.error(f"Required field 'date' is missing or has invalid type in data: {raw_data}")
            raise ValueError(f"Missing or invalid type for 'date' in data: {raw_data}")

        raw_sentiment_score: Optional[float | int | str] = raw_data.get('sentiment_score')
        processed_sentiment_score: Optional[float] = None

        if raw_sentiment_score is not None:
            try:
                processed_sentiment_score = float(raw_sentiment_score)
            except (ValueError, TypeError):
                logger.warning(
                    f"Could not convert sentiment_score '{raw_sentiment_score}' to float in data: {raw_data}. Setting to None.")

        return ReviewPreparedArgs(
            url=processed_text_fields['url'],
            title=processed_text_fields['title'],
            content=processed_text_fields['content'],
            date=processed_date,
            sentiment_score=processed_sentiment_score
        )

    def as_serializable_dict(self) -> ReviewSerializableData:
        """
        Converts the Review instance to a serializable dictionary.

        :return: A dictionary containing the serializable data of the review.
        """
        return ReviewSerializableData(
            url=self.url, title=self.title, content=self.content, date=self.date, sentiment_score=self.sentiment_score
        )


@dataclass(kw_only=True,frozen=True)
class PublicReview(Review):
    """
    Represents a public review, extending Review with a reply count.

    Inherits attributes from :class:`~Review`.
    :ivar reply_count: The number of replies to this public review.
    """
    reply_count: int

    @classmethod
    def _prepare_constructor_args(cls: Type[SelfReview], raw_data: PublicReviewRawData) -> PublicReviewPreparedArgs:
        """
        Prepares keyword arguments for the PublicReview constructor.

        Extends the base Review's argument preparation by adding 'reply_count'.

        :param cls: The class itself.
        :param raw_data: The raw dictionary containing data for a public review.
        :return: A dictionary of keyword arguments suitable for instantiating PublicReview.
        :raises ValueError: If 'reply_count' is missing, has an invalid format, or
                            if errors occur during base argument preparation.
        """
        logger: Logger = LoggingManager().get_logger('root')
        # noinspection PyTypeChecker
        base_kwargs: ReviewPreparedArgs = super()._prepare_constructor_args(raw_data)

        raw_reply_count: Optional[int | str] = raw_data.get('reply_count')
        processed_reply_count: int

        if raw_reply_count is not None:
            try:
                processed_reply_count = int(raw_reply_count)
            except (ValueError, TypeError) as e:
                msg = f"Invalid reply_count value '{raw_reply_count}' in PublicReview data: {raw_data}."
                logger.error(msg)
                raise ValueError(f"Invalid reply_count value: {raw_reply_count}") from e
        else:
            msg = f"Required field 'reply_count' missing in PublicReview data: {raw_data}"
            logger.error(msg)
            raise ValueError(msg)

        return PublicReviewPreparedArgs(**{**base_kwargs, 'reply_count': processed_reply_count})

    def as_serializable_dict(self) -> PublicReviewSerializableData:
        """
        Converts the PublicReview instance to a serializable dictionary.

        :return: A dictionary containing the serializable data of the public review.
        """
        return PublicReviewSerializableData(
            url=self.url,
            title=self.title,
            content=self.content,
            date=self.date,
            sentiment_score=self.sentiment_score,
            reply_count=self.reply_count
        )


@dataclass(kw_only=True,frozen=True)
class ExpertReview(Review):
    """
    Represents an expert review, extending Review with an expert score.

    Inherits attributes from :class:`~Review`.
    :ivar expert_score: The score given by the expert.
    """
    expert_score: float

    @classmethod
    def _prepare_constructor_args(cls: Type[SelfReview], raw_data: ExpertReviewRawData) -> ExpertReviewPreparedArgs:
        """
        Prepares keyword arguments for the ExpertReview constructor, including expert_score.

        :param cls: The class itself.
        :param raw_data: The raw dictionary containing data for an expert review.
        :return: A dictionary of keyword arguments suitable for instantiating ExpertReview.
        :raises ValueError: If 'expert_score' is missing, has an invalid format, or
                            if errors occur during base argument preparation.
        """
        logger: Logger = LoggingManager().get_logger('root')

        # noinspection PyTypeChecker
        base_kwargs: ReviewPreparedArgs = super()._prepare_constructor_args(raw_data)

        raw_expert_score: Optional[str | float | int] = raw_data.get('expert_score')
        processed_expert_score: float

        if raw_expert_score is not None:
            try:
                processed_expert_score = float(raw_expert_score)
            except (ValueError, TypeError) as e:
                msg = f"Invalid expert_score value '{raw_expert_score}' in ExpertReview data: {raw_data}."
                logger.error(msg)
                raise ValueError(f"Invalid expert_score value: {raw_expert_score}") from e
        else:
            msg = f"Required field 'expert_score' missing in ExpertReview data: {raw_data}"
            logger.error(msg)
            raise ValueError(msg)

        return ExpertReviewPreparedArgs(**{**base_kwargs, 'expert_score': processed_expert_score})

    def as_serializable_dict(self) -> ExpertReviewSerializableData:
        """
        Converts the ExpertReview instance to a serializable dictionary.

        :return: A dictionary containing the serializable data of the expert review.
        """
        return ExpertReviewSerializableData(
            url=self.url,
            title=self.title,
            content=self.content,
            date=self.date,
            sentiment_score=self.sentiment_score,
            expert_score=self.expert_score
        )

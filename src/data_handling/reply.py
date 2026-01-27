from dataclasses import dataclass
from datetime import datetime
from logging import Logger
from typing import get_args, Literal, Optional, Type, TypeAlias, TypedDict, TypeVar

from src.core.logging_manager import LoggingManager
from src.data_handling.loader_mixin import MovieAuxiliaryDataMixin

SelfReply = TypeVar('SelfReply', bound='Reply')

ReplyRating: TypeAlias = Literal["推", "噓", "→"]


class ReplyRawData(TypedDict, total=False):
    """
    Represents the raw data structure for a reply before processing.

    :ivar rating: The reaction/rating of the reply (e.g., "推", "噓").
    :ivar content: The main textual content of the reply.
    :ivar time: The original publication time of the reply as a string.
    """
    rating: ReplyRating
    content: str
    time: str


class ReplyPreparedArgs(TypedDict):
    """
    Prepared arguments structure for creating a Reply instance.

    :ivar rating: The reaction/rating of the reply.
    :ivar content: The content of the reply.
    :ivar time: The publication time as a datetime object.
    """
    rating: ReplyRating
    content: str
    time: datetime


class ReplySerializableData(TypedDict):
    """
    Serializable data structure for a Reply instance.

    :ivar rating: The reaction/rating of the reply.
    :ivar content: The content of the reply.
    :ivar time: The publication time as an ISO format string.
    """
    rating: ReplyRating
    content: str
    time: str


@dataclass(kw_only=True, frozen=True)
class Reply(
    MovieAuxiliaryDataMixin[SelfReply, ReplyRawData, ReplyPreparedArgs, ReplySerializableData]
):
    """
    Represents a single reply to a review.

    This dataclass is designed to be a self-contained domain object, capable of
    serializing itself to a dictionary and being constructed from a raw dictionary.

    :ivar rating: The reaction/rating of the reply (e.g., "推", "噓", "→").
    :ivar content: The textual content of the reply.
    :ivar time: The publication time of the reply.
    """
    rating: ReplyRating
    content: str
    time: datetime

    def __str__(self) -> str:
        """
        Returns a human-readable string representation of the reply.

        :return: A string formatted as "[rating] time content".
        """
        return f"[{self.rating}] {self.time} {self.content}"

    @classmethod
    def _prepare_constructor_args(cls: Type[SelfReply], raw_data: ReplyRawData) -> ReplyPreparedArgs:
        """
        Prepares keyword arguments for the Reply constructor from raw data.

        This is the **deserialization** logic for converting a dictionary (from YAML)
        into constructor arguments. It handles type conversion and validation.

        :param cls: The class itself.
        :param raw_data: The raw dictionary containing data for a reply.
        :return: A dictionary of keyword arguments suitable for instantiating the class.
        :raises ValueError: If required fields are missing or have incorrect types.
        """
        logger: Logger = LoggingManager().get_logger('root')

        rating: Optional[ReplyRating] = raw_data.get('rating')
        content: Optional[str] = raw_data.get('content')
        raw_time: Optional[str] = raw_data.get('time')

        # Ensure all required fields are present before proceeding.
        if rating is None or content is None or raw_time is None:
            _missing_fields: list[str] = [
                field for field, value in [('rating', rating), ('content', content), ('time', raw_time)] if
                value is None
            ]
            _msg: str = f"Missing required fields in Reply raw data: {', '.join(_missing_fields)}. Data: {raw_data}"
            logger.error(_msg)
            raise ValueError(_msg)

        # Value Validation and Type Conversion
        allowed_ratings: tuple[str, ...] = get_args(ReplyRating)
        if rating not in allowed_ratings:
            _msg: str = f"Invalid rating '{rating}' in Reply data. Must be one of {allowed_ratings}."
            logger.error(_msg)
            raise ValueError(_msg)

        try:
            parsed_time: datetime = datetime.fromisoformat(raw_time)
        except (ValueError, TypeError):
            logger.error(f"Invalid time format '{raw_time}' in Reply data. Must be ISO format.")
            raise ValueError(f"Invalid time format for Reply: {raw_time}")

        return ReplyPreparedArgs(
            rating=rating,
            content=content,
            time=parsed_time
        )

    def as_serializable_dict(self) -> ReplySerializableData:
        """
        Converts the Reply instance to a serializable dictionary.

        This is the **serialization** logic for converting the object into a simple
        dictionary suitable for YAML or JSON storage.

        :return: A dictionary containing the serializable data of the reply.
        """
        return ReplySerializableData(
            rating=self.rating,
            content=self.content,
            time=self.time.isoformat()
        )

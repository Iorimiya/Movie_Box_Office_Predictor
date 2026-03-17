from dataclasses import dataclass
from datetime import datetime
from logging import Logger
from typing import ClassVar, get_args, Literal, Optional, Type, TypeAlias, TypedDict, TypeVar, cast

from src.core.logging_manager import LoggingManager
from src.data_handling.loader_mixin import MovieAuxiliaryDataMixin, RawDataSchema

SelfReply = TypeVar('SelfReply', bound='Reply')

ReplyRating: TypeAlias = Literal["push", "boo", "arrow"]


class ReplyRawData(TypedDict, total=False):
    """
    Represents the raw data structure for a reply before processing.

    :ivar type: The reaction/type of the reply (e.g., "推", "噓").
    :ivar content: The main textual content of the reply.
    :ivar created_at: The original publication time of the reply as a string or datetime object.
    """
    type: ReplyRating | str
    content: str
    created_at: str | datetime


class ReplyPreparedArgs(TypedDict):
    """
    Prepared arguments structure for creating a Reply instance.

    :ivar type: The reaction/type of the reply.
    :ivar content: The content of the reply.
    :ivar created_at: The publication time as a datetime object.
    """
    type: ReplyRating
    content: str
    created_at: datetime


class ReplySerializableData(TypedDict):
    """
    Serializable data structure for a Reply instance.

    :ivar type: The reaction/type of the reply.
    :ivar content: The content of the reply.
    :ivar created_at: The publication time as an ISO format string.
    """
    type: ReplyRating
    content: str
    created_at: str


@dataclass(kw_only=True, frozen=True)
class Reply(
    MovieAuxiliaryDataMixin[SelfReply, ReplyRawData, ReplyPreparedArgs, ReplySerializableData]
):
    """
    Represents a single reply to a review.

    This dataclass is designed to be a self-contained domain object, capable of
    serializing itself to a dictionary and being constructed from a raw dictionary.

    :ivar type: The reaction/type of the reply (e.g., "推", "噓", "→").
    :ivar content: The textual content of the reply.
    :ivar created_at: The publication time of the reply.
    """
    type: ReplyRating
    content: str
    created_at: datetime

    __DISPLAY_MAP: ClassVar[dict[str, str]] = {
        'push': '推',
        'boo': '噓',
        'arrow': '→'
    }

    def __str__(self) -> str:
        """
        Returns a human-readable string representation of the reply.

        :return: A string formatted as "[type] time content".
        """

        display_type: str = self.__DISPLAY_MAP.get(self.type, self.type)
        return f"[{display_type}] {self.created_at} {self.content}"

    @classmethod
    def _prepare_constructor_args(
        cls: Type[SelfReply], raw_data: ReplyRawData, schema_type: RawDataSchema
    ) -> ReplyPreparedArgs:
        """
        Prepares keyword arguments for the Reply constructor from raw data.

        This is the **deserialization** logic for converting a dictionary (from YAML or DB)
        into constructor arguments. It handles type conversion and validation.

        :param cls: The class itself.
        :param raw_data: The raw dictionary containing data for a reply.
        :param schema_type: Specifies the schema of the input dictionaries when 'source' contains raw data.
        :return: A dictionary of keyword arguments suitable for instantiating the class.
        :raises ValueError: If required fields are missing or have incorrect types.
        """
        logger: Logger = LoggingManager().get_logger('root')

        raw_type:Optional[ReplyRating | str] = raw_data.get('type')
        content: Optional[str] = raw_data.get('content')
        raw_time: Optional[str | datetime] = raw_data.get('created_at')

        # Map display string back to internal type if necessary
        reverse_display_map = {v: k for k, v in cls.__DISPLAY_MAP.items()}

        review_type: Optional[ReplyRating]
        if raw_type in reverse_display_map:
            review_type = cast(ReplyRating, reverse_display_map[raw_type])
        else:
            review_type = cast(Optional[ReplyRating], raw_type)

        # Ensure all required fields are present before proceeding.
        if review_type is None or content is None or raw_time is None:
            _missing_fields: list[str] = [
                field for field, value in [('type', review_type), ('content', content), ('time', raw_time)]
                if value is None
            ]
            _msg: str = f"Missing required fields in Reply raw data: {', '.join(_missing_fields)}. Data: {raw_data}"
            logger.error(_msg)
            raise ValueError(_msg)

        # Value Validation and Type Conversion
        allowed_types: tuple[str, ...] = get_args(ReplyRating)
        if review_type not in allowed_types:
            _msg: str = f"Invalid type '{review_type}' in Reply data. Must be one of {allowed_types}."
            logger.error(_msg)
            raise ValueError(_msg)

        parsed_time: datetime
        if isinstance(raw_time, datetime):
            parsed_time = raw_time
        elif isinstance(raw_time, str):
            try:
                parsed_time = datetime.fromisoformat(raw_time)
            except ValueError:
                logger.error(f"Invalid time format '{raw_time}' in Reply data. Must be ISO format.")
                raise ValueError(f"Invalid time format for Reply: {raw_time}")
        else:
            _msg: str = f"Invalid type for 'created_at': {type(raw_time)}. Must be str or datetime."
            logger.error(_msg)
            raise ValueError(_msg)

        return ReplyPreparedArgs(
            type=review_type,
            content=content,
            created_at=parsed_time
        )

    def as_serializable_dict(self) -> ReplySerializableData:
        """
        Converts the Reply instance to a serializable dictionary.

        This is the **serialization** logic for converting the object into a simple
        dictionary suitable for YAML or JSON storage.

        :return: A dictionary containing the serializable data of the reply.
        """
        return ReplySerializableData(
            type=self.type,
            content=self.content,
            created_at=self.created_at.isoformat()
        )

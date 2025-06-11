from dataclasses import dataclass
from datetime import date
from logging import Logger
from typing import Optional, Type, TypedDict

from src.core.logging_manager import LoggingManager
from src.data_handling.loader_mixin import MovieAuxiliaryDataMixin


class BoxOfficeRawData(TypedDict, total=False):
    """
    Represents the raw box office data structure.

    All fields in this TypedDict are optional due to `total=False`, meaning
    they may not be present in the raw data.

    :ivar start_date: The start date of the box office period.
    :ivar end_date: The end date of the box office period.
    :ivar box_office: The box office revenue.
    """
    start_date: str
    end_date: str
    box_office: str | int


class BoxOfficePreparedArgs(TypedDict):
    """
    Represents the prepared arguments for creating a BoxOffice instance.

    :ivar start_date: The start date of the box office period.
    :ivar end_date: The end date of the box office period.
    :ivar box_office: The box office revenue.
    """
    start_date: date
    end_date: date
    box_office: int


class BoxOfficeSerializableData(TypedDict):
    """
    Represents the serializable box office data structure.

    :ivar start_date: The start date of the box office period.
    :ivar end_date: The end date of the box office period.
    :ivar box_office: The box office revenue.
    """
    start_date: date
    end_date: date
    box_office: int


@dataclass(kw_only=True)
class BoxOffice(MovieAuxiliaryDataMixin
                ["BoxOffice", BoxOfficeRawData, BoxOfficePreparedArgs, BoxOfficeSerializableData]):
    """
    Represents box office data for a specific period.

    :ivar start_date: The start date of the box office period.
    :ivar end_date: The end date of the box office period.
    :ivar box_office: The box office revenue for the period.
    """
    start_date: date
    end_date: date
    box_office: int

    @classmethod
    def _prepare_constructor_args(cls: Type["BoxOffice"], raw_data: BoxOfficeRawData) -> BoxOfficePreparedArgs:
        """
        Prepares the raw box office data for the BoxOffice constructor.

        This method validates and converts the raw data fields to the correct types.

        :param cls: The class itself.
        :param raw_data: The raw box office data dictionary.
        :raises ValueError: If required fields are missing or data types are incorrect.
        :return: A dictionary containing the prepared arguments for the constructor.
        """

        logger: Logger = LoggingManager().get_logger('root')

        raw_start_date: Optional[str] = raw_data.get('start_date')
        raw_end_data: Optional[str] = raw_data.get('end_date')
        raw_box_office: Optional[str | int] = raw_data.get('box_office')

        if not isinstance(raw_start_date, str):
            msg: str = f"Required field 'start_date' is missing or not a string in BoxOffice data: {raw_data}"
            logger.error(msg)
            raise ValueError(msg)
        if not isinstance(raw_end_data, str):
            msg: str = f"Required field 'end_date' is missing or not a string in BoxOffice data: {raw_data}"
            logger.error(msg)
            raise ValueError(msg)
        if raw_box_office is None:
            msg: str = f"Required field 'box_office' is missing in BoxOffice data: {raw_data}"
            logger.error(msg)
            raise ValueError(msg)

        try:
            processed_start_date: date = date.fromisoformat(raw_start_date)
            processed_end_date: date = date.fromisoformat(raw_end_data)
            processed_box_office: int = int(raw_box_office)
        except (ValueError, TypeError) as e:
            msg: str = f"Error parsing BoxOffice data fields: {e}. Data: {raw_data}"
            logger.error(msg)
            raise ValueError(msg) from e

        return BoxOfficePreparedArgs(
            start_date=processed_start_date, end_date=processed_end_date, box_office=processed_box_office
        )

    def as_serializable_dict(self) -> BoxOfficeSerializableData:
        """
        Converts the BoxOffice instance to a serializable dictionary.

        This method returns a dictionary representation of the BoxOffice data suitable for serialization.

        :return: A dictionary containing the serializable box office data.
        """
        return BoxOfficeSerializableData(
            start_date=self.start_date,
            end_date=self.end_date,
            box_office=self.box_office
        )

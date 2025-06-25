import json
import re
from dataclasses import dataclass
from datetime import date
from logging import Logger
from pathlib import Path
from typing import Final, Optional, Type, TypedDict

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


@dataclass(kw_only=True, frozen=True)
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

    @classmethod
    def from_json_file(cls: Type["BoxOffice"], file_path: Path, encoding: str = 'utf-8-sig') -> list["BoxOffice"]:
        """
        Loads and parses box office data from a JSON file to create a list of BoxOffice instances.

        The JSON file is expected to have a 'Rows' key, where each item in 'Rows'
        contains 'Date' (e.g., "YYYY-MM-DD~YYYY-MM-DD") and 'Amount' fields.

        :param file_path: The path to the JSON file.
        :param encoding: The encoding of the JSON file.
        :raises FileNotFoundError: If the specified `file_path` does not exist.
        :raises json.JSONDecodeError: If the JSON file is malformed.
        :raises ValueError: If the JSON 'Rows' data is missing, empty, or all items are malformed,
                           or if data conversion within `BoxOffice.create_multiple` fails.
        :return: A list of `BoxOffice` instances.
        """
        logger: Logger = LoggingManager().get_logger('root')
        date_split_pattern: Final[str] = '~'

        if not file_path.exists():
            logger.error(f"JSON file not found: {file_path}")
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        try:
            with open(file=file_path, mode='r', encoding=encoding) as f:
                json_data: dict = json.load(fp=f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from {file_path}: {e}")
            raise

        json_rows: list[dict] = json_data.get('Rows', [])

        if not json_rows:
            msg: str = f"No 'Rows' found in JSON data from {file_path} or 'Rows' is empty."
            logger.error(msg)
            raise ValueError(msg)

        prepared_raw_data_list: list[BoxOfficeRawData] = []
        for week_data_item in json_rows:
            date_str: Optional[str] = week_data_item.get("Date")
            amount_val: Optional[str | int | float] = week_data_item.get("Amount")

            if date_str is None:
                logger.warning(
                    f"Missing 'Date' in week_data_item: {week_data_item} from file {file_path}. Skipping item."
                )
                continue

            try:
                start_date_str, end_date_str = map(str.strip, re.split(date_split_pattern, date_str))
            except ValueError:
                logger.warning(
                    f"Could not split 'Date' string '{date_str}' in {week_data_item} from file {file_path}. Skipping item."
                )
                continue

            box_office_for_raw: str | int
            if amount_val is None:
                # Default to "0" if Amount is missing, _prepare_constructor_args will handle int conversion
                # and raise ValueError if it's truly missing and required.
                # Or, if None implies an error/skip:
                # logger.warning(f"Missing 'Amount' in week_data_item: {week_data_item} from file {file_path}. Skipping item.")
                # continue
                box_office_for_raw = "0"
            elif isinstance(amount_val, float):
                box_office_for_raw = int(amount_val)
            elif isinstance(amount_val, int):
                box_office_for_raw = amount_val
            else:  # Assumed to be string
                box_office_for_raw = str(amount_val)

            prepared_raw_data_list.append(BoxOfficeRawData(
                start_date=start_date_str, end_date=end_date_str, box_office=box_office_for_raw
            ))

        if not prepared_raw_data_list: # This means all items in json_rows were skipped
            msg = f"All items in 'Rows' from {file_path} were malformed or lacked necessary data. No data prepared."
            logger.error(msg)
            raise ValueError(msg)

        try:
            # create_multiple will call _try_create_from_source, which calls _prepare_constructor_args
            # _prepare_constructor_args will raise ValueError for individual item parsing errors.
            # If all items fail _prepare_constructor_args, create_multiple returns an empty list.
            # noinspection PyTypeChecker
            weekly_box_office_data: list["BoxOffice"] = cls.create_multiple(source=prepared_raw_data_list)
        except ValueError as e: # Catch errors from _prepare_constructor_args if they propagate
            logger.error(
                f"Error creating BoxOffice instances from prepared data from {file_path}: {e}",
                exc_info=True
            )
            raise # Re-raise the ValueError from _prepare_constructor_args

        if not weekly_box_office_data: # If create_multiple returned empty list (all items failed validation)
            msg = f"Failed to create any valid BoxOffice objects from the prepared data in {file_path}."
            logger.error(msg)
            raise ValueError(msg)

        return weekly_box_office_data

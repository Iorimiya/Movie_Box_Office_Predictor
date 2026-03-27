from abc import ABC, abstractmethod
from logging import Logger
from typing import Generic, Literal, Optional, Type, TypeAlias, TypeVar

from src.core.logging_manager import LoggingManager

RawDataType = TypeVar("RawDataType")
PreparedArgsType = TypeVar("PreparedArgsType")
SerializableDataType = TypeVar("SerializableDataType")

_Self = TypeVar("_Self", bound="MovieAuxiliaryDataMixin")
RawDataSchema: TypeAlias = Literal['FLAT', 'NESTED']

class MovieAuxiliaryDataMixin(Generic[_Self, RawDataType, PreparedArgsType, SerializableDataType], ABC):
    """
    An abstract mixin class for auxiliary movie data entities.

    This mixin provides a standardized way to create and serialize
    data entities that are auxiliary to the main movie data. It uses generic
    types to allow subclasses to define their specific raw data structures,
    prepared constructor arguments, and serializable data formats.

    Generic Types:
        _Self: The type of the subclass inheriting this mixin.
        RawDataType: The TypedDict structure of the raw data as loaded from a source.
        PreparedArgsType: The TypedDict structure of arguments prepared for the subclass constructor.
        SerializableDataType: The TypedDict structure for the serializable representation of the subclass instance.
    """

    @classmethod
    @abstractmethod
    def _prepare_constructor_args(
        cls: Type[_Self], raw_data: RawDataType, schema_type: RawDataSchema
    ) -> PreparedArgsType:
        """
        Prepares constructor arguments from a raw data item.

        Subclasses must implement this method to transform a raw data dictionary
        into a dictionary of arguments suitable for their constructor. This typically
        involves validation, type conversion, and structuring of the data.

        :param cls: The class itself (subclass of MovieAuxiliaryDataMixin).
        :param raw_data: The raw data dictionary for a single item.
        :param schema_type: Specifies the schema of the input dictionaries when 'source' contains raw data.
        :return: A dictionary of prepared arguments for the class constructor.
        """
        pass

    @classmethod
    def _try_create_from_source(cls: Type[_Self], raw_data: RawDataType, schema_type: RawDataSchema) -> Optional[_Self]:
        """
        Attempts to create an instance of the class from raw data.

        This method calls `_prepare_constructor_args` and then instantiates
        the class. It logs a warning and returns None if instantiation fails
        due to ValueError or TypeError during argument preparation or construction.

        :param cls: The class itself.
        :param raw_data: The raw data dictionary for a single item.
        :param schema_type: Specifies the schema of the input dictionaries.
        :return: An instance of the class if successful, otherwise None.
        """
        logger: Logger = LoggingManager().get_logger('root')
        try:
            constructor_args: PreparedArgsType = cls._prepare_constructor_args(
                raw_data=raw_data, schema_type=schema_type
            )
            return cls(**constructor_args)
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Skipping {cls.__name__} item creation from raw data due to error: {e}. "
                f"Data: {raw_data}"
            )
            return None

    @classmethod
    def create_single(cls: Type[_Self], raw_data: RawDataType, schema_type: RawDataSchema) -> _Self:
        """
        Creates a single instance of the class from raw data.

        This method attempts to create an instance using `_try_create_from_source`.
        If creation fails (returns None), it raises a ValueError.

        :param cls: The class itself.
        :param raw_data: The raw data dictionary for a single item.
        :param schema_type: Specifies the schema of the input dictionaries.
        :raises ValueError: If the instance creation from raw data fails.
        :return: A created instance of the class.
        """
        instance: Optional[_Self] = cls._try_create_from_source(raw_data=raw_data, schema_type=schema_type)
        if instance:
            return instance
        else:
            raise ValueError(f"Failed to create {cls.__name__} from data: {raw_data}. Check logs for details.")

    @classmethod
    def create_multiple(
        cls: Type[_Self], source: list[RawDataType] | list[_Self], schema_type: RawDataSchema
    ) -> list[_Self]:
        """
        Creates a list of objects from a list of raw data dictionaries or existing objects.

        This method acts as a unified factory for converting raw data into domain objects.
        It no longer handles file I/O directly; the caller is responsible for loading
        data from files or databases into a list of dictionaries first.

        :param source: The source data to create objects from. Can be a list of dictionaries
                       or a list of existing objects.
        :param schema_type: Specifies the schema of the input dictionaries when 'source' contains raw data.
                            - 'NESTED': Expects data to follow the nested structure defined in YAML files.
                            - 'FLAT': Expects data to follow the flat structure of database rows.
        :return: A list of instantiated objects.
        :raises TypeError: If the source is not a list or contains invalid types.
        """
        logger: Logger = LoggingManager().get_logger('root')
        raw_data: list[RawDataType]

        if not isinstance(source, list):
            msg: str = (f"Invalid source type for {cls.__name__} creation: {type(source)}. "
                        f"Expected list of raw data or list of instances.")
            logger.error(msg)
            raise TypeError(msg)

        if not source:
            logger.info(f"Received an empty list as source for {cls.__name__} creation, returning empty list.")
            return []

        # Check all items in the list
        all_are_instances: bool = all(isinstance(item, cls) for item in source)
        all_are_dicts: bool = all(isinstance(item, dict) for item in source)

        if all_are_instances:
            logger.info(f"Received a list of {cls.__name__} instances as source, returning directly.")
            # noinspection PyTypeChecker
            return source

        elif all_are_dicts:
            logger.info(
                f"Processing list of {len(source)} dictionaries to create {cls.__name__} instances (schema: {schema_type}).")
            # noinspection PyTypeChecker
            raw_data = source
        else:
            # Mixed types or invalid types found
            first_invalid_item_type: str = "mixed types or unknown"
            for item_in_list in source:
                if not isinstance(item_in_list, dict) and not isinstance(item_in_list, cls):
                    first_invalid_item_type = str(type(item_in_list))
                    break

            # If loop finished but we are here, it means mixed dicts and instances
            if first_invalid_item_type == "mixed types or unknown":
                 first_invalid_item_type = "mixed dicts and instances"

            msg: str = (f"Invalid list content for {cls.__name__} creation. "
                   f"List must contain either all {cls.__name__} instances or all dictionaries. "
                   f"Found items of type like: {first_invalid_item_type}.")
            logger.error(msg)
            raise TypeError(msg)

        created_instances: list[_Self] = [
            obj for item in raw_data
            if (obj := cls._try_create_from_source(raw_data=item, schema_type=schema_type)) is not None
        ]

        num_raw_items: int = len(raw_data)
        num_created: int = len(created_instances)

        if num_raw_items > 0:
            if num_created == 0:
                logger.info(
                    f"Processed {num_raw_items} raw {cls.__name__} items, "
                    f"but none resulted in a valid instance. Check previous warnings for details."
                )
            elif num_created < num_raw_items:
                logger.info(
                    f"Successfully created {num_created} {cls.__name__} instances out of {num_raw_items} "
                    f"raw items. {num_raw_items - num_created} items were skipped."
                )
            else:
                logger.debug(
                    f"Successfully created all {num_created} {cls.__name__} instances "
                    f"from {num_raw_items} raw items."
                )
        return created_instances

    @abstractmethod
    def as_serializable_dict(self) -> SerializableDataType:
        """
        Converts the instance into a serializable dictionary format.

        Subclasses must implement this method to define how their data
        is represented in a format suitable for serialization (e.g., to JSON or YAML).

        :return: A dictionary representing the serializable data of the instance.
        """
        pass

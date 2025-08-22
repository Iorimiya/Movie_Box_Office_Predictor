from abc import ABC, abstractmethod
from logging import Logger
from pathlib import Path
from typing import cast, Generic, Optional, Type, TypedDict, TypeVar

from src.core.logging_manager import LoggingManager
from src.data_handling.file_io import YamlFile

RawDataType = TypeVar("RawDataType", bound=TypedDict)
PreparedArgsType = TypeVar("PreparedArgsType", bound=TypedDict)
SerializableDataType = TypeVar("SerializableDataType", bound=TypedDict)

_Self = TypeVar("_Self", bound="MovieAuxiliaryDataMixin")


def _common_load_yaml_list_of_dicts(file_source: Path | YamlFile, data_entity_name: str, ) -> list[dict[str, any]]:
    """
    Loads and validates data as a list of dictionaries from a YAML file.

    This is a common utility function to ensure that the loaded data from
    a YAML source conforms to the expected structure (a list of dictionaries)
    before further processing.

    :param file_source: The source YAML file, can be a Path object or a YamlFile instance.
    :param data_entity_name: A descriptive name for the data entity being loaded, used for logging and error messages.
    :raises FileNotFoundError: If the source file specified by `file_source` does not exist.
    :raises TypeError: If the loaded data is not a list, or if the list does not exclusively contain dictionaries.
    :return: A list of dictionaries loaded from the YAML file.
    """
    logger: Logger = LoggingManager().get_logger('root')
    loaded_data: list[dict[str, any]]
    source_identifier: str

    if isinstance(file_source, Path):
        source_identifier = str(file_source)
        if not file_source.exists():
            raise FileNotFoundError(f"{data_entity_name} source file not found: {source_identifier}")
        loaded_data = YamlFile(path=file_source).load()
    elif isinstance(file_source, YamlFile):
        source_identifier = str(file_source.path)
        if not file_source.path.exists():
            raise FileNotFoundError(f"{data_entity_name} source file not found: {source_identifier}")
        loaded_data = file_source.load()
    else:
        msg = f"Internal error: Unsupported source type for _common_load_yaml_list_of_dicts: {type(file_source)}"
        logger.error(msg)
        raise TypeError(msg)

    if not isinstance(loaded_data, list):
        msg = (
            f"Data loaded from '{source_identifier}' for {data_entity_name} is not a list. "
            f"Found type: {type(loaded_data)}."
        )
        logger.error(msg)
        raise TypeError(msg)

    if not all(isinstance(item, dict) for item in loaded_data):
        first_non_dict_type = "Unknown (list might be empty or mixed)"
        if loaded_data:
            non_dict_item = next((item for item in loaded_data if not isinstance(item, dict)), None)
            if non_dict_item is not None:
                first_non_dict_type = str(type(non_dict_item))
        msg = (
            f"Data loaded from '{source_identifier}' for {data_entity_name} is not a list of dictionaries. "
            f"List contains items of type: {first_non_dict_type}."
        )
        logger.error(msg)
        raise TypeError(msg)

    return loaded_data


class MovieAuxiliaryDataMixin(Generic[_Self, RawDataType, PreparedArgsType, SerializableDataType], ABC):
    """
    An abstract mixin class for auxiliary movie data entities.

    This mixin provides a standardized way to load, create, and serialize
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
    def _prepare_constructor_args(cls: Type[_Self], raw_data: RawDataType) -> PreparedArgsType:
        """
        Prepares constructor arguments from a raw data item.

        Subclasses must implement this method to transform a raw data dictionary
        into a dictionary of arguments suitable for their constructor. This typically
        involves validation, type conversion, and structuring of the data.

        :param cls: The class itself (subclass of MovieAuxiliaryDataMixin).
        :param raw_data: The raw data dictionary for a single item.
        :return: A dictionary of prepared arguments for the class constructor.
        """
        pass

    @classmethod
    def _try_create_from_source(cls: Type[_Self], raw_data: RawDataType) -> Optional[_Self]:
        """
        Attempts to create an instance of the class from raw data.

        This method calls `_prepare_constructor_args` and then instantiates
        the class. It logs a warning and returns None if instantiation fails
        due to ValueError or TypeError during argument preparation or construction.

        :param cls: The class itself.
        :param raw_data: The raw data dictionary for a single item.
        :return: An instance of the class if successful, otherwise None.
        """
        logger: Logger = LoggingManager().get_logger('root')
        try:
            constructor_args: PreparedArgsType = cls._prepare_constructor_args(raw_data)
            return cls(**constructor_args)
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Skipping {cls.__name__} item creation from raw data due to error: {e}. "
                f"Data: {raw_data}"
            )
            return None

    @classmethod
    def create_single(cls: Type[_Self], raw_data: RawDataType) -> _Self:
        """
        Creates a single instance of the class from raw data.

        This method attempts to create an instance using `_try_create_from_source`.
        If creation fails (returns None), it raises a ValueError.

        :param cls: The class itself.
        :param raw_data: The raw data dictionary for a single item.
        :raises ValueError: If the instance creation from raw data fails.
        :return: A created instance of the class.
        """
        instance: Optional[_Self] = cls._try_create_from_source(raw_data)
        if instance:
            return instance
        else:
            raise ValueError(f"Failed to create {cls.__name__} from data: {raw_data}. Check logs for details.")

    @classmethod
    def create_multiple(cls: Type[_Self], source: Path | YamlFile | list[RawDataType] | list[_Self]) -> list[_Self]:
        """
        Creates multiple instances of the class from a specified source.

        The source can be a file path to a YAML file, a YamlFile object,
        a list of raw data dictionaries, or a list of pre-existing instances
        of the class (in which case the list is returned directly).
        Errors during the creation of individual items are logged, and
        problematic items are skipped.

        :param cls: The class itself.
        :param source: The data source. Can be a Path, YamlFile, list of raw data, or list of instances.
        :raises TypeError: If the `source` is a list containing mixed types or unsupported item types,
                           or if data loaded from a file is not a list of dictionaries.
        :raises ValueError: If an unsupported `source` type is provided.
        :return: A list of created class instances. Returns an empty list if the source file is not found
                 or if type errors occur during file processing.
        """
        logger: Logger = LoggingManager().get_logger('root')
        raw_data: list[RawDataType]

        if isinstance(source, list):
            source_name_for_log = f"input list (length {len(source)})"
            if not source:
                logger.info(f"Received an empty list as source for {cls.__name__} creation, returning empty list.")
                return []
            elif all(isinstance(item, cls) for item in source):
                logger.info(f"Received a list of {cls.__name__} instances as source, returning directly.")
                return source
            elif all(isinstance(item, dict) for item in source):
                logger.info(
                    f"Processing {source_name_for_log} containing dictionaries to create {cls.__name__} instances.")
                raw_data = cast(list[RawDataType], source)
            else:
                first_invalid_item_type = "mixed types or unknown"
                for item_in_list in source:
                    if not isinstance(item_in_list, dict) and not isinstance(item_in_list, cls):
                        first_invalid_item_type = str(type(item_in_list))
                        break

                msg = (f"Invalid list source for {cls.__name__} creation. "
                       f"List must contain either all {cls.__name__} instances or all dictionaries. "
                       f"Found items of type like: {first_invalid_item_type} in {source_name_for_log}.")
                logger.error(msg)
                raise TypeError(msg)
        elif isinstance(source, (Path, YamlFile)):
            source_path_for_log: Path = source if isinstance(source, Path) else source.path
            source_name_for_log: str = f"file \"{source_path_for_log}\""
            try:
                logger.info(f"Loading {source_name_for_log} to create multiple {cls.__name__} objects.")
                loaded_raw_data_list: list[dict[str, any]] = _common_load_yaml_list_of_dicts(
                    file_source=source,
                    data_entity_name=cls.__name__
                )
                raw_data = cast(list[RawDataType], loaded_raw_data_list)
            except FileNotFoundError as e:
                logger.error(f"File not found when loading {cls.__name__} data from {source_name_for_log}: {e}")
                return []
            except TypeError:
                logger.error(
                    f"Type error encountered while processing data from {source_name_for_log} for {cls.__name__}.")
                return []
            except Exception as e:
                logger.error(f"Unexpected error loading raw {cls.__name__} data from {source_name_for_log}: {e}")
                return []
        else:
            msg = f"Invalid source type for {cls.__name__} creation: {type(source)}. Expected Path, YamlFile, list of raw data, or list of instances."
            logger.error(msg)
            raise ValueError(msg)

        created_instances = list(filter(None.__ne__, map(cls._try_create_from_source, raw_data)))

        num_raw_items: int = len(raw_data)
        num_created: int = len(created_instances)

        if num_raw_items > 0:
            if num_created == 0:
                logger.info(
                    f"Processed {num_raw_items} raw {cls.__name__} items from {source_name_for_log}, "
                    f"but none resulted in a valid instance. Check previous warnings for details."
                )
            elif num_created < num_raw_items:
                logger.info(
                    f"Successfully created {num_created} {cls.__name__} instances out of {num_raw_items} "
                    f"raw items from {source_name_for_log}. {num_raw_items - num_created} items were skipped."
                )
            else:
                logger.info(
                    f"Successfully created all {num_created} {cls.__name__} instances "
                    f"from {num_raw_items} raw items from {source_name_for_log}."
                )
        return created_instances

    def as_serializable_dict(self) -> SerializableDataType:
        """
        Converts the instance into a serializable dictionary format.

        Subclasses must implement this method to define how their data
        is represented in a format suitable for serialization (e.g., to JSON or YAML).

        :return: A dictionary representing the serializable data of the instance.
        """
        pass

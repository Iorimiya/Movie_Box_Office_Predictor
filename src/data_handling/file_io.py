import pickle
from abc import ABC, abstractmethod
from csv import DictReader, DictWriter
from pathlib import Path
from typing import Callable, Final, Optional

import yaml

from src.core.constants import Constants


class File(ABC):
    """
    Abstract base class for file operations.

    :ivar path: The path to the file.
    :ivar encoding: The encoding of the file.
    """

    def __init__(self, path: Path, encoding: str = Constants.DEFAULT_ENCODING):
        """Initializes the File object.

        :param path: The path to the file.
        :param encoding: The encoding of the file, defaults to 'utf-8'.
        """
        self.path: Final[Path] = path
        self.encoding: Final[str] = encoding

    @property
    def exists(self) -> bool:
        """
        Checks if the file exists at the specified path.

        :returns: ``True`` if the file exists, ``False`` otherwise.
        """
        return self.path.exists()

    def touch(self, exist_ok: bool = False) -> None:
        """
        Creates the file if it does not exist.

        This is a wrapper around ``pathlib.Path.touch``.

        :param exist_ok: If ``False`` (the default), ``FileExistsError`` is raised if the file
                         already exists. If ``True``, the operation is a no-op if the file exists.
        """
        return self.path.touch(exist_ok=exist_ok)

    def delete(self, missing_ok: bool = False) -> None:
        """
        Deletes the file.

        This is a wrapper around ``pathlib.Path.unlink``.

        :param missing_ok: If ``False`` (the default), ``FileNotFoundError`` is raised if the
                           file does not exist. If ``True``, the exception is suppressed.
        """
        return self.path.unlink(missing_ok=missing_ok)

    @abstractmethod
    def save(self, data: list[dict[any, any]] | dict[any, any]) -> None:
        """
        Saves data to the file.
        This method must be implemented by subclasses.

        :param data: The data to be saved, which can be a list of dictionaries or a single dictionary.
        """
        pass

    @abstractmethod
    def load(self) -> list[dict[any, any]] | dict[any, any]:
        """
        Loads data from the file.
        This method must be implemented by subclasses.

        :returns: The loaded data, which can be a list of dictionaries or a single dictionary.
        """
        pass


class CsvFile(File):
    """
    Handles CSV file operations.

    This class provides methods to save a list of dictionaries to a CSV file
    and load data from a CSV file into a list of dictionaries.
    """

    def __init__(self, path: Path, encoding: str = Constants.DEFAULT_ENCODING,
                 header: Optional[tuple[str, ...]] = None):
        """
        Initializes the CsvFile object.

        :param path: The path to the CSV file.
        :param encoding: The encoding of the file, defaults to 'utf-8'.
        :param header: An optional tuple of strings to be used as the CSV header.
        """
        super().__init__(path=path, encoding=encoding)
        self.header: Final[Optional[tuple[str, ...]]] = header

    def save(self, data: list[dict[any, any]]) -> None:
        """
        Saves a list of dictionaries to the CSV file.

        The header is dynamically generated from the keys of the first dictionary in the list,
        or all dictionaries if their keys vary, to ensure all data is captured.
        If the parent directory of the file does not exist, it will be created.
        If the data list is empty, an empty file is created.

        :param data: A list of dictionaries to write to the CSV file.
        """
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)

        field_names: Optional[list[str]]

        if self.header is not None:
            field_names = list(self.header)
        elif data:  # data is not empty, and no header was provided, so infer from data
            # Ensure all keys are strings for field names
            field_names = list(dict.fromkeys(str(key) for dictionary in data for key in dictionary.keys()))
        else:  # data is empty and no header was provided
            self.touch(exist_ok=True)  # Create an empty file
            return

        # At this point, field_names should be set if we are proceeding to write
        if field_names is None:
            # This case should ideally not be reached if logic above is correct,
            # but as a safeguard for empty data and no header.
            self.touch(exist_ok=True)
            return

        with open(file=self.path, mode='w', encoding=self.encoding, newline='') as file:
            # noinspection PyTypeChecker
            writer: DictWriter = DictWriter(f=file, fieldnames=field_names)
            writer.writeheader()
            if data:
                writer.writerows(rowdicts=data)
        return


    def load(self, row_factory: Optional[Callable[[dict[str, str]], any]] = None) -> list[any]:
        """
        Loads data from the CSV file.

        Each row in the CSV is converted into a dictionary where keys are column headers.
        If a `row_factory` is provided, it is called for each row dictionary to transform it.

        :param row_factory: An optional callable that takes a raw row dictionary (dict[str, str])
                            and returns a transformed row.
        :returns: A list of dictionaries (or transformed objects if `row_factory` is used).
        :raises FileNotFoundError: If the CSV file does not exist at the specified path.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"File {self.path} does not exist")
        with open(file=self.path, mode='r', encoding=self.encoding) as file:
            reader: DictReader = DictReader(f=file)
            if row_factory:
                return [row_factory(row) for row in reader]
            else:
                return list(reader)


class YamlFile(File):
    """
    Handles YAML file operations.

    This class provides methods to save data to a YAML file and load data
    from a YAML file. It supports both multi-document and single-document
    YAML formats.
    """

    def save_single_document(self, data: list[dict[any, any]] | dict[any, any]) -> None:
        """
        Saves data (a list of dictionaries or a single dictionary) as a single YAML document.

        If the parent directory of the file does not exist, it will be created.
        YAML aliases are ignored.

        :param data: The data to be saved as a single YAML document.
        """
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        yaml.Dumper.ignore_aliases = lambda _, __: True
        with open(file=self.path, mode='w', encoding=self.encoding) as file:
            yaml.dump(data=data, stream=file, allow_unicode=True, sort_keys=False)
        return

    def load_single_document(self) -> Optional[list[dict[any, any]] | dict[any, any]]:
        """
        Loads data from a YAML file expected to contain a single document.

        Uses ``yaml.SafeLoader`` for security.

        :returns: The loaded data, which can be a list of dictionaries, a single dictionary,
                  or ``None`` if the file is empty.
        :raises FileNotFoundError: If the YAML file does not exist at the specified path.
        :raises yaml.YAMLError: If there is an error parsing the YAML content.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"File {self.path} does not exist")
        with open(file=self.path, mode='r', encoding=self.encoding) as file:
            try:
                content: str = file.read()
                if not content.strip():
                    return None
                loaded_data: any = yaml.safe_load(stream=content)
                return loaded_data
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Error parsing YAML file {self.path}: {e}")

    def save_multi_document(self, data: list[dict[any, any]]) -> None:
        """
        Saves a list of dictionaries as multiple YAML documents in a single file.

        Each dictionary in the list becomes a separate document, separated by '---'.
        If the parent directory of the file does not exist, it will be created.
        YAML aliases are ignored to ensure each document is self-contained.

        :param data: A list of dictionaries, where each dictionary will be a separate YAML document.
        """
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        yaml.Dumper.ignore_aliases = lambda _, __: True
        with open(file=self.path, mode='w', encoding=self.encoding) as file:
            yaml.dump_all(documents=data, stream=file, allow_unicode=True, sort_keys=False)
        return

    def load_multi_document(self) -> list[dict[any, any]]:
        """
        Loads data from a YAML file that may contain multiple documents
        separated by '---'.

        Uses ``yaml.SafeLoader`` for security. Only documents that are dictionaries
        and not None are returned.

        :returns: A list of dictionaries, each representing a YAML document.
                  Returns an empty list if the file is empty or contains no valid dictionary documents.
        :raises FileNotFoundError: If the YAML file does not exist at the specified path.
        :raises yaml.YAMLError: If there is an error parsing the YAML content.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"File {self.path} does not exist")
        loaded_data: list[dict] = []
        with open(file=self.path, mode='r', encoding=self.encoding) as file:
            for doc in yaml.load_all(stream=file, Loader=yaml.SafeLoader):
                if isinstance(doc, dict) and doc is not None:
                    loaded_data.append(doc)
        return loaded_data

    def save(self, data: list[dict[any, any]] | dict[any, any]) -> None:
        """
        Default save method for ``YamlFile``. Saves data as multiple YAML documents.

        This method calls ``save_multi_document``. If the input `data` is a single
        dictionary, it will be wrapped in a list before being passed to
        ``save_multi_document``, effectively saving it as a single document
        within a multi-document structure.

        :param data: The data to be saved. Can be a single dictionary or a list of dictionaries.
        """
        data_to_save: list[dict[any, any]] = [data] if isinstance(data, dict) else data
        self.save_multi_document(data=data_to_save)
        return

    def load(self) -> Optional[list[dict[any, any]] | dict[any, any]]:
        """
        Default load method for ``YamlFile``. Loads data as multiple YAML documents.

        This method calls ``load_multi_document`` and is suitable for files
        expected to contain one or more YAML documents.

        :returns: A list of dictionaries, where each dictionary represents a YAML document.
                 Returns an empty list if the file is empty or contains no valid dictionary documents.
        """
        return self.load_multi_document()


class PickleFile(File):
    """
    Handles Python object serialization and deserialization using pickle.

    This class provides a simple interface to save any Python object to a
    binary file and load it back.
    """

    def save(self, data: any) -> None:
        """
        Serializes and saves a Python object to the file.

        If the parent directory of the file does not exist, it will be created.

        :param data: The Python object to be saved.
        """
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)

        with open(file=self.path, mode='wb') as handle:
            # noinspection PyTypeChecker
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self) -> any:
        """
        Loads and deserializes a Python object from the file.

        :returns: The deserialized Python object.
        :raises FileNotFoundError: If the pickle file does not exist.
        :raises pickle.UnpicklingError: If the file content is corrupted or not a valid pickle stream.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Pickle file not found: {self.path}")

        with open(file=self.path, mode='rb') as handle:
            return pickle.load(handle)

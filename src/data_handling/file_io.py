from abc import ABC, abstractmethod
from csv import DictReader, DictWriter
from pathlib import Path
from typing import Final, Optional

import yaml


class File(ABC):
    """
    Abstract base class for file operations.

    :ivar path: The path to the file.
    :ivar encoding: The encoding of the file.
    """

    def __init__(self, path: Path, encoding: str = 'utf-8'):
        """Initializes the File object.

        :param path: The path to the file.
        :param encoding: The encoding of the file, defaults to 'utf-8'.
        """
        self.path: Final[Path] = path
        self.encoding: Final[str] = encoding

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

        :return: The loaded data, which can be a list of dictionaries or a single dictionary.
        """
        pass


class CsvFile(File):
    """
    Handles CSV file operations.

    This class provides methods to save a list of dictionaries to a CSV file
    and load data from a CSV file into a list of dictionaries.
    """

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
        if not data:
            self.path.touch()
            return
        field_names: list[str] = list(dict.fromkeys(str(key) for dictionary in data for key in dictionary.keys()))
        with open(file=self.path, mode='w', encoding=self.encoding, newline='') as file:
            # noinspection PyTypeChecker
            writer: DictWriter = DictWriter(file, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(data)
        return

    def load(self) -> list[dict[str, str]]:
        """
        Loads data from the CSV file into a list of dictionaries.

        Each row in the CSV is converted into a dictionary where keys are column headers.

        :return: A list of dictionaries, where each dictionary represents a row.
        :raises FileNotFoundError: If the CSV file does not exist at the specified path.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"File {self.path} does not exist")
        with open(file=self.path, mode='r', encoding=self.encoding) as file:
            return list(DictReader(f=file))


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

        :return: The loaded data, which can be a list of dictionaries, a single dictionary,
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

        :return: A list of dictionaries, each representing a YAML document.
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

        :return: A list of dictionaries, where each dictionary represents a YAML document.
                 Returns an empty list if the file is empty or contains no valid dictionary documents.
        """
        return self.load_multi_document()

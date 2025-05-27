from pathlib import Path
from typing import Final, Any, Optional
import yaml
import csv
from abc import ABC, abstractmethod


class File(ABC):
    """
    Abstract base class for file operations.

    Attributes:
        path (Path): The path to the file.
        encoding (str): The encoding of the file.
    """

    def __init__(self, path: Path, encoding: str = 'utf-8'):
        self.path: Final[Path] = path
        self.encoding: Final[str] = encoding

    @abstractmethod
    def save(self, data: list[dict] | dict) -> None:
        """
        Saves data to the file.
        This method must be implemented by subclasses.

        Args:
            data: The data to be saved.
        """
        pass

    @abstractmethod
    def load(self) -> list[dict] | dict:
        """
        Loads data from the file.
        This method must be implemented by subclasses.

        Returns:
            The loaded data.
        """
        pass


class CsvFile(File):
    """
    Handles CSV file operations.

    This class provides methods to save a list of dictionaries to a CSV file
    and load data from a CSV file into a list of dictionaries.
    """

    def save(self, data: list[dict]) -> None:
        """
        Saves a list of dictionaries to the CSV file.

        The header is dynamically generated from the keys of the dictionaries.
        If the parent directory of the file does not exist, it will be created.

        Args:
            data: A list of dictionaries to write to the CSV file.
        """
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        if not data:  # Handle empty data list
            # Create an empty file with no header or write a specific message
            self.path.touch()
            return
        # Ensure all dictionaries for header generation
        field_names: list[str] = list(dict.fromkeys(key for dictionary in data for key in dictionary.keys()))
        with open(file=self.path, mode='w', encoding=self.encoding, newline='') as file:
            writer: csv.DictWriter = csv.DictWriter(file, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(data)
        return

    def load(self) -> list[dict]:
        """
        Loads data from the CSV file.

        Returns:
            A list of dictionaries, where each dictionary represents a row.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"File {self.path} does not exist")
        with open(file=self.path, mode='r', encoding=self.encoding) as file:
            return list(csv.DictReader(f=file))


class YamlFile(File):
    """
    Handles YAML file operations.

    This class provides methods to save data to a YAML file and load data
    from a YAML file. It supports both multi-document and single-document
    YAML formats.
    """

    def save_multi_document(self, data: list[dict]) -> None:
        """
        Saves a list of dictionaries as multiple YAML documents in a single file,
        separated by '---'.

        If the parent directory of the file does not exist, it will be created.
        YAML aliases are ignored to ensure each document is self-contained.

        Args:
            data: A list of dictionaries, where each dictionary will be a separate YAML document.
        """
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        yaml.Dumper.ignore_aliases = lambda self, data_node: True
        with open(file=self.path, mode='w', encoding=self.encoding) as file:
            yaml.dump_all(documents=data, stream=file, allow_unicode=True, sort_keys=False)
        return

    def load_multi_document(self) -> list[dict]:
        """
        Loads data from a YAML file that may contain multiple documents
        separated by '---'.

        Only dictionary type documents are returned.

        Returns:
            A list of dictionaries, each representing a YAML document.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"File {self.path} does not exist")
        loaded_data: list[dict] = []
        with open(file=self.path, mode='r', encoding=self.encoding) as file:
            for doc in yaml.load_all(stream=file, Loader=yaml.SafeLoader):  # Changed to SafeLoader
                if isinstance(doc, dict) and doc is not None:
                    loaded_data.append(doc)
        return loaded_data

    def save_single_document(self, data: list[dict] | dict) -> None:
        """
        Saves data (a list of dictionaries or a single dictionary) as a single YAML document.

        If the parent directory of the file does not exist, it will be created.
        YAML aliases are ignored.

        Args:
            data: The data to be saved as a single YAML document.
                  Can be a list of dictionaries or a single dictionary.
        """
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        yaml.Dumper.ignore_aliases = lambda self, data_node: True
        with open(file=self.path, mode='w', encoding=self.encoding) as file:
            yaml.dump(data=data, stream=file, allow_unicode=True, sort_keys=False)
        return

    def load_single_document(self) -> Optional[list[dict] | dict]:
        """
        Loads data from a YAML file expected to contain a single document.

        Uses SafeLoader for security.

        Returns:
            The loaded data, which can be a list of dictionaries or a single dictionary,
            or None if the file is empty or contains no valid YAML.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            yaml.YAMLError: If there is an error parsing the YAML content.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"File {self.path} does not exist")
        with open(file=self.path, mode='r', encoding=self.encoding) as file:
            try:
                # Read the whole file content to ensure it's treated as a single document context
                content: str = file.read()
                if not content.strip():  # Handle empty file
                    return None
                loaded_data: Any = yaml.safe_load(stream=content)
                return loaded_data
            except yaml.YAMLError as e:
                # Log or handle specific YAML parsing errors if needed
                # For now, re-raise to indicate a problem with the file format
                raise yaml.YAMLError(f"Error parsing YAML file {self.path}: {e}")

    # Implementing the abstract methods by choosing a default behavior
    # You might want to change these defaults based on your primary use case.
    def save(self, data: list[dict] | dict) -> None:
        """
        Default save method. Saves data as a single YAML document.

        Args:
            data: The data to be saved.
        """
        self.save_single_document(data=data)
        return

    def load(self) -> Optional[list[dict] | dict]:
        """
        Default load method. Loads data as a single YAML document.

        Returns:
            The loaded data.
        """
        return self.load_single_document()


if __name__ == "__main__":
    box_paths = list(Path('../../datasets/box_office_prediction/dataset_2024/box_office').glob('*.yaml'))
    review_paths = list(Path('../../datasets/box_office_prediction/dataset_2024/public_review').glob('*.yaml'))
    box_old_data = [{'path': path, 'data': YamlFile(path=path).load_multi_document(),
                     'new_path': path.parent.with_name('new_box_office').joinpath(path.name)} for path in box_paths]
    review_old_data = [{'path': path, 'data': YamlFile(path=path).load_multi_document(),
                        'new_path': path.parent.with_name('new_public_review').joinpath(path.name)} for path in
                       review_paths]
    box_new_data = [{
        **single_file_data,
        'new_data': [{
            'start_date': data.get('start_date'),
            'end_date': data.get('end_date'),
            'box_office': data.get('box_office')
        } for data in single_file_data.get('data')]}
        for single_file_data in box_old_data]

    review_new_data = [{
        **single_file_data,
        'new_data': [{
            'title': data.get('title'),
            'url': data.get('url'),
            'date': data.get('date'),
            'content': data.get('content'),
            'reply_count': data.get('reply_count'),
            'sentiment_score': 1 if bool(data.get('emotion_analyse')) else 0
        } for data in single_file_data.get('data')]}
        for single_file_data in review_old_data]
    saving_func = lambda file_data: YamlFile(file_data.get('new_path')).save_single_document(
        data=file_data.get('new_data', []))
    for _ in map(saving_func, box_new_data): pass
    for _ in map(saving_func, review_new_data): pass
    a = 1

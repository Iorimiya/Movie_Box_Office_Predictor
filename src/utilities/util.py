import csv
from dataclasses import dataclass
from pathlib import Path

from src.core.constants import Constants


@dataclass
class CSVFileData:
    """Represents the metadata for a CSV file.

    :ivar path: The path to the CSV file.
    :ivar header: The header for the CSV file.
                  If a string, it typically represents a single column name (e.g., for reading a specific value).
                  If a tuple of strings, it represents the complete header row (e.g., for writing or defining multiple specific columns).
    """
    path: Path
    header: tuple[str] | str


def read_data_from_csv(path: Path) -> list:
    """Reads data from a CSV file.

    Each row is read as a dictionary where keys are column headers.

    :param path: The path to the CSV file.
    :raises FileNotFoundError: If the CSV file does not exist at the given path.
    :raises Exception: For other potential I/O errors during file reading.
    :returns: A list of dictionaries, where each dictionary represents a row in the CSV file.
    """
    with open(file=path, mode='r', encoding='utf-8') as file:
        return list(csv.DictReader(file))


def write_data_to_csv(path: Path, data: list[dict], header: tuple[str]) -> None:
    """Writes data to a CSV file.

    The data is a list of dictionaries, where each dictionary's keys should
    correspond to the provided header.

    :param path: The path to the CSV file. The file will be created or overwritten.
    :param data: The data to write, as a list of dictionaries.
    :param header: The header row for the CSV file.
    :raises Exception: For potential I/O errors during file writing.
    :returns: None
    """
    with open(file=path, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(data)
    return


def initialize_index_file(input_file: CSVFileData, index_file: CSVFileData | None = None) -> None:
    """Initializes an index file from an input CSV file.

    The function reads movie names from a specified column in the input CSV
    and creates an index file mapping these names to sequential indices.

    :param input_file: The input CSV file data.
                       Its ``header`` attribute is expected to be a string representing the
                       column name from which to extract movie names.
    :param index_file: The index CSV file data. If ``None``, defaults are used
                       (``Constants.INDEX_PATH``, ``Constants.INDEX_HEADER``).
                       If provided, its ``header`` attribute is expected to be a tuple
                       of two strings. The first string will be the header for the 'index' column,
                       and the second for the 'name' column in the output index file.
    :raises FileNotFoundError: If the ``input_file.path`` does not exist.
    :raises KeyError: If ``input_file.header`` (when a string) is not a valid column in the input CSV.
    :raises IndexError: If ``index_file.header`` (when a tuple) does not contain at least two elements when accessed.
    :raises Exception: For other potential I/O errors.
    :returns: None
    """
    # get movie names from input csv
    if index_file is None:
        index_file = CSVFileData(path=Constants.INDEX_PATH, header=Constants.INDEX_HEADER)
    with open(file=input_file.path, mode='r', encoding='utf-8') as file:
        movie_names: list[str] = [row[input_file.header] for row in csv.DictReader(file)]
    # create index file
    if not index_file.path.exists():
        index_file.path.parent.mkdir(parents=True, exist_ok=True)
    index_file.path.touch()
    write_data_to_csv(path=index_file.path,
                      data=[{index_file.header[0]: index, index_file.header[1]: name} for index, name in
                            enumerate(movie_names)],
                      header=index_file.header)
    return


def recreate_folder(path: Path) -> None:
    """Deletes a folder (and its contents) or a file if it exists, then recreates the folder.

    If the path points to a file, the file is deleted, and a folder with the same
    name is created. If the path points to a directory, it is recursively deleted
    and then recreated as an empty folder.

    :param path: The path to the folder to be recreated.
    :raises Exception: For potential I/O or permission errors during deletion or creation.
    :returns: None
    """
    if path.exists(): rmtree(path=path)
    path.mkdir(parents=True, exist_ok=True)
    return


def rmtree(path: Path) -> None:
    """Recursively removes a file or a directory and its contents.

    If the path points to a file, it is unlinked. If it points to a directory,
    all its contents are removed recursively, and then the directory itself is removed.

    :param path: The path to the file or directory to be removed.
    :raises OSError: For permission errors or other OS-level issues during deletion if the path exists.
                     Note: If the path does not exist, this function does nothing and does not raise an error.
    :returns: None
    """
    if path.is_file():
        path.unlink()
    else:
        for child in path.iterdir():
            rmtree(child)
        path.rmdir()
    return


delete_duplicate = lambda item: list(set(item))
"""Removes duplicate items from an iterable, returning a new list.

The order of items in the returned list is not guaranteed to be the same
as their first appearance in the input iterable due to the use of ``set``.

:param item: The input iterable (e.g., a list) from which to remove duplicates.
:returns: A new list containing unique items from the input iterable.
          Order is not preserved.
"""

check_path = lambda path: isinstance(path, Path) and path.exists()
"""Checks if a given object is a ``pathlib.Path`` instance and if the path exists on the filesystem.

:param path: The object to check.
:returns: ``True`` if ``path`` is a ``pathlib.Path`` object and it exists, ``False`` otherwise.
"""

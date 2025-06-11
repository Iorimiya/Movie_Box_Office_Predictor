import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Iterable, TypeVar

from src.core.constants import Constants
from src.data_handling.file_io import CsvFile

T = TypeVar('T', bound=Hashable)

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


def initialize_index_file(input_file: CSVFileData, index_file: CSVFileData | None = None) -> None:
    """Initializes an index file from an input CSV file.

    The function reads movie names from a specified column in the input CSV
    and creates an index file mapping these names to sequential indices.

    Note:
        - The `input_file.header` attribute **must** be a string for this function.
        - If `index_file` is provided, its `header` attribute **must** be a tuple of two strings.

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
    :raises TypeError: If `input_file.header` is not a string, or if `index_file.header` is not a suitable tuple when expected.
    :raises IndexError: If ``index_file.header`` (when a tuple) does not contain at least two elements when accessed.
    :raises Exception: For other potential I/O errors.
    :returns: None
    """
    # get movie names from input csv
    if index_file is None:
        index_file = CSVFileData(path=Constants.INDEX_PATH, header=Constants.INDEX_HEADER)

    # Runtime check for input_file.header type, as docstring specifies it must be str
    if not isinstance(input_file.header, str):
        raise TypeError(f"input_file.header must be a string, but got {type(input_file.header)}")

    # Runtime check for index_file.header type if it's not the default one from Constants
    # (assuming Constants.INDEX_HEADER is correctly a tuple of two strings)
    if not (isinstance(index_file.header, tuple) and len(index_file.header) >= 2):
         raise TypeError(f"index_file.header must be a tuple of at least two strings, but got {index_file.header}")


    with open(file=input_file.path, mode='r', encoding='utf-8') as file:
        # The previous check ensures input_file.header is a string here
        movie_names: list[str] = [row[input_file.header] for row in csv.DictReader(file)]


    CsvFile(path=index_file.path).save(data=[{index_file.header[0]: index, index_file.header[1]: name}
                                             for index, name in enumerate(movie_names)])
    return


def recreate_folder(path: Path) -> None:
    """Deletes a folder (and its contents) or a file if it exists, then recreates the folder.

    If the path points to a file, the file is deleted, and a folder with the same
    name (at the same location) is created. If the path points to a directory, it is recursively deleted
    and then recreated as an empty folder.

    :param path: The path to the folder to be recreated.
    :raises Exception: For potential I/O or permission errors during deletion or creation.
    :returns: None
    """
    if path.exists():
        rmtree(path=path) # Calls the rmtree function defined below
    path.mkdir(parents=True, exist_ok=True)
    return


def rmtree(path: Path) -> None:
    """Recursively removes a file or a directory and its contents.

    If the path points to a file or a symbolic link, it is unlinked.
    If it points to a directory, all its contents are removed recursively,
    and then the directory itself is removed.
    If the path does not exist, this function does nothing and does not raise an error.

    :param path: The path to the file or directory to be removed.
    :raises OSError: For permission errors or other OS-level issues during deletion if the path exists
                     and is a file/directory that cannot be removed.
    :returns: None
    """
    if not path.exists():
        return

    if path.is_file() or path.is_symlink():
        path.unlink()
    elif path.is_dir():
        for child in path.iterdir():
            rmtree(child)
        path.rmdir()
    # else: path exists but is not a regular file, symlink, or directory.
    # This case is currently not explicitly handled (e.g. it won't be deleted).
    return


def delete_duplicate(items: Iterable[T]) -> list[T]:
    """
    Removes duplicate items from an iterable, preserving the order of first appearance.

    :param items: The input iterable (e.g., a list) from which to remove duplicates.
                  Items must be hashable.
    :type items: Iterable[T]
    :returns: A new list containing unique items from the input iterable,
              with the order of first appearance preserved.
    :rtype: list[T]
    """
    seen: set[T] = set()
    ordered_unique_items: list[T] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered_unique_items.append(item)
    return ordered_unique_items


def check_path_exists(path_obj: Path | None) -> bool:
    """Checks if a given object is a ``pathlib.Path`` instance and if the path exists on the filesystem.

    :param path_obj: The object to check. Can be None.
    :returns: ``True`` if ``path_obj`` is a ``pathlib.Path`` object and it exists, ``False`` otherwise (including if path_obj is None).
    """
    return isinstance(path_obj, Path) and path_obj.exists()

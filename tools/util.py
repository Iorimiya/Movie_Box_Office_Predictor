import csv
from pathlib import Path
from dataclasses import dataclass

from tools.constant import Constants


@dataclass
class CSVFileData:
    path: Path
    header: tuple[str] | str


def read_data_from_csv(path: Path) -> list:
    """
    Reads data from a CSV file.

    Args:
        path (Path): The path to the CSV file.

    Returns:
        list: A list of dictionaries representing the rows in the CSV file.
    """
    with open(file=path, mode='r', encoding='utf-8') as file:
        return list(csv.DictReader(file))


def write_data_to_csv(path: Path, data: list[dict], header: tuple[str]) -> None:
    """
    Writes data to a CSV file.

    Args:
        path (Path): The path to the CSV file.
        data (list[dict]): The data to write.
        header (tuple[str]): The header row for the CSV file.
    """
    with open(file=path, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(data)
    return


def initialize_index_file(input_file: CSVFileData, index_file: CSVFileData | None = None) -> None:
    """
    Initializes an index file from an input CSV file.

    Args:
        input_file (CSVFileData): The input CSV file data.
        index_file (CSVFileData | None): The index CSV file data. Defaults to None.
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


delete_duplicate = lambda item: list(set(item))
"""
Deletes duplicate items from a list.

Args:
    item: The input list.

Returns:
    A new list containing only unique items from the input list.
"""

check_path = lambda path: isinstance(path, Path) and path.exists()
"""
Checks if a path exists and is a Path object.

Args:
    path: The path to check.

Returns:
    True if the path is a Path object and exists, False otherwise.
"""

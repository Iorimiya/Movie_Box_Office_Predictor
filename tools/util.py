import csv
from pathlib import Path
from dataclasses import dataclass

from tools.constant import Constants


@dataclass
class CSVFileData:
    path: Path
    header: tuple[str] | str


def read_data_from_csv(path: Path) -> list:
    with open(file=path, mode='r', encoding='utf-8') as file:
        return list(csv.DictReader(file))


def write_data_to_csv(path: Path, data: list[dict], header: tuple[str]) -> None:
    with open(file=path, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(data)
    return


def initialize_index_file(input_file: CSVFileData, index_file: CSVFileData | None = None) -> None:
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
check_path = lambda path: isinstance(path, Path) and path.exists()

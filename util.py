import csv
from enum import Enum
from pathlib import Path
from typing import TypedDict


class Constant(Enum):
    STATUS_BAR_FORMAT:str = '{desc}: {percentage:3.2f}%|{bar}{r_bar}'

    class Headers(Enum):
        INDEX_HEADER: list[str] = ['id', 'name']
        INPUT_MOVIE_LIST_HEADER: str = 'movie_name'
        BOX_OFFICE_DOWNLOAD_PROGRESS_HEADER: list[str] = ['id', 'movie_page_url', 'file_path']

    class Paths(Enum):
        PROJECT_FOLDER: Path = Path(__file__).parent
        DATA_FOLDER: Path = PROJECT_FOLDER.value.joinpath('data')
        BOX_OFFICE_FOLDER: Path = DATA_FOLDER.value.joinpath('web_scraping_data', 'box_office')
        PUBLIC_REVIEW_FOLDER: Path = DATA_FOLDER.value.joinpath('web_scraping_data', 'public_review')
        EXPERT_REVIEW_FOLDER: Path = DATA_FOLDER.value.joinpath('web_scraping_data', 'expert_review')
        INDEX_PATH: Path = DATA_FOLDER.value.joinpath('web_scraping_data', 'index.csv')



class CSVFileData(TypedDict):
    path: Path
    header: list[str] | str


def read_data_from_csv(path: Path) -> list:
    with open(file=path, mode='r', encoding='utf-8') as file:
        return list(csv.DictReader(file))


def write_data_to_csv(path: Path, data: list[dict], header: list) -> None:
    with open(file=path, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(data)
    return


def initialize_index_file(input_file: CSVFileData, index_file: CSVFileData | None = None) -> None:
    # get movie names from input csv
    if index_file is None:
        index_file = CSVFileData(path=Constant.Paths.INDEX_PATH.value, header=Constant.Headers.INDEX_HEADER.value)
    with open(file=input_file['path'], mode='r', encoding='utf-8') as file:
        movie_names: list[str] = [row[input_file['header']] for row in csv.DictReader(file)]
    # create index file
    if not index_file['path'].exists():
        index_file['path'].parent.mkdir(parents=True, exist_ok=True)
    index_file['path'].touch()
    write_data_to_csv(path=index_file['path'],
                      data=[{index_file['header'][0]: index, index_file['header'][1]: name} for index, name in
                            enumerate(movie_names)],
                      header=index_file['header'])
    return

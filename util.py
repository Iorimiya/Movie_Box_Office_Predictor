import csv
from pathlib import Path
from typing import TypedDict,Final

class Constants:
    STATUS_BAR_FORMAT:Final[str] = '{desc}: {percentage:3.2f}%|{bar}{r_bar}'
    INDEX_HEADER: Final[list[str]] = ['id', 'name']
    INPUT_MOVIE_LIST_HEADER: Final[str] = 'movie_name'
    BOX_OFFICE_DOWNLOAD_PROGRESS_HEADER: Final[list[str]] = ['id', 'movie_page_url', 'file_path']
    PROJECT_FOLDER: Final[Path] = Path(__file__).parent
    DATA_FOLDER: Final[Path] = PROJECT_FOLDER.joinpath('data')
    BOX_OFFICE_FOLDER: Final[Path] = DATA_FOLDER.joinpath('web_scraping_data', 'box_office')
    PUBLIC_REVIEW_FOLDER: Final[Path] = DATA_FOLDER.joinpath('web_scraping_data', 'public_review')
    EXPERT_REVIEW_FOLDER: Final[Path] = DATA_FOLDER.joinpath('web_scraping_data', 'expert_review')
    INDEX_PATH: Final[Path] = DATA_FOLDER.joinpath('index.csv')


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
        index_file = CSVFileData(path=Constants.INDEX_PATH, header=Constants.INDEX_HEADER)
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

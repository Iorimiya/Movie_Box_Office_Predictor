import csv
from pathlib import Path
from dataclasses import dataclass

from tools.constant import Constants
from movie_data import MovieData
from machine_learning_model.emotion_analyser import EmotionAnalyser

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
    with open(file=input_file['path'], mode='r', encoding='utf-8') as file:
        movie_names: list[str] = [row[input_file['header']] for row in csv.DictReader(file)]
    # create index file
    if not index_file.path.exists():
        index_file.path.parent.mkdir(parents=True, exist_ok=True)
    index_file.path.touch()
    write_data_to_csv(path=index_file.path,
                      data=[{index_file.header[0]: index, index_file.header[1]: name} for index, name in
                            enumerate(movie_names)],
                      header=index_file.header)
    return


def read_index_file(file_path: Path = Constants.INDEX_PATH, index_header=None) -> list[MovieData]:
    if index_header is None:
        index_header = Constants.INDEX_HEADER
    return [MovieData(movie_id=int(movie[index_header[0]]), movie_name=movie[index_header[1]]) for movie in
            read_data_from_csv(path=file_path)]


def analyse_review(movie_id: int) -> None:
    movie_data:MovieData = next(filter(lambda movie:movie.movie_id == movie_id,read_index_file()),None)
    movie_data.load_public_review(load_folder_path=Constants.PUBLIC_REVIEW_FOLDER.with_name(f"{Constants.PUBLIC_REVIEW_FOLDER.name}_PTT"))
    analyzer: EmotionAnalyser = EmotionAnalyser(model_path = Constants.DATA_FOLDER.joinpath('emotion_analysis','model','emotion_analysis_model_1000.keras'), tokenizer_path=Constants.DATA_FOLDER.joinpath('emotion_analysis','dataset','tokenizer.pickle'))
    for review in movie_data.public_reviews:
        print(f"{review} | {analyzer.test(review)}")
        

if __name__ == '__main__':
    for i in range(len(read_index_file())):
        analyse_review(i)
from typing import TypedDict
from datetime import date
from pathlib import Path
import yaml


class BoxOffice(TypedDict):
    start_date: date
    end_date: date
    box_office: int


class ExpertReview(TypedDict):
    content: str
    date: date
    score: float


class PublicReview(TypedDict):
    content: str
    date: date
    reply_count: int


class MovieData:
    def __init__(self, movie_id: int, movie_name: str, box_office: list[BoxOffice] | None = None,
                 expert_review: list[ExpertReview] | None = None,
                 public_review: list[PublicReview] | None = None) -> None:
        self.movie_id: int = movie_id
        self.movie_name: str = movie_name
        self.release_date: date | None = None
        self.box_office: list[BoxOffice] | None = None
        self.box_office_week_lens: int = 0
        self.expert_reviews: list[ExpertReview] | None = expert_review
        self.public_reviews: list[PublicReview] | None = public_review

        if box_office:
            self.update_box_office_data(box_office)

    def update_box_office_data(self, box_offices: list[BoxOffice]) -> None:
        self.box_office = box_offices
        self.box_office_week_lens = len(self.box_office)
        return

    def save_box_office(self, save_folder_path: Path, encoding: str = "utf-8"):
        if not save_folder_path.exists():
            save_folder_path.mkdir(parents=True)
        save_file_name: Path = save_folder_path.joinpath(f"{self.movie_id}.yaml")
        yaml.Dumper.ignore_aliases = lambda self, _: True
        with open(save_file_name, mode='w', encoding=encoding) as file:
            yaml.dump_all(self.box_office, file, allow_unicode=True)

    def load_box_office(self, load_file_path: Path, encoding: str = "utf-8"):
        if not load_file_path.exists():
            raise FileNotFoundError(f"File {load_file_path} does not exist")
        with open(load_file_path, mode='r', encoding=encoding) as file:
            self.box_office = yaml.safe_load(file)

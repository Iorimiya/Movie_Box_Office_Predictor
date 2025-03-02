from typing import Optional
from datetime import date
from pathlib import Path
from dataclasses import dataclass
import yaml

from tools.constant import Constants


@dataclass(kw_only=True)
class BoxOffice:
    start_date: date
    end_date: date
    box_office: int


@dataclass(kw_only=True)
class Review:
    url: Optional[str]
    title: Optional[str]
    content: str
    date: date
    emotion_analyse: Optional[bool]

    def __key(self):
        if self.url:
            return self.url
        else:
            return self.content

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, Review):
            return self.__key() == other.__key()
        return NotImplemented


@dataclass(kw_only=True)
class ExpertReview(Review):
    score: float

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        return super().__eq__(other=other)


@dataclass(kw_only=True)
class PublicReview(Review):
    reply_count: int

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        return super().__eq__(other=other)


class MovieData:
    def __init__(self, movie_id: int, movie_name: str, release_date: Optional[date] = None,
                 box_office: Optional[list[BoxOffice]] = None,
                 expert_review: Optional[list[ExpertReview]] = None,
                 public_review: Optional[list[PublicReview]] = None) -> None:
        self.movie_id: int = movie_id
        self.movie_name: str = movie_name
        self.release_date: date | None = release_date
        self.box_office: list[BoxOffice] | None = None

        self.expert_reviews: list[ExpertReview] | None = expert_review
        self.public_reviews: list[PublicReview] | None = public_review

        if box_office:
            self.update_data(box_offices=box_office)

    @property
    def box_office_week_lens(self) -> int:
        return len(self.box_office) if self.box_office else 0

    @property
    def public_reply_count(self) -> int:
        return sum(public_review.reply_count for public_review in self.public_reviews)

    @property
    def public_review_count(self) -> int:
        return len(self.public_reviews) if self.public_reviews else 0

    def update_data(self,
                    release_date: Optional[date] = None,
                    box_offices: Optional[list[BoxOffice]] = None,
                    expert_reviews: Optional[list[ExpertReview]] = None,
                    public_reviews: Optional[list[PublicReview]] = None) -> None:
        if release_date:
            self.release_date = release_date
        if box_offices:
            self.box_office = box_offices
        if expert_reviews:
            self.expert_reviews = expert_reviews
        if public_reviews:
            self.public_reviews = public_reviews
        return

    @staticmethod
    def __save(file_path: Path, data: list[any], encoding: str = Constants.DEFAULT_ENCODING) -> None:
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True)
        yaml.Dumper.ignore_aliases = lambda self, _: True
        with open(file_path, mode='w', encoding=encoding) as file:
            yaml.dump_all(data, file, allow_unicode=True)

    @staticmethod
    def __load(file_path: Path, encoding: str = Constants.DEFAULT_ENCODING) -> list[dict]:
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")
        with open(file_path, mode='r', encoding=encoding) as file:
            data = [data for data in yaml.safe_load_all(file)]
        return data

    def save_box_office(self, save_folder_path: Path, encoding: str = Constants.DEFAULT_ENCODING) -> None:
        file_extension: str = Constants.DEFAULT_SAVE_FILE_EXTENSION
        self.__save(file_path=save_folder_path.joinpath(f"{self.movie_id}.{file_extension}"), data=self.box_office,
                    encoding=encoding)
        return

    def load_box_office(self, load_folder_path: Path, encoding: str = Constants.DEFAULT_ENCODING) -> None:
        file_extension: str = Constants.DEFAULT_SAVE_FILE_EXTENSION
        self.box_office = self.__load(file_path=load_folder_path.joinpath(f"{self.movie_id}.{file_extension}"),
                                      encoding=encoding)
        return

    def save_public_review(self, save_folder_path: Path, encoding: str = Constants.DEFAULT_ENCODING) -> None:
        file_extension: str = Constants.DEFAULT_SAVE_FILE_EXTENSION
        save_path = save_folder_path.joinpath(f"{self.movie_id}.{file_extension}")
        self.__save(file_path=save_path, data=self.public_reviews,
                    encoding=encoding) if self.public_review_count else save_path.touch(exist_ok=True)
        return

    def load_public_review(self, load_folder_path: Path, encoding: str = Constants.DEFAULT_ENCODING) -> None:
        file_extension: str = Constants.DEFAULT_SAVE_FILE_EXTENSION
        self.public_reviews = self.__load(file_path=load_folder_path.joinpath(f"{self.movie_id}.{file_extension}"),
                                          encoding=encoding)
        return

from typing import Optional
from datetime import date, datetime
from pathlib import Path
from dataclasses import dataclass, asdict
import yaml

from tools.constant import Constants
from tools.util import read_data_from_csv, delete_duplicate


@dataclass(kw_only=True)
class BoxOffice:
    start_date: date
    end_date: date
    box_office: int

    @classmethod
    def from_dict(cls, dictionary):
        date_format: str = '%Y-%m-%d'
        return cls(start_date=datetime.strptime(dictionary['start_date'], date_format).date(),
                   end_date=datetime.strptime(dictionary['end_date'], date_format).date(),
                   box_office=int(dictionary['box_office']))


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

    @classmethod
    def from_dict(cls, dictionary):
        date_format: str = '%Y-%m-%d'
        return cls(url=dictionary["url"], title=dictionary["title"], content=dictionary["content"],
                   date=datetime.strptime(dictionary['date'], date_format).date(), score=float(dictionary["score"]))


@dataclass(kw_only=True)
class PublicReview(Review):
    reply_count: int

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        return super().__eq__(other=other)

    @classmethod
    def from_dict(cls, dictionary):
        date_format: str = '%Y-%m-%d'
        return cls(url=dictionary["url"], title=dictionary["title"], content=dictionary["content"],
                   date=datetime.strptime(dictionary['date'], date_format).date(),
                   reply_count=int(dictionary["reply_count"]))


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
            if self.box_office:
                self.box_office = delete_duplicate(self.box_office + box_offices)
            else:
                self.box_office = box_offices
        if expert_reviews:
            if self.expert_reviews:
                self.expert_reviews = delete_duplicate(self.expert_reviews + expert_reviews)
            else:
                self.expert_reviews = expert_reviews
        if public_reviews:
            if self.public_reviews:
                self.public_reviews = delete_duplicate(self.public_reviews + public_reviews)
            else:
                self.public_reviews = public_reviews
        return

    @staticmethod
    def __save(file_path: Path, data: list[any], encoding: str = Constants.DEFAULT_ENCODING) -> None:
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True)
        data = [asdict(x) for x in data]
        yaml.Dumper.ignore_aliases = lambda self, _: True
        with open(file_path, mode='w', encoding=encoding) as file:
            yaml.dump_all(data, file, allow_unicode=True)

    @staticmethod
    def __load(file_path: Path, encoding: str = Constants.DEFAULT_ENCODING) -> list[dict]:
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")
        with open(file_path, mode='r', encoding=encoding) as file:
            load_data = yaml.load_all(file, yaml.loader.BaseLoader)
            data = [data for data in load_data]
        return data

    def save_box_office(self, save_folder_path: Path, encoding: str = Constants.DEFAULT_ENCODING) -> None:
        file_extension: str = Constants.DEFAULT_SAVE_FILE_EXTENSION
        self.__save(file_path=save_folder_path.joinpath(f"{self.movie_id}.{file_extension}"), data=self.box_office,
                    encoding=encoding)
        return

    def load_box_office(self, load_folder_path: Path = Constants.BOX_OFFICE_FOLDER,
                        encoding: str = Constants.DEFAULT_ENCODING) -> None:
        file_extension: str = Constants.DEFAULT_SAVE_FILE_EXTENSION
        self.box_office = [BoxOffice.from_dict(data) for data in
                           self.__load(file_path=load_folder_path.joinpath(f"{self.movie_id}.{file_extension}"),
                                       encoding=encoding)]
        return

    def save_public_review(self, save_folder_path: Path, encoding: str = Constants.DEFAULT_ENCODING) -> None:
        file_extension: str = Constants.DEFAULT_SAVE_FILE_EXTENSION
        save_path = save_folder_path.joinpath(f"{self.movie_id}.{file_extension}")
        self.__save(file_path=save_path, data=self.public_reviews,
                    encoding=encoding) if self.public_review_count else save_path.touch(exist_ok=True)
        return

    def load_public_review(self, load_folder_path: Path = Constants.PUBLIC_REVIEW_FOLDER,
                           encoding: str = Constants.DEFAULT_ENCODING) -> None:
        file_extension: str = Constants.DEFAULT_SAVE_FILE_EXTENSION
        self.public_reviews = [PublicReview.from_dict(data) for data in
                               self.__load(file_path=load_folder_path.joinpath(f"{self.movie_id}.{file_extension}"),
                                           encoding=encoding)]
        return


def load_index_file(file_path: Path = Constants.INDEX_PATH, index_header=None) -> list[MovieData]:
    if index_header is None:
        index_header = Constants.INDEX_HEADER
    return [MovieData(movie_id=int(movie[index_header[0]]), movie_name=movie[index_header[1]]) for movie in
            read_data_from_csv(path=file_path)]

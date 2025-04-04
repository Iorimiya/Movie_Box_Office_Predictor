from typing import Optional
from datetime import date, datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import yaml

from tools.constant import Constants
from tools.util import read_data_from_csv, delete_duplicate


@dataclass(kw_only=True)
class BoxOffice:
    """
    Represents box office data for a specific period.
    """
    start_date: date
    end_date: date
    box_office: int

    @classmethod
    def from_dict(cls, dictionary):
        """
        Creates a BoxOffice object from a dictionary.

        Args:
            dictionary: The dictionary containing box office data.

        Returns:
            BoxOffice: A new BoxOffice object.
        """
        date_format: str = '%Y-%m-%d'
        return cls(start_date=datetime.strptime(dictionary['start_date'], date_format).date(),
                   end_date=datetime.strptime(dictionary['end_date'], date_format).date(),
                   box_office=int(dictionary['box_office']))


@dataclass(kw_only=True)
class Review:
    """
    Represents a review with basic information.
    """
    url: Optional[str]
    title: Optional[str]
    content: str
    date: date
    sentiment_score: Optional[bool]

    def __key(self):
        """
        Returns a unique key for the review.
        """
        if self.url:
            return self.url
        else:
            return self.content

    def __hash__(self):
        """
        Returns the hash of the review.
        """
        return hash(self.__key())

    def __eq__(self, other):
        """
        Checks if two reviews are equal.
        """
        if isinstance(other, Review):
            return self.__key() == other.__key()
        return NotImplemented


@dataclass(kw_only=True)
class ExpertReview(Review):
    """
    Represents a review from an expert.
    """
    score: float

    def __hash__(self):
        """
        Returns the hash of the expert review.
        """
        return super().__hash__()

    def __eq__(self, other):
        """
        Checks if two expert reviews are equal.
        """
        return super().__eq__(other=other)

    @classmethod
    def from_dict(cls, dictionary):
        """
        Creates an ExpertReview object from a dictionary.

        Args:
            dictionary: The dictionary containing expert review data.

        Returns:
            ExpertReview: A new ExpertReview object.
        """
        date_format: str = '%Y-%m-%d'
        return cls(url=dictionary["url"], title=dictionary["title"], content=dictionary["content"],
                   date=datetime.strptime(dictionary['date'], date_format).date(), score=float(dictionary["score"]),
                   sentiment_score=bool(dictionary["emotion_analyse"]))


@dataclass(kw_only=True)
class PublicReview(Review):
    """
    Represents a public review.
    """
    reply_count: int

    def __hash__(self):
        """
        Returns the hash of the public review.
        """
        return super().__hash__()

    def __eq__(self, other):
        """
        Checks if two public reviews are equal.
        """
        return super().__eq__(other=other)

    @classmethod
    def from_dict(cls, dictionary):
        """
        Creates a PublicReview object from a dictionary.

        Args:
            dictionary: The dictionary containing public review data.

        Returns:
            PublicReview: A new PublicReview object.
        """
        date_format: str = '%Y-%m-%d'
        return cls(url=dictionary["url"], title=dictionary["title"], content=dictionary["content"],
                   date=datetime.strptime(dictionary['date'], date_format).date(),
                   reply_count=int(dictionary["reply_count"]), sentiment_score=bool(dictionary["emotion_analyse"]))


class MovieData:
    """
    Represents data for a movie, including box office and reviews.
    """
    def __init__(self, movie_id: int, movie_name: str, release_date: Optional[date] = None,
                 box_office: Optional[list[BoxOffice]] = None,
                 expert_review: Optional[list[ExpertReview]] = None,
                 public_review: Optional[list[PublicReview]] = None) -> None:
        """
        Initializes the MovieData.

        Args:
            movie_id (int): The unique ID of the movie.
            movie_name (str): The name of the movie.
            release_date (Optional[date]): The release date of the movie. Defaults to None.
            box_office (Optional[list[BoxOffice]]): The box office data for the movie. Defaults to None.
            expert_review (Optional[list[ExpertReview]]): The expert reviews for the movie. Defaults to None.
            public_review (Optional[list[PublicReview]]): The public reviews for the movie. Defaults to None.
        """
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
        """
        Returns the number of weeks of box office data available.
        """
        return len(self.box_office) if self.box_office else 0

    @property
    def public_reply_count(self) -> int:
        """
        Returns the total number of replies across all public reviews.
        """
        return sum(public_review.reply_count for public_review in self.public_reviews)

    @property
    def public_review_count(self) -> int:
        """
        Returns the number of public reviews available.
        """
        return len(self.public_reviews) if self.public_reviews else 0

    def update_data(self,
                    release_date: Optional[date] = None,
                    box_offices: Optional[list[BoxOffice]] = None,
                    expert_reviews: Optional[list[ExpertReview]] = None,
                    public_reviews: Optional[list[PublicReview]] = None) -> None:
        """
        Updates the movie data with new information.

        Args:
            release_date (Optional[date]): The release date to update. Defaults to None.
            box_offices (Optional[list[BoxOffice]]): The box office data to update. Defaults to None.
            expert_reviews (Optional[list[ExpertReview]]): The expert reviews to update. Defaults to None.
            public_reviews (Optional[list[PublicReview]]): The public reviews to update. Defaults to None.
        """
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
        """
        Saves data to a YAML file.

        Args:
            file_path (Path): The path to the file.
            data (list[any]): The data to save.
            encoding (str): The encoding to use. Defaults to Constants.DEFAULT_ENCODING.
        """
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True)
        data = [asdict(x) for x in data]
        yaml.Dumper.ignore_aliases = lambda self, _: True
        with open(file_path, mode='w', encoding=encoding) as file:
            yaml.dump_all(data, file, allow_unicode=True)

    @staticmethod
    def __load(file_path: Path, encoding: str = Constants.DEFAULT_ENCODING) -> list[dict]:
        """
        Loads data from a YAML file.

        Args:
            file_path (Path): The path to the file.
            encoding (str): The encoding to use. Defaults to Constants.DEFAULT_ENCODING.

        Returns:
            list[dict]: The loaded data as a list of dictionaries.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")
        with open(file_path, mode='r', encoding=encoding) as file:
            load_data = yaml.load_all(file, yaml.loader.BaseLoader)
            data = [data for data in load_data]
        return data

    def save_box_office(self, save_folder_path: Path, encoding: str = Constants.DEFAULT_ENCODING) -> None:
        """
        Saves the box office data to a YAML file.

        Args:
            save_folder_path (Path): The folder to save the data in.
            encoding (str): The encoding to use. Defaults to Constants.DEFAULT_ENCODING.
        """
        file_extension: str = Constants.DEFAULT_SAVE_FILE_EXTENSION
        self.__save(file_path=save_folder_path.joinpath(f"{self.movie_id}.{file_extension}"), data=self.box_office,
                    encoding=encoding)
        return

    def load_box_office(self, load_folder_path: Path = Constants.BOX_OFFICE_FOLDER,
                        encoding: str = Constants.DEFAULT_ENCODING) -> None:
        """
        Loads the box office data from a YAML file.

        Args:
            load_folder_path (Path): The folder to load the data from. Defaults to Constants.BOX_OFFICE_FOLDER.
            encoding (str): The encoding to use. Defaults to Constants.DEFAULT_ENCODING.
        """
        file_extension: str = Constants.DEFAULT_SAVE_FILE_EXTENSION
        self.box_office = [BoxOffice.from_dict(data) for data in
                           self.__load(file_path=load_folder_path.joinpath(f"{self.movie_id}.{file_extension}"),
                                       encoding=encoding)]
        return

    def save_public_review(self, save_folder_path: Path, encoding: str = Constants.DEFAULT_ENCODING) -> None:
        """
        Saves the public reviews to a YAML file.

        Args:
            save_folder_path (Path): The folder to save the data in.
            encoding (str): The encoding to use. Defaults to Constants.DEFAULT_ENCODING.
        """
        file_extension: str = Constants.DEFAULT_SAVE_FILE_EXTENSION
        save_path = save_folder_path.joinpath(f"{self.movie_id}.{file_extension}")
        self.__save(file_path=save_path, data=self.public_reviews,
                    encoding=encoding) if self.public_review_count else save_path.touch(exist_ok=True)
        return

    def load_public_review(self, load_folder_path: Path = Constants.PUBLIC_REVIEW_FOLDER,
                           encoding: str = Constants.DEFAULT_ENCODING) -> None:
        """
        Loads the public reviews from a YAML file.

        Args:
            load_folder_path (Path): The folder to load the data from. Defaults to Constants.PUBLIC_REVIEW_FOLDER.
            encoding (str): The encoding to use. Defaults to Constants.DEFAULT_ENCODING.
        """
        file_extension: str = Constants.DEFAULT_SAVE_FILE_EXTENSION
        self.public_reviews = [PublicReview.from_dict(data) for data in
                               self.__load(file_path=load_folder_path.joinpath(f"{self.movie_id}.{file_extension}"),
                                           encoding=encoding)]
        return


class IndexLoadMode(Enum):
    """
    Enum representing the mode for loading the index file.
    """
    ID_NAME = 0
    FULL = 1


def load_index_file(file_path: Path = Constants.INDEX_PATH, index_header=None,
                    mode: IndexLoadMode = IndexLoadMode.ID_NAME) -> list[MovieData]:
    """
    Loads movie data from the index file.

    Args:
        file_path (Path): The path to the index file. Defaults to Constants.INDEX_PATH.
        index_header: The header for the index file. Defaults to None.
        mode (IndexLoadMode): The mode for loading the index file. Defaults to IndexLoadMode.ID_NAME.

    Returns:
        list[MovieData]: A list of MovieData objects.

    Raises:
        ValueError: If an unknown index load mode is specified.
    """
    if index_header is None:
        index_header = Constants.INDEX_HEADER
    movie_list: list[MovieData] = [MovieData(movie_id=int(movie[index_header[0]]), movie_name=movie[index_header[1]])
                                   for movie in read_data_from_csv(path=file_path)]
    match mode:
        case IndexLoadMode.ID_NAME:
            return movie_list
        case IndexLoadMode.FULL:
            for movie in movie_list:
                movie.load_box_office()
                movie.load_public_review()
            return movie_list
        case _:
            raise ValueError(f"Unknown index load mode {mode}")

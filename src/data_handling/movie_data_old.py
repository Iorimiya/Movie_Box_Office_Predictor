# DEPRECATED: This module is deprecated
# TODO: Deprecate this file
from dataclasses import asdict, dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml

from src.core.constants import Constants
from src.data_handling.file_io import CsvFile
from src.utilities.util import delete_duplicate


@dataclass(kw_only=True)
class BoxOffice:
    """Represents box office data for a specific period.

    :ivar start_date: The start date of the box office period.
    :ivar end_date: The end date of the box office period.
    :ivar box_office: The box office revenue for the period.
    """
    start_date: date
    end_date: date
    box_office: int

    @classmethod
    def from_dict(cls, dictionary) -> 'BoxOffice':
        """Creates a BoxOffice object from a dictionary.

        Assumes date strings in the dictionary are in '%Y-%m-%d' format.

        :param dictionary: The dictionary containing box office data with keys
                           'start_date', 'end_date', and 'box_office'.
        :returns: A new ``BoxOffice`` object.
        :raises KeyError: If required keys are missing from the dictionary.
        :raises ValueError: If date strings are not in the expected format or
                            if 'box_office' cannot be converted to an integer.
        """
        date_format: str = '%Y-%m-%d'
        return cls(start_date=datetime.strptime(dictionary['start_date'], date_format).date(),
                   end_date=datetime.strptime(dictionary['end_date'], date_format).date(),
                   box_office=int(dictionary['box_office']))


@dataclass(kw_only=True)
class Review:
    """Represents a review with basic information.

    This class is designed to be hashable and comparable based on its URL
    (if available) or content.

    :ivar url: The URL of the review, if available.
    :ivar title: The title of the review, if available.
    :ivar content: The main content of the review.
    :ivar date: The date the review was published or recorded.
    :ivar sentiment_score: The sentiment score of the review (e.g., True for positive).
                           This is optional.
    """
    url: Optional[str]
    title: Optional[str]
    content: str
    date: date
    sentiment_score: Optional[bool]

    def __key(self) -> str:
        """Returns a unique key for the review, prioritizing URL over content.

        :returns: The URL if it exists, otherwise the content of the review.
        """
        if self.url:
            return self.url
        else:
            return self.content

    def __hash__(self) -> int:
        """Returns the hash of the review based on its unique key.

        :returns: The hash value of the review's key.
        """
        return hash(self.__key())

    def __eq__(self, other) -> bool:
        """Checks if two reviews are equal based on their unique keys.

        :param other: The object to compare with.
        :returns: ``True`` if ``other`` is a ``Review`` instance and their keys are equal,
                  ``False`` otherwise. Returns ``NotImplemented`` if ``other`` is not a ``Review``.
        """
        if isinstance(other, Review):
            return self.__key() == other.__key()
        return NotImplemented


@dataclass(kw_only=True)
class ExpertReview(Review):
    """Represents a review from an expert, inheriting from ``Review`` and adding a numerical score.

    :ivar score: The numerical score given by the expert.
    """
    score: float

    def __hash__(self) -> int:
        """Returns the hash of the expert review, using the base ``Review`` class's hash.

        :returns: The hash value of the review.
        """
        return super().__hash__()

    def __eq__(self, other) -> bool:
        """Checks if two expert reviews are equal, using the base ``Review`` class's equality check.

        :param other: The object to compare with.
        :returns: Result of the base class equality check.
        """
        return super().__eq__(other=other)

    @classmethod
    def from_dict(cls, dictionary) -> "ExpertReview":
        """Creates an ``ExpertReview`` object from a dictionary.

        Assumes date strings in the dictionary are in '%Y-%m-%d' format.
        The 'emotion_analyse' key is mapped to ``sentiment_score``.

        :param dictionary: The dictionary containing expert review data with keys
                           'url', 'title', 'content', 'date', 'score', and 'emotion_analyse'.
        :returns: A new ``ExpertReview`` object.
        :raises KeyError: If required keys are missing from the dictionary.
        :raises ValueError: If date strings are not in the expected format or
                            if 'score' cannot be converted to a float.
        """
        date_format: str = '%Y-%m-%d'
        return cls(url=dictionary["url"], title=dictionary["title"], content=dictionary["content"],
                   date=datetime.strptime(dictionary['date'], date_format).date(), score=float(dictionary["score"]),
                   sentiment_score=bool(dictionary["emotion_analyse"]))


@dataclass(kw_only=True)
class PublicReview(Review):
    """Represents a public review, inheriting from ``Review`` and adding a reply count.

    :ivar reply_count: The number of replies to the public review.
    """
    reply_count: int

    def __hash__(self) -> int:
        """Returns the hash of the public review, using the base ``Review`` class's hash.

        :returns: The hash value of the review.
        """
        return super().__hash__()

    def __eq__(self, other) -> bool:
        """Checks if two public reviews are equal, using the base ``Review`` class's equality check.

        :param other: The object to compare with.
        :returns: Result of the base class equality check.
        """
        return super().__eq__(other=other)

    @classmethod
    def from_dict(cls, dictionary) -> 'PublicReview':
        """Creates a ``PublicReview`` object from a dictionary.

        Assumes date strings in the dictionary are in '%Y-%m-%d' format.
        The 'emotion_analyse' key is mapped to ``sentiment_score``.

        :param dictionary: The dictionary containing public review data with keys
                           'url', 'title', 'content', 'date', 'reply_count', and 'emotion_analyse'.
        :returns: A new ``PublicReview`` object.
        :raises KeyError: If required keys are missing from the dictionary.
        :raises ValueError: If date strings are not in the expected format or
                            if 'reply_count' cannot be converted to an integer.
        """
        date_format: str = '%Y-%m-%d'
        return cls(url=dictionary["url"], title=dictionary["title"], content=dictionary["content"],
                   date=datetime.strptime(dictionary['date'], date_format).date(),
                   reply_count=int(dictionary["reply_count"]), sentiment_score=bool(dictionary["emotion_analyse"]))


class MovieData:
    """Represents comprehensive data for a single movie.

    This includes its ID, name, release date, box office records,
    expert reviews, and public reviews. It provides methods to update,
    save, and load this data.

    :ivar movie_id: The unique identifier for the movie.
    :ivar movie_name: The name of the movie.
    :ivar release_date: The release date of the movie. Can be ``None``.
    :ivar box_office: A list of ``BoxOffice`` objects for the movie. Can be ``None``.
    :ivar expert_reviews: A list of ``ExpertReview`` objects for the movie. Can be ``None``.
    :ivar public_reviews: A list of ``PublicReview`` objects for the movie. Can be ``None``.
    """

    def __init__(self, movie_id: int, movie_name: str, release_date: Optional[date] = None,
                 box_office: Optional[list[BoxOffice]] = None,
                 expert_review: Optional[list[ExpertReview]] = None,
                 public_review: Optional[list[PublicReview]] = None) -> None:
        """Initializes the ``MovieData`` object.

        :param movie_id: The unique ID of the movie.
        :param movie_name: The name of the movie.
        :param release_date: The release date of the movie. Defaults to ``None``.
        :param box_office: A list of ``BoxOffice`` records. Defaults to ``None``.
                           If provided, ``update_data`` is called to process it.
        :param expert_review: A list of ``ExpertReview`` objects. Defaults to ``None``.
        :param public_review: A list of ``PublicReview`` objects. Defaults to ``None``.
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
        """Returns the number of weeks for which box office data is available.

        :returns: The count of ``BoxOffice`` records, or 0 if none exist.
        """
        return len(self.box_office) if self.box_office else 0

    @property
    def public_reply_count(self) -> int:
        """Returns the total number of replies across all available public reviews.

        :returns: The sum of reply counts from all ``PublicReview`` objects,
                  or 0 if no public reviews exist.
        """
        return sum(public_review.reply_count for public_review in self.public_reviews)

    @property
    def public_review_count(self) -> int:
        """Returns the number of public reviews available for the movie.

        :returns: The count of ``PublicReview`` objects, or 0 if none exist.
        """
        return len(self.public_reviews) if self.public_reviews else 0

    def update_data(self,
                    release_date: Optional[date] = None,
                    box_offices: Optional[list[BoxOffice]] = None,
                    expert_reviews: Optional[list[ExpertReview]] = None,
                    public_reviews: Optional[list[PublicReview]] = None) -> None:
        """Updates the movie's data with new information.

        If new data is provided for ``box_offices``, ``expert_reviews``, or ``public_reviews``,
        it is appended to the existing list (if any) and duplicates are removed.

        :param release_date: The new release date to set. If ``None``, the existing date is unchanged.
        :param box_offices: A list of new ``BoxOffice`` records to add.
        :param expert_reviews: A list of new ``ExpertReview`` objects to add.
        :param public_reviews: A list of new ``PublicReview`` objects to add.
        :returns: None
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
    def __save(file_path: Path, data: list, encoding: str = Constants.DEFAULT_ENCODING) -> None:
        """Saves a list of dataclass objects to a YAML file.

        Each object in the ``data`` list is converted to a dictionary before saving.
        The parent directory of ``file_path`` is created if it doesn't exist.
        YAML aliases are ignored during dumping.

        :param file_path: The path to the YAML file where data will be saved.
        :param data: The list of dataclass objects to save.
        :param encoding: The file encoding to use.
        :raises Exception: For potential I/O errors during folder creation or file writing.
        """
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True)
        data = [asdict(x) for x in data]
        yaml.Dumper.ignore_aliases = lambda self, _: True
        with open(file_path, mode='w', encoding=encoding) as file:
            yaml.dump_all(data, file, allow_unicode=True)

    @staticmethod
    def __load(file_path: Path, encoding: str = Constants.DEFAULT_ENCODING) -> list[dict]:
        """Loads data from a YAML file, expecting a sequence of documents.

        Each document in the YAML file is loaded as a dictionary.

        :param file_path: The path to the YAML file to load.
        :param encoding: The file encoding to use.
        :returns: A list of dictionaries, where each dictionary represents a document
                  from the YAML file.
        :raises FileNotFoundError: If ``file_path`` does not exist.
        :raises yaml.YAMLError: If there is an error parsing the YAML file.
        :raises Exception: For other potential I/O errors.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")
        with open(file_path, mode='r', encoding=encoding) as file:
            load_data = yaml.load_all(file, yaml.loader.BaseLoader)
            data = [data for data in load_data]
        return data

    def save_box_office(self, save_folder_path: Path, encoding: str = Constants.DEFAULT_ENCODING) -> None:
        """Saves the movie's box office data to a YAML file.

        The filename is constructed using the ``movie_id`` and ``Constants.DEFAULT_SAVE_FILE_EXTENSION``.
        The data is saved only if ``self.box_office`` is not empty.

        :param save_folder_path: The directory where the box office file will be saved.
        :param encoding: The file encoding to use.
        :returns: None
        """
        file_extension: str = Constants.DEFAULT_SAVE_FILE_EXTENSION
        self.__save(file_path=save_folder_path.joinpath(f"{self.movie_id}.{file_extension}"), data=self.box_office,
                    encoding=encoding)
        return

    def load_box_office(self, load_folder_path: Path = Constants.BOX_OFFICE_FOLDER,
                        encoding: str = Constants.DEFAULT_ENCODING) -> None:
        """Loads box office data for the movie from a YAML file.

        The filename is constructed using the ``movie_id``. Loaded data is parsed
        into ``BoxOffice`` objects and updates ``self.box_office``.

        :param load_folder_path: The directory from which to load the box office file.
        :param encoding: The file encoding to use.
        :raises FileNotFoundError: If the expected box office file does not exist.
        :raises Exception: For errors during YAML parsing or ``BoxOffice`` object creation.
        :returns: None
        """
        file_extension: str = Constants.DEFAULT_SAVE_FILE_EXTENSION
        self.box_office = [BoxOffice.from_dict(data) for data in
                           self.__load(file_path=load_folder_path.joinpath(f"{self.movie_id}.{file_extension}"),
                                       encoding=encoding)]
        return

    def save_public_review(self, save_folder_path: Path, encoding: str = Constants.DEFAULT_ENCODING) -> None:
        """Saves the movie's public reviews to a YAML file.

        The filename is constructed using the ``movie_id``. If there are public reviews,
        they are saved. If not, an empty file is touched to indicate that the
        saving process was attempted.

        :param save_folder_path: The directory where the public review file will be saved.
        :param encoding: The file encoding to use.
        :returns: None
        """
        file_extension: str = Constants.DEFAULT_SAVE_FILE_EXTENSION
        save_path = save_folder_path.joinpath(f"{self.movie_id}.{file_extension}")
        self.__save(file_path=save_path, data=self.public_reviews,
                    encoding=encoding) if self.public_review_count else save_path.touch(exist_ok=True)
        return

    def load_public_review(self, load_folder_path: Path = Constants.PUBLIC_REVIEW_FOLDER,
                           encoding: str = Constants.DEFAULT_ENCODING) -> None:
        """Loads public reviews for the movie from a YAML file.

        The filename is constructed using the ``movie_id``. Loaded data is parsed
        into ``PublicReview`` objects and updates ``self.public_reviews``.

        :param load_folder_path: The directory from which to load the public review file.
        :param encoding: The file encoding to use.
        :raises FileNotFoundError: If the expected public review file does not exist.
        :raises Exception: For errors during YAML parsing or ``PublicReview`` object creation.
        :returns: None
        """
        file_extension: str = Constants.DEFAULT_SAVE_FILE_EXTENSION
        self.public_reviews = [PublicReview.from_dict(data) for data in
                               self.__load(file_path=load_folder_path.joinpath(f"{self.movie_id}.{file_extension}"),
                                           encoding=encoding)]
        return


class IndexLoadMode(Enum):
    """Enum representing the mode for loading movie data from an index file.

    :cvar ID_NAME: Load only movie ID and name.
    :cvar FULL: Load movie ID, name, and also load associated box office and public review data.
    """
    ID_NAME = 0
    FULL = 1


def load_index_file(file_path: Path = Constants.INDEX_PATH, index_header=None,
                    mode: IndexLoadMode = IndexLoadMode.ID_NAME) -> list[MovieData]:
    """Loads movie data from an index CSV file.

    The index file is expected to map movie IDs to movie names.
    Depending on the ``mode``, it can also trigger loading of detailed
    box office and public review data for each movie.

    :param file_path: The path to the index CSV file.
    :param index_header: A list or tuple specifying the column names for 'movie_id'
                         and 'movie_name' in the index file. If ``None``,
                         ``Constants.INDEX_HEADER`` is used.
    :param mode: The mode for loading data (``ID_NAME`` or ``FULL``).
    :returns: A list of ``MovieData`` objects.
    :raises FileNotFoundError: If ``file_path`` does not exist.
    :raises ValueError: If an unknown ``IndexLoadMode`` is specified.
    :raises KeyError: If ``index_header`` names are not found in the CSV.
    :raises Exception: For other potential errors during CSV reading or data loading.
    """
    if index_header is None:
        index_header = Constants.INDEX_HEADER
    movie_list: list[MovieData] = [MovieData(movie_id=int(movie[index_header[0]]), movie_name=movie[index_header[1]])
                                   for movie in CsvFile(path=file_path).load()]
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

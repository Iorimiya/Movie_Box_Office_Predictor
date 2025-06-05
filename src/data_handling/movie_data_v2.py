from dataclasses import dataclass, field, fields, MISSING
from datetime import date
from itertools import chain
from logging import Logger
from pathlib import Path
from typing import get_args, Literal, Optional, Type, TypeVar, Union

import numpy as np

from src.core.logging_manager import LoggingManager
from src.core.project_config import ProjectConfig
from src.data_handling.file_io import CsvFile, YamlFile



@dataclass(kw_only=True)
class Review:
    url: str
    title: str
    content: str
    date: date
    sentiment_score: Optional[int] = None
    reply_count: int

    @classmethod
    def create_reviews(cls, reviews_source: Path | YamlFile | list[dict]) -> list['Review']:
        if isinstance(reviews_source, Path):
            return cls.create_reviews(YamlFile(path=reviews_source))
        elif isinstance(reviews_source, YamlFile):
            logging.info(f"loading file \"{reviews_source.path}\" to create multiple Review.")
            reviews: list[dict] = reviews_source.load()
            return cls.create_reviews(reviews)
        elif isinstance(reviews_source, list) and all(
                isinstance(review, dict) for review in reviews_source):
            required_fields = {field.name for field in fields(Review) \
                               if field.init and \
                               not (getattr(field.type, '__origin__', None) is Union
                                    and type(None) in get_args(field.type))}
            return [
                cls(
                    **{
                        **{k: v for k, v in data.items() if k != 'emotion_analyse'},
                        'date': date.fromisoformat(data.get('date')) \
                            if isinstance(data.get('date'), str) else data.get('date'),
                        'sentiment_score': None \
                            if data.get('emotion_analyse') == 'true' or data.get('emotion_analyse') == 'false'
                        else int(data.get('sentiment_score')) \
                            if isinstance(data.get('sentiment_score'), str) else data.get('sentiment_score'),
                        'reply_count': int(data.get('reply_count', '0')) \
                            if isinstance(data.get('reply_count'), str) else data.get('reply_count')
                    }
                )
                for data in reviews_source if all(key in data for key in required_fields)
            ]
        else:
            raise ValueError(f"Invalid type {type(reviews_source)}")


@dataclass(kw_only=True)
class WeekData:
    start_date: date
    end_date: date
    reviews: list[Review] = field(default_factory=list)  # training input value
    box_office: int  # training input value

    @property
    def review_count(self) -> int:  # training input value
        return len(self.reviews)

    @property
    def average_sentiment_score(self) -> float:  # training input value
        scores = [review.sentiment_score for review in self.reviews if review.sentiment_score is not None]
        return np.mean(scores) if scores else 0.0

    @property
    def total_reply_count(self) -> int:  # training input value
        return sum(review.reply_count for review in self.reviews)

    def update_reviews(self, reviews_source: Path | YamlFile | list[dict | Review]) -> None:
        """
        Updates the reviews for the week data.

        Args:
            reviews_source (Path | YamlFile | list[dict | Review]): The source of reviews,
                                                                    which can be a Path, a YamlFile object,
                                                                    or a list of dictionaries or Review objects.

        Returns:
            None
        """
        processed_reviews: list[Review]  # Initialize a list to hold the final Review objects

        if isinstance(reviews_source, Path):
            # Load from Path using YamlFile
            loaded_data = YamlFile(path=reviews_source).load()
            processed_reviews = Review.create_reviews(reviews_source=loaded_data)
        elif isinstance(reviews_source, YamlFile):
            # Load from YamlFile object
            loaded_data = reviews_source.load()
            processed_reviews = Review.create_reviews(reviews_source=loaded_data)
        elif isinstance(reviews_source, list):
            if not reviews_source:
                processed_reviews = []
            elif all(isinstance(review_item, dict) for review_item in reviews_source):
                processed_reviews = Review.create_reviews(reviews_source=reviews_source)
            elif all(isinstance(review_item, Review) for review_item in reviews_source):
                processed_reviews = reviews_source
            else:
                raise TypeError("List must contain only dict or Review objects.")
        else:
            raise TypeError(f"Invalid type {type(reviews_source)} for reviews_source.")

        # Filter reviews by date range after all processing
        self.reviews = [review for review in processed_reviews if self.start_date <= review.date <= self.end_date]
        return

    def with_reviews(self, reviews_source: Path | YamlFile | list[dict | Review]) -> 'WeekData':
        """
        Updates the reviews for the week data and returns the WeekData instance itself.
        This method is designed to allow for method chaining or direct use in expressions.

        Args:
            reviews_source (Path | YamlFile | list[dict | Review]): The source of reviews.

        Returns:
            WeekData: The current WeekData instance after its reviews have been updated.
        """
        self.update_reviews(reviews_source=reviews_source)  # 內部呼叫原本的 update_reviews 進行實際更新
        return self  # 返回物件本身

    @classmethod
    def create_weeks_data(cls, weeks_data_source: Path | YamlFile | list[dict]) -> list['WeekData']:
        if isinstance(weeks_data_source, Path):
            return cls.create_weeks_data(weeks_data_source=YamlFile(path=weeks_data_source))
        elif isinstance(weeks_data_source, YamlFile):
            logging.info(f"loading file \"{weeks_data_source.path}\" to create multiple WeekData.")
            weeks_data: list[dict] = weeks_data_source.load()
            return cls.create_weeks_data(weeks_data_source=weeks_data)
        elif isinstance(weeks_data_source, list) and all(
                isinstance(week_data, dict) for week_data in weeks_data_source):
            required_fields = {field.name for field in fields(WeekData) \
                               if field.init and field.name != 'reviews' and \
                               not (getattr(field.type, '__origin__', None) is Union
                                    and type(None) in get_args(field.type))}
            return [
                cls(
                    box_office=int(data.get('box_office')) \
                        if isinstance(data.get('box_office'), str) else data.get('box_office'),
                    start_date=date.fromisoformat(data.get('start_date')) \
                        if isinstance(data.get('start_date'), str) else data.get('start_date'),
                    end_date=date.fromisoformat(data.get('end_date')) \
                        if isinstance(data.get('end_date'), str) else data.get('end_date'),
                    reviews=[]
                )
                for data in weeks_data_source
                if all(key in data for key in required_fields)
            ]
        else:
            raise ValueError(f"Invalid type {type(weeks_data_source)}")


@dataclass(kw_only=True)
class MovieSessionData:
    movie_id: int
    movie_name: str
    weeks_data: list[WeekData]  # len(week_data) = number_of_weeks

    def update_weeks_data(self, weeks_data_source: Path | YamlFile | list[dict | WeekData]) -> None:
        """
        Updates the weekly data for the movie session.

        Args:
            weeks_data_source (Path | YamlFile | list[dict | WeekData]): The source of weekly data,
                                                                          which can be a Path, a YamlFile object,
                                                                          or a list of dictionaries or WeekData objects.

        Returns:
            None
        """
        processed_weeks_data: list[WeekData]

        if isinstance(weeks_data_source, Path):
            # If the source is a Path, load it using YamlFile
            loaded_data: list[dict] = YamlFile(path=weeks_data_source).load()
            processed_weeks_data = WeekData.create_weeks_data(weeks_data_source=loaded_data)
        elif isinstance(weeks_data_source, YamlFile):
            # If the source is a YamlFile object, load data from it
            loaded_data: list[dict] = weeks_data_source.load()
            processed_weeks_data = WeekData.create_weeks_data(weeks_data_source=loaded_data)
        elif isinstance(weeks_data_source, list):
            if not weeks_data_source:  # Handle empty list explicitly
                processed_weeks_data = []
            elif all(isinstance(week_data_item, dict) for week_data_item in weeks_data_source):
                # If the list contains dictionaries, create WeekData objects from them
                processed_weeks_data = WeekData.create_weeks_data(weeks_data_source=weeks_data_source)
            elif all(isinstance(week_data_item, WeekData) for week_data_item in weeks_data_source):
                # If the list already contains WeekData objects, use them directly
                processed_weeks_data = weeks_data_source
            else:
                raise TypeError("List must contain only dict or WeekData objects.")
        else:
            raise TypeError(f"Invalid type {type(weeks_data_source)} for weeks_data_source.")

        self.weeks_data = processed_weeks_data
        return

    @classmethod
    def create_movie_sessions_from_index(cls, index_file_path: Path, number_of_weeks: int) -> list['MovieSessionData']:
        """

        :param index_file_path:
        :param number_of_weeks:
        :return:
        """
        """
                Creates a list of MovieSessionData objects by loading movie metadata from an index file,
                and integrating box office and review data.

                Args:
                    index_file_path (Path): The path to the CSV index file containing movie IDs and names.
                    number_of_weeks (int): The number of consecutive weeks of box office data to consider for each session.

                Returns:
                    list[MovieSessionData]: A list of MovieSessionData objects, each representing
                                             a movie session with its associated weekly data.
                """
        index_file: CsvFile = CsvFile(path=index_file_path)
        movies_data: list[dict] = index_file.load()

        all_movie_sessions: list[cls] = []

        for movie_data in movies_data:
            movie_id: int = movie_data.get('id')
            movie_name: str = movie_data.get('name')

            # Load box office data for the current movie
            box_office_file: YamlFile = YamlFile(path=Constants.BOX_OFFICE_FOLDER.joinpath(f"{movie_id}.yaml"))
            loaded_box_office: list[dict] = box_office_file.load()

            # Load review data for the current movie (loaded once per movie)
            review_file: YamlFile = YamlFile(path=Constants.PUBLIC_REVIEW_FOLDER.joinpath(f"{movie_id}.yaml"))
            loaded_reviews = Review.create_reviews(review_file)
            # We don't need to load reviews inside the inner loop repeatedly.
            # Instead, we pass the YamlFile object to update_reviews which handles loading.

            # Create batches of box office data for the specified number of weeks
            multiple_batches: list[list[dict]] = [
                loaded_box_office[i: i + number_of_weeks]
                for i in range(len(loaded_box_office) - number_of_weeks + 1)
            ]

            # Filter out batches where any week's box office is zero
            filtered_batches: list[list[dict]] = [
                single_batch
                for single_batch in multiple_batches
                if all(int(week_data.get('box_office')) != 0 for week_data in single_batch)
            ]

            for single_batch in filtered_batches:
                all_movie_sessions.append(
                    cls(
                        movie_id=movie_id,
                        movie_name=movie_name,
                        weeks_data=[week_data.with_reviews(reviews_source=loaded_reviews) \
                                    for week_data in WeekData.create_weeks_data(weeks_data_source=single_batch)])
                )

        return all_movie_sessions


if __name__ == '__main__':
    LoggingManager.create_predefined_manager()
    LoggingManager().get_logger('root').info('start')
    a = MovieSessionData.create_movie_sessions_from_index(index_file_path=Constants.INDEX_PATH, number_of_weeks=3)
    LoggingManager().get_logger('root').info('end')
    b = [x for x in a if x.weeks_data]
    wait: str = input("Enter to continue.")

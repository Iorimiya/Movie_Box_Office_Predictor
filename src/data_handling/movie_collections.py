from dataclasses import dataclass, field
from datetime import date
from itertools import chain
from logging import Logger
from pathlib import Path
from typing import Final, Literal, Optional, Type, TypeAlias, TypeVar

import numpy as np

from src.core.logging_manager import LoggingManager
from src.data_handling.box_office import BoxOffice, BoxOfficeRawData, BoxOfficeSerializableData
from src.data_handling.dataset import Dataset
from src.data_handling.file_io import YamlFile
from src.data_handling.movie_metadata import MovieMetadata, MoviePathMetadata
from src.data_handling.reviews import (
    ReviewRawData,
    PublicReview, PublicReviewRawData, PublicReviewSerializableData,
    ExpertReview, ExpertReviewRawData, ExpertReviewSerializableData
)
from src.utilities.util import delete_duplicate

WeekDataReviewType = TypeVar('WeekDataReviewType', PublicReview, ExpertReview)
PublicReviewLoadableSource: TypeAlias = Path | YamlFile | list[PublicReviewRawData]
ExpertReviewLoadableSource: TypeAlias = Path | YamlFile | list[ExpertReviewRawData]

MovieComponentSerializableData: TypeAlias = \
    list[BoxOfficeSerializableData] | list[PublicReviewSerializableData] | list[ExpertReviewSerializableData]
MovieComponent: TypeAlias = list[BoxOffice] | list[PublicReview] | list[ExpertReview]
COMPONENT_CLASS_MAP: Final[dict[Literal['box_office', 'public_reviews', 'expert_reviews']],
Type[BoxOffice] | Type[PublicReview] | Type[ExpertReview]] = {
    'box_office': BoxOffice, 'public_reviews': PublicReview, 'expert_reviews': ExpertReview
}


@dataclass(kw_only=True)
class WeekData:
    """
    Represents aggregated data for a single week of a movie's run.

    This includes box office figures and collections of public and expert reviews
    pertaining to that specific week.

    :ivar box_office_data: Box office information for the week.
    :ivar public_reviews: A list of public reviews published during the week.
    :ivar expert_reviews: A list of expert reviews published during the week.
    """
    box_office_data: BoxOffice
    public_reviews: list[PublicReview] = field(default_factory=list)
    expert_reviews: list[ExpertReview] = field(default_factory=list)

    @property
    def public_review_count(self) -> int:
        """
        Number of public reviews for the week.

        :return: The count of public reviews.
        """
        return len(self.public_reviews)

    @property
    def expert_review_count(self) -> int:
        """
        Number of expert reviews for the week.

        :return: The count of expert reviews.
        """
        return len(self.expert_reviews)

    @property
    def review_count(self) -> int:
        """
        Total number of public and expert reviews for the week.

        :return: The combined count of public and expert reviews.
        """
        return self.public_review_count + self.expert_review_count

    @property
    def average_sentiment_score(self) -> Optional[float]:
        """
        Average sentiment score of both public and expert reviews for the week.

        Calculates the mean of `sentiment_score` from all reviews
        where the score is not None.

        :return: The average sentiment score, or None if no scores are available.
        """

        # noinspection PyTypeChecker
        scores: list[float] = [review.sentiment_score for review in chain(self.public_reviews, self.expert_reviews) if
                               review.sentiment_score is not None]
        return float(np.mean(scores)) if scores else None

    @property
    def average_expert_score(self) -> Optional[float]:
        """
        Average expert score from expert reviews for the week.

        Calculates the mean of `expert_score` from all expert reviews.

        :return: The average expert score, or None if no expert reviews are available.
        """
        scores: list[float] = [review.expert_score for review in self.expert_reviews]
        return float(np.mean(scores)) if scores else None

    @property
    def total_reply_count(self) -> int:
        """
        Total reply count from all public reviews for the week.

        :return: The sum of reply counts from all public reviews.
        """
        return sum(review.reply_count for review in self.public_reviews)

    def _update_specific_reviews_list(
        self,
        reviews_source: Path | YamlFile | list[ReviewRawData] | list[WeekDataReviewType],
        review_class: Type[WeekDataReviewType]
    ) -> None:
        """
        Internal helper to update a specific list of reviews (public or expert) for the week.

        Loads reviews from the given source, creates instances of the specified `review_class`,
        filters them to include only those within the week's date range, and updates
        the target attribute (either 'public_reviews' or 'expert_reviews') on the instance.

        :param reviews_source: The source of review data.
        :param review_class: The specific Review subclass (e.g., PublicReview, ExpertReview) to create.
        :raises TypeError: If `review_class` is the base `Review` class, or if `reviews_source`
                           is a list containing mixed or invalid types.
        :raises ValueError: If `review_class` is not PublicReview or ExpertReview.
        """

        if review_class is PublicReview:
            target_attribute_name: str = 'public_reviews'
        elif review_class is ExpertReview:
            target_attribute_name: str = 'expert_reviews'
        else:

            raise ValueError(
                f"Unsupported review_class: {review_class.__name__}. "
                "Expected PublicReview or ExpertReview."
            )

        all_reviews: list[WeekDataReviewType] = review_class.create_multiple(source=reviews_source)

        filtered_reviews: list[WeekDataReviewType] = self._filter_review_by_week(
            reviews=all_reviews,
            start_date=self.box_office_data.start_date,
            end_date=self.box_office_data.end_date
        )

        setattr(self, target_attribute_name, filtered_reviews)
        return

    def update_public_reviews(
        self,
        reviews_source: PublicReviewLoadableSource | list[PublicReview]) -> None:
        """
        Updates the public reviews for this WeekData instance.

        Reviews are loaded from the source, filtered by the week's date range,
        and stored in the `public_reviews` attribute.

        :param reviews_source: The source of public review data.
        """
        self._update_specific_reviews_list(reviews_source=reviews_source, review_class=PublicReview)

    def update_expert_reviews(
        self,
        reviews_source: ExpertReviewLoadableSource | list[ExpertReview]) -> None:
        """
        Updates the expert reviews for this WeekData instance.

        Reviews are loaded from the source, filtered by the week's date range,
        and stored in the `expert_reviews` attribute.

        :param reviews_source: The source of expert review data.
        """
        self._update_specific_reviews_list(reviews_source=reviews_source, review_class=ExpertReview)

    def with_public_reviews(
        self,
        reviews_source: PublicReviewLoadableSource | list[PublicReview]) -> 'WeekData':
        """
        Updates public reviews and returns the WeekData instance for chaining.

        :param reviews_source: The source of public review data.
        :return: The WeekData instance itself.
        """
        self.update_public_reviews(reviews_source=reviews_source)
        return self

    def with_expert_reviews(
        self,
        reviews_source: ExpertReviewLoadableSource | list[ExpertReview]) -> 'WeekData':
        """
        Updates expert reviews and returns the WeekData instance for chaining.

        :param reviews_source: The source of expert review data.
        :return: The WeekData instance itself.
        """
        self.update_expert_reviews(reviews_source=reviews_source)
        return self

    @classmethod
    def create_multiple_week_data(cls,
                                  weeks_data_source: Path | YamlFile | list[BoxOfficeRawData] | list[BoxOffice],
                                  public_reviews_master_source: \
                                      Optional[PublicReviewLoadableSource | list[PublicReview]] = None,
                                  expert_reviews_master_source: \
                                      Optional[ExpertReviewLoadableSource | list[ExpertReview]] = None
                                  ) -> list['WeekData']:
        """
        Creates a list of WeekData objects from various sources.

        Each WeekData object represents a week's box office data and associated reviews.
        Reviews are filtered from the master sources to match each week's date range.

        :param weeks_data_source: The source of week data (box office, start/end dates).
        :param public_reviews_master_source: An optional master source for all public reviews.
                                             If provided, reviews will be filtered and assigned to relevant weeks.
        :param expert_reviews_master_source: An optional master source for all expert reviews.
                                             If provided, reviews will be filtered and assigned to relevant weeks.
        :return: A list of created WeekData objects.
        :raises ValueError: If `weeks_data_source` is of an invalid type (propagated from underlying calls).
        """
        logger: Logger = LoggingManager().get_logger('root')
        all_box_office_instances: list[BoxOffice] = BoxOffice.create_multiple(source=weeks_data_source)

        if not all_box_office_instances:
            logger.info(
                f"No valid BoxOffice instances could be created from weeks_data_source. Cannot create WeekData.")
            return []

        all_public_reviews: list[PublicReview] = PublicReview.create_multiple(source=public_reviews_master_source) \
            if public_reviews_master_source is not None else []
        all_expert_reviews: list[ExpertReview] = ExpertReview.create_multiple(source=expert_reviews_master_source) \
            if expert_reviews_master_source is not None else []

        assembled_weekly_data_list: list[tuple[BoxOffice, list[PublicReview], list[ExpertReview]]] = \
            [(current_box_office_data,
              cls._filter_review_by_week(
                  reviews=all_public_reviews,
                  start_date=current_box_office_data.start_date,
                  end_date=current_box_office_data.end_date),
              cls._filter_review_by_week(
                  reviews=all_expert_reviews,
                  start_date=current_box_office_data.start_date,
                  end_date=current_box_office_data.end_date))
             for current_box_office_data in all_box_office_instances]

        return [cls(box_office_data=box_office, public_reviews=public_reviews, expert_reviews=expert_reviews)
                for box_office, public_reviews, expert_reviews in assembled_weekly_data_list]

    @staticmethod
    def _filter_review_by_week(reviews: list[WeekDataReviewType], start_date: date, end_date: date) \
        -> list[WeekDataReviewType]:
        """
        Filters a list of reviews to include only those within a specific date range.

        :param reviews: A list of reviews (PublicReview or ExpertReview) to filter.
        :param start_date: The start date of the filtering period (inclusive).
        :param end_date: The end date of the filtering period (inclusive).
        :return: A new list containing only the reviews that fall within the specified date range.
        """
        logger: Logger = LoggingManager().get_logger('root')
        filtered_review: list[WeekDataReviewType] = []
        try:
            filtered_review = \
                [review for review in reviews if start_date <= review.date <= end_date]
        except Exception as e:
            logger.error(
                f"Error filtering reviews for week {start_date}-{end_date}: {e}")
        return filtered_review


@dataclass(kw_only=True)
class MovieSessionData(MovieMetadata):
    """
    Represents session data for a movie over several weeks.

    This includes the movie's ID, name, and a list of WeekData objects
    representing its performance and reviews over consecutive weeks.

    :ivar id: The unique identifier for the movie.
    :ivar name: The name of the movie.
    :ivar weeks_data: A list of WeekData objects for the movie session.
    """
    weeks_data: list[WeekData]

    @classmethod
    def _create_sessions_for_single_movie(cls,
                                          movie_meta_item: MoviePathMetadata,
                                          number_of_weeks: int) -> list['MovieSessionData']:
        """
        Helper method to process a single movie's metadata and create all its weekly sessions.

        Loads box office and review data for a movie, then segments the box office data
        into batches of `number_of_weeks`. For each valid batch, a MovieSessionData
        instance is created with corresponding WeekData.

        :param movie_meta_item: Metadata for the movie, including 'id', 'name',
                                and paths to its data files.
        :param number_of_weeks: The number of weeks each movie session should span.
        :return: A list of MovieSessionData objects for the given movie,
                  or an empty list if processing fails or no valid sessions are found.
        """

        movie_id_val: int = movie_meta_item.id
        movie_name_val: str = movie_meta_item.name
        box_office_file_path: Path = movie_meta_item.box_office_file_path
        public_reviews_file_path: Path = movie_meta_item.public_reviews_file_path
        expert_reviews_file_path: Path = movie_meta_item.expert_reviews_file_path

        logger: Logger = LoggingManager().get_logger('root')
        if movie_id_val is None or movie_name_val is None:
            logger.warning(f"Skipping movie due to missing ID or name in index: {movie_meta_item}")
            return []
        try:
            movie_id: int = int(movie_id_val) if isinstance(movie_id_val, str) else movie_id_val
        except ValueError:
            logger.warning(f"Skipping movie due to invalid ID '{movie_id_val}' in index: {movie_meta_item}")
            return []
        if not box_office_file_path or not box_office_file_path.exists():
            logger.warning(f"Box office file not found for movie ID {movie_id}. Skipping movie '{movie_name_val}'.")
            return []

        all_box_office_objects: list[BoxOffice] = BoxOffice.create_multiple(source=box_office_file_path)
        if not all_box_office_objects:
            logger.info(f"No valid BoxOffice objects could be created for movie ID {movie_id}. Skipping.")
            return []

        multiple_batches: list[list[BoxOffice]] = [
            all_box_office_objects[i: i + number_of_weeks]
            for i in range(len(all_box_office_objects) - number_of_weeks + 1)
        ]

        filtered_batches: list[list[BoxOffice]] = [
            batch for batch in multiple_batches
            if all(map(lambda week: week.box_office != 0, batch))
        ]
        if not filtered_batches:
            logger.info(f"No valid {number_of_weeks}-week sessions found after filtering for movie ID {movie_id}.")
            return []

        loaded_public_reviews: list[PublicReview] = []
        if public_reviews_file_path and public_reviews_file_path.exists():
            loaded_public_reviews = PublicReview.create_multiple(source=public_reviews_file_path)
        else:
            logger.info(f"Review file not found for movie ID {movie_id} ('{movie_name_val}')."
                        f"Proceeding without reviews for this movie.")

        loaded_expert_reviews: list[ExpertReview] = []
        if expert_reviews_file_path and expert_reviews_file_path.exists():
            loaded_expert_reviews = ExpertReview.create_multiple(source=expert_reviews_file_path)
        else:
            logger.info(f"Review file not found for movie ID {movie_id} ('{movie_name_val}')."
                        f"Proceeding without reviews for this movie.")

        return [cls(id=movie_id,
                    name=movie_name_val,
                    weeks_data=WeekData.create_multiple_week_data(
                        weeks_data_source=single_batch_week_dicts,
                        public_reviews_master_source=loaded_public_reviews,
                        expert_reviews_master_source=loaded_expert_reviews))
                for single_batch_week_dicts in filtered_batches]

    @classmethod
    def create_movie_sessions_from_dataset(cls, dataset_name: str, number_of_weeks: int) -> list['MovieSessionData']:
        """
        Creates MovieSessionData objects for all movies in a specified dataset.

        It reads an index file from the dataset to get movie metadata, then for each movie,
        it constructs paths to its box office and review files. These are then passed to
        `_create_sessions_for_single_movie` to generate the sessions.

        :param dataset_name: The name of the dataset to process.
        :param number_of_weeks: The number of weeks each movie session should span.
        :return: A flattened list of all MovieSessionData objects created from the dataset.
        :raises FileNotFoundError: If the dataset's index file is not found (propagated).
        :raises ValueError: If the index file is found but contains no processable movie metadata (propagated).
        """

        logger: Logger = LoggingManager().get_logger("root")
        dataset: Dataset = Dataset(name=dataset_name)
        source_infos: list[MoviePathMetadata] = dataset.load_movie_source_info()
        if not source_infos:
            logger.info(f"No processable movie metadata after initial validation from '{dataset.index_file_path}'.")
            return []
        return chain.from_iterable([cls._create_sessions_for_single_movie(
            movie_meta_item=movie_meta_with_paths_item,
            number_of_weeks=number_of_weeks)
            for movie_meta_with_paths_item in source_infos
        ])


@dataclass(kw_only=True)
class MovieData(MovieMetadata):
    """
    Represents comprehensive data for a single movie.

    This includes its metadata (ID, name), and lists of all its
    box office records, public reviews, and expert reviews.

    Inherits 'id' and 'name' from :class:`~.MovieMetadata`.

    :ivar box_office: A list of all box office records for the movie.
    :ivar public_reviews: A list of all public reviews for the movie.
    :ivar expert_reviews: A list of all expert reviews for the movie.
    """
    box_office: list[BoxOffice] = field(default_factory=list)
    public_reviews: list[PublicReview] = field(default_factory=list)
    expert_reviews: list[ExpertReview] = field(default_factory=list)

    @property
    def box_office_week_lens(self) -> int:
        """
        Returns the number of weeks for which box office data is available.

        :return: The count of ``BoxOffice`` records, or 0 if none exist.
        """
        return len(self.box_office) if self.box_office else 0

    @property
    def public_reply_count(self) -> int:
        """
        Returns the total number of replies across all available public reviews.

        :return: The sum of reply counts from all ``PublicReview`` objects,
                  or 0 if no public reviews exist.
        """
        return sum(public_review.reply_count for public_review in self.public_reviews)

    @property
    def public_review_count(self) -> int:
        """
        Returns the number of public reviews available for the movie.

        :return: The count of ``PublicReview`` objects, or 0 if none exist.
        """
        return len(self.public_reviews) if self.public_reviews else 0

    @classmethod
    def load_multiple_movie_data_from_dataset(cls, dataset_name: str, mode: Literal['ALL', 'META']) \
        -> list['MovieData']:
        """
        Loads multiple MovieData instances from a specified dataset.

        Delegates to the Dataset class to load all movie data based on the given mode.

        :param dataset_name: The name of the dataset to load from.
        :param mode: Specifies the loading mode ('ALL' for full data, 'META' for metadata only).
        :return: A list of MovieData instances.
        """
        return Dataset(name=dataset_name).load_all_movie_data(mode=mode)

    def __save_component(self, component_type: Literal['box_office', 'public_reviews', 'expert_reviews'],
                         target_directory: Path) -> None:
        """
        Internal helper to save a specific component (box office, public reviews, or expert reviews) to a YAML file.

        The data for the specified component is serialized and saved to a file
        named after the movie's ID within the target directory.

        :param component_type: The type of component to save.
        :param target_directory: The directory where the component's YAML file will be saved.
        :raises Exception: Propagates exceptions from `YamlFile.save()`.
        """
        logger: Logger = LoggingManager().get_logger("root")
        component_name: str = component_type.replace('_', ' ')
        output_file_path: Path = target_directory / f"{self.id}.yaml"

        logger.info(
            f"Attempting to save {component_name} for movie ID {self.id} to '{output_file_path}'."
        )
        component_data: MovieComponentSerializableData = [component.as_serializable_dict() for component in
                                                          getattr(self, component_type, [])]
        try:
            YamlFile(path=output_file_path).save(component_data)
        except Exception as e:
            logger.error(f"Error saving {component_name} data for movie ID {self.id} to '{output_file_path}': {e}")
            raise
        logger.info(
            f"Successfully saved {len(component_data)} {component_name} items "
            f"for movie ID {self.id} to '{output_file_path}'."
        )

    def save_box_office(self, target_directory: Path) -> None:
        """
        Saves the movie's box office data to a YAML file in the specified directory.

        :param target_directory: The directory where the box office data file will be saved.
        """
        self.__save_component(component_type='box_office', target_directory=target_directory)

    def save_public_reviews(self, target_directory: Path) -> None:
        """
        Saves the movie's public reviews data to a YAML file in the specified directory.

        :param target_directory: The directory where the public reviews data file will be saved.
        """
        self.__save_component(component_type='public_reviews', target_directory=target_directory)

    def save_expert_reviews(self, target_directory: Path) -> None:
        """
        Saves the movie's expert reviews data to a YAML file in the specified directory.

        :param target_directory: The directory where the expert reviews data file will be saved.
        """
        self.__save_component(component_type='expert_reviews', target_directory=target_directory)

    def __load_component(self, component_type: Literal['box_office', 'public_reviews', 'expert_reviews'],
                         target_directory: Path) -> None:
        """
        Internal helper to load a specific component's data from a YAML file.

        If the file exists, data is loaded, and instances of the appropriate
        class (BoxOffice, PublicReview, or ExpertReview) are created and
        assigned to the corresponding attribute of this MovieData instance.
        If the file doesn't exist or an error occurs, the attribute is set to an empty list.

        :param component_type: The type of component to load.
        :param target_directory: The directory from which to load the component's YAML file.
        :raises ValueError: If an internal error occurs due to a missing class mapping for `component_type`.
        """
        logger: Logger = LoggingManager().get_logger("root")
        component_name_for_log: str = component_type.replace('_', ' ')
        source_file_path: Path = target_directory / f"{self.id}.yaml"

        logger.info(
            f"Attempting to load {component_name_for_log} for movie ID {self.id} from '{source_file_path}'."
        )


        if not source_file_path.exists():
            logger.warning(
                f"Data file not found for {component_name_for_log} for movie ID {self.id} at '{source_file_path}'. "
                f"The '{component_type}' list in MovieData instance will be empty."
            )
            setattr(self, component_type, [])
            return

        try:

            target_class: Optional[Type[BoxOffice] | Type[PublicReview] | Type[ExpertReview]] = \
                COMPONENT_CLASS_MAP.get(component_type)

            if target_class is None:
                msg = (f"Internal error: No class mapping found for component type '{component_type}' "
                       f"for movie ID {self.id}")
                logger.critical(msg)
                raise ValueError(msg)

            loaded_items_list: MovieComponent = target_class.create_multiple(source=source_file_path)

            setattr(self, component_type, loaded_items_list)


        except Exception as e:
            logger.error(
                f"Unexpected error during the loading process of {component_name_for_log} data "
                f"for movie ID {self.id} from '{source_file_path}': {e}"
            )
            setattr(self, component_type, [])

    def load_box_office(self, target_directory: Path) -> None:
        """
        Loads the movie's box office data from a YAML file in the specified directory.

        Updates the `box_office` attribute of this instance.

        :param target_directory: The directory from which to load the box office data file.
        """
        self.__load_component(component_type='box_office', target_directory=target_directory)

    def load_public_reviews(self, target_directory: Path) -> None:
        """
        Loads the movie's public reviews data from a YAML file in the specified directory.

        Updates the `public_reviews` attribute of this instance.

        :param target_directory: The directory from which to load the public reviews data file.
        """
        self.__load_component(component_type='public_reviews', target_directory=target_directory)

    def load_expert_reviews(self, target_directory: Path) -> None:
        """
        Loads the movie's expert reviews data from a YAML file in the specified directory.

        Updates the `expert_reviews` attribute of this instance.

        :param target_directory: The directory from which to load the expert reviews data file.
        """
        self.__load_component(component_type='expert_reviews', target_directory=target_directory)

    def __update_component(self, component_type: Literal['box_office', 'public_reviews', 'expert_reviews'],
                           update_method: Literal['REPLACE', 'EXTEND'],
                           data: MovieComponent) -> None:
        """
        Internal helper to update a specific component's data list (box office, public reviews, or expert reviews).

        The component's data can be either replaced entirely or extended with new data.
        Duplicate items are removed after the operation.

        :param component_type: The type of component to update.
        :param update_method: The method of update ('REPLACE' or 'EXTEND').
        :param data: A list of new data items (BoxOffice, PublicReview, or ExpertReview instances).
        :raises ValueError: If an invalid `update_method` is provided.
        """
        logger: Logger = LoggingManager().get_logger("root")
        component_name_for_log: str = component_type.replace('_', ' ')
        incoming_data_count: int = len(data)

        logger.info(
            f"Attempting to update {component_name_for_log} for movie ID {self.id} "
            f"using method '{update_method}' with {incoming_data_count} new items."
        )

        current_data_list: MovieComponent = getattr(self, component_type, [])
        original_data_count: int = len(current_data_list)

        match update_method:
            case 'replace':
                deduplicated_new_data: MovieComponent = delete_duplicate(data)
                setattr(self, component_type, deduplicated_new_data)
                new_count: int = len(deduplicated_new_data)
                logger.info(
                    f"Replaced {component_name_for_log} for movie ID {self.id}. "
                    f"Previous count: {original_data_count}, New count: {new_count}."
                )
            case 'extend':
                logger.info(
                    f"Extending {component_name_for_log} for movie ID {self.id}. "
                    f"Original count: {original_data_count}, Items to add: {incoming_data_count}."
                )
                combined_data: MovieComponent = current_data_list + data

                deduplicated_new_data: MovieComponent = delete_duplicate(combined_data)
                setattr(self, component_type, deduplicated_new_data)
                final_count: int = len(deduplicated_new_data)
                logger.info(
                    f"Extended and deduplicated {component_name_for_log} for movie ID {self.id}. "
                    f"Final count: {final_count} (was {original_data_count}, added {incoming_data_count} before deduplication)."
                )
            case _:
                msg: str = f"Invalid update_method \"{update_method}\" for component '{component_type}' on movie ID {self.id}."
                logger.error(msg)
                raise ValueError(msg)

    def update_box_office(self, update_method: Literal['REPLACE', 'EXTEND'], data: list[BoxOffice]) -> None:
        """
        Updates the movie's box office data.

        The existing box office data can be replaced or extended with the provided data.
        Duplicates are handled.

        :param update_method: How to update ('replace' or 'extend').
        :param data: A list of new BoxOffice instances.
        """
        self.__update_component(component_type='box_office', update_method=update_method, data=data)

    def update_public_reviews(self, update_method: Literal['REPLACE', 'EXTEND'], data: list[PublicReview]) -> None:
        """
        Updates the movie's public reviews data.

        The existing public reviews can be replaced or extended with the provided data.
        Duplicates are handled.

        :param update_method: How to update ('replace' or 'extend').
        :param data: A list of new PublicReview instances.
        """
        self.__update_component(component_type='public_reviews', update_method=update_method, data=data)

    def update_expert_reviews(self, update_method: Literal['REPLACE', 'EXTEND'], data: list[ExpertReview]) -> None:
        """
        Updates the movie's expert reviews data.

        The existing expert reviews can be replaced or extended with the provided data.
        Duplicates are handled.

        :param update_method: How to update ('replace' or 'extend').
        :param data: A list of new ExpertReview instances.
        """
        self.__update_component(component_type='expert_reviews', update_method=update_method, data=data)

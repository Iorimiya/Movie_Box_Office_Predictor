from argparse import ArgumentParser, Namespace
from logging import Logger
from pathlib import Path

from src.core.logging_manager import LoggingManager
from src.core.project_config import ProjectDatasetType, ProjectModelType, ProjectPaths
from src.data_collection.box_office_collector import BoxOfficeCollector
from src.data_collection.review_collector import ReviewCollector, TargetWebsite
from src.data_handling.box_office import BoxOffice
from src.data_handling.dataset import Dataset
from src.data_handling.file_io import CsvFile
from src.data_handling.reviews import PublicReview


class DatasetHandler:
    """
    Handles Command-Line Interface (CLI) commands related to dataset management.

    :ivar _logger: The logger instance for this class.
    :ivar _parser: The argument parser instance for handling CLI arguments.
    """
    _logger: Logger
    _parser: ArgumentParser

    def __init__(self, parser: ArgumentParser) -> None:
        """
        Initializes the DatasetHandler.

        :param parser: The argument parser instance.
        """
        self._parser = parser
        self._logger = LoggingManager().get_logger()

    def create_index(self, args: Namespace) -> None:
        """
        Creates an index file for a structured dataset from a source CSV file.

        :param args: The namespace object containing command-line arguments,
                     expected to have 'structured_dataset_name' and 'source_file'.
        :raises FileNotFoundError: If the specified source file does not exist.
        """
        self._logger.info(
            f"Executing: Create dataset index for '{args.structured_dataset_name}'"
        )
        source_path: Path = ProjectPaths.raw_index_sources_dir / args.source_file
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found at: {source_path}")

        Dataset(name=args.structured_dataset_name).initialize_index_file(
            source_csv=CsvFile(path=source_path)
        )

    @staticmethod
    def _validate_dataset_path(dataset_name: str) -> Path:
        """
        Validates the existence of a structured dataset path.

        :param dataset_name: The name of the structured dataset.
        :returns: The validated path to the dataset.
        :raises FileNotFoundError: If the dataset path does not exist.
        """
        dataset_path: Path = ProjectPaths.get_dataset_path(
            dataset_name=dataset_name,
            dataset_type=ProjectDatasetType.STRUCTURED
        )
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset '{dataset_name}' not found at expected path: {dataset_path}")
        return dataset_path

    @staticmethod
    def _log_box_office_data(logger: Logger, movie_name: str, box_office_data: list[BoxOffice]) -> None:
        """
        Logs formatted box office data for a movie.

        :param logger: The logger instance to use for logging.
        :param movie_name: The name of the movie.
        :param box_office_data: A list of BoxOffice objects.
        """
        logger.info(f"--- Box Office Data for '{movie_name}' ---")
        for entry in box_office_data:
            logger.info(str(entry))
        logger.info(f"---------------------------------------")

    @staticmethod
    def _log_reviews_data(logger: Logger, movie_name: str, reviews: list[PublicReview], review_type: str) -> None:
        """
        Logs formatted review data for a movie.

        :param logger: The logger instance to use for logging.
        :param movie_name: The name of the movie.
        :param reviews: A list of PublicReview objects.
        :param review_type: The type of review (e.g., "PTT", "Dcard").
        """
        logger.info(f"--- {review_type} Reviews for '{movie_name}' ---")
        for i, review in enumerate(reviews):
            logger.info(f"Review {i + 1}:")
            logger.info(str(review))
            if i < len(reviews) - 1:
                logger.info("-" * 40)
        logger.info(f"---------------------------------------")

    def collect_box_office(self, args: Namespace) -> None:
        """
        Collects box office data for an entire dataset or a single movie.

        :param args: The namespace object containing command-line arguments,
                     expected to have 'structured_dataset_name' or 'movie_name'.
        :raises FileNotFoundError: If the specified dataset path does not exist.
        """
        if args.structured_dataset_name:
            dataset_name: str = args.structured_dataset_name
            self._logger.info(f"Executing: Collect box office data for dataset '{dataset_name}'.")
            self._validate_dataset_path(dataset_name=dataset_name)
            Dataset(name=dataset_name).collect_box_office()
        elif args.movie_name:
            movie_name: str = args.movie_name
            self._logger.info(f"Executing: Collect box office data for movie '{movie_name}'.")

            try:
                with BoxOfficeCollector(download_mode='WEEK') as collector:
                    box_office_data: list[BoxOffice] = collector.download_box_office_data_for_movie(
                        movie_name=movie_name
                    )

                    if box_office_data:
                        self._logger.info(f"Successfully retrieved box office data for '{movie_name}'.")
                        self._log_box_office_data(
                            logger=self._logger, movie_name=movie_name, box_office_data=box_office_data
                        )
                    else:
                        self._logger.warning(f"No box office data found for '{movie_name}'.")
            except RuntimeError as e:
                self._logger.error(f"Failed to retrieve box office data for '{movie_name}': {e}")
            except Exception as e:
                self._logger.error(
                    f"An unexpected error occurred while collecting box office data for '{movie_name}': {e}",
                    exc_info=True
                )

    def _collect_reviews(
        self, args: Namespace, target_website: TargetWebsite, website_display_name: str
    ) -> None:
        """
        A generic helper method to collect public reviews for a given website.

        This method contains the shared logic for collecting reviews for either
        an entire dataset or a single movie, abstracting away the specific
        target website.

        :param args: The namespace object containing command-line arguments.
        :param target_website: The enum member representing the target website (e.g., TargetWebsite.PTT).
        :param website_display_name: The user-friendly name of the website for logging (e.g., "PTT").
        :raises FileNotFoundError: If the specified dataset path does not exist.
        """
        if args.structured_dataset_name:
            dataset_name: str = args.structured_dataset_name
            self._logger.info(f"Executing: Collect {website_display_name} reviews for dataset '{dataset_name}'.")
            self._validate_dataset_path(dataset_name=dataset_name)
            # noinspection PyTypeChecker
            Dataset(name=dataset_name).collect_public_review(target_website=target_website.name)
        elif args.movie_name:
            movie_name: str = args.movie_name
            self._logger.info(f"Executing: Collect {website_display_name} reviews for movie '{movie_name}'.")

            try:
                collector = ReviewCollector(target_website=target_website)
                reviews: list[PublicReview] = collector.collect_reviews_for_movie(movie_name=movie_name)
                if reviews:
                    self._logger.info(f"Successfully retrieved {website_display_name} reviews for '{movie_name}'.")
                    self._log_reviews_data(
                        logger=self._logger,
                        movie_name=movie_name,
                        reviews=reviews,
                        review_type=website_display_name
                    )
                else:
                    self._logger.warning(f"No {website_display_name} reviews found for '{movie_name}'.")

            except Exception as e:
                self._logger.error(
                    f"An error occurred while collecting {website_display_name} reviews for '{movie_name}': {e}",
                    exc_info=True
                )

    def collect_ptt_review(self, args: Namespace) -> None:
        """
        Collects PTT reviews for an entire dataset or a single movie.

        This method delegates the core logic to the generic `_collect_reviews` helper.

        :param args: The namespace object containing command-line arguments,
                     expected to have 'structured_dataset_name' or 'movie_name'.
        """
        self._collect_reviews(
            args=args,
            target_website=TargetWebsite.PTT,
            website_display_name="PTT"
        )

    def collect_dcard_review(self, args: Namespace) -> None:
        """
        Collects Dcard reviews for an entire dataset or a single movie.

        This method delegates the core logic to the generic `_collect_reviews` helper.

        :param args: The namespace object containing command-line arguments,
                     expected to have 'structured_dataset_name' or 'movie_name'.
        """
        self._collect_reviews(
            args=args,
            target_website=TargetWebsite.DCARD,
            website_display_name="Dcard"
        )

    def compute_sentiment(self, args: Namespace) -> None:
        """
        Computes sentiment scores for a structured dataset using a specified model.

        :param args: The namespace object containing command-line arguments,
                     expected to have 'structured_dataset_name', 'model_id', and 'epoch'.
        :raises FileNotFoundError: If the dataset or model path does not exist.
        """
        self._logger.info(
            f"Executing: Compute sentiment for dataset '{args.structured_dataset_name}' "
            f"using model '{args.model_id}' (epoch: {args.epoch})."
        )
        dataset_name: str = args.structured_dataset_name
        self._validate_dataset_path(dataset_name=dataset_name)

        model_path: Path = ProjectPaths.get_model_root_path(
            model_id=args.model_id,
            model_type=ProjectModelType.SENTIMENT
        )
        if not model_path.exists():
            raise FileNotFoundError(f"Model '{args.model_id}' not found at expected path: {model_path}")

        Dataset(name=dataset_name).compute_sentiment(
            model_id=args.model_id,
            model_epoch=args.epoch
        )

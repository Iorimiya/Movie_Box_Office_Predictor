import logging
import random
from argparse import Namespace
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final, Optional

from numpy import argmax
from numpy.typing import NDArray

from src.core.project_config import ProjectModelType, ProjectPaths
from src.data_collection.box_office_collector import BoxOfficeCollector
from src.data_collection.review_collector import ReviewCollector, TargetWebsite
from src.data_handling.box_office import BoxOffice
from src.data_handling.dataset import BaseDataset, DatabaseDataset
from src.data_handling.file_io import YamlFile
from src.data_handling.movie_collections import MovieData
from src.data_handling.reviews import PublicReview
from src.models.box_office_classification.components.data_processor import (
    BoxOfficeClassificationConfigDict,
    BoxOfficeClassificationDataProcessor,
)
from src.models.box_office_classification.components.model_core import (
    BoxOfficeClassificationModelCore,
    BoxOfficeClassificationPredictParams,
)
from src.models.box_office_classification.pipelines.training_pipeline import (
    BoxOfficeClassificationPipelineConfig,
    BoxOfficeClassificationTrainingPipeline,
)


class BoxOfficeClassificationModelHandler:
    """
    Handles CLI commands related to the Box Office Classification Model.

    This class manages the complete lifecycle for the classification model,
    including training with database data and making predictions.

    :cvar _DEFAULT_CONFIG_FILENAME: The filename for default classification settings.
    """

    _DEFAULT_CONFIG_FILENAME: Final[str] = "box_office_classification_defaults.yaml"

    def __init__(self) -> None:
        """
        Initializes the BoxOfficeClassificationModelHandler.
        """
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._model_type: ProjectModelType = ProjectModelType.BOX_OFFICE_CLASSIFICATION

    def train(self, args: Namespace) -> None:
        """
        Orchestrates the Box Office Classification Model training process.

        :param args: The arguments containing model configuration and training parameters.
        """
        self._logger.info(f"Starting training for classification model: {args.model_id}")

        effective_config: BoxOfficeClassificationConfigDict = self._prepare_training_config(args=args)

        try:
            artifacts_folder: Path = ProjectPaths.get_model_root_path(
                model_id=args.model_id, model_type=self._model_type
            )
            artifacts_folder.mkdir(parents=True, exist_ok=True)

            # Save the final effective configuration to config.yaml
            final_config_path: Path = artifacts_folder / "config.yaml"
            YamlFile(path=final_config_path).save_single_document(data=effective_config)
            self._logger.info(f"Effective configuration saved to: {final_config_path}")

            pipeline_config: BoxOfficeClassificationPipelineConfig = BoxOfficeClassificationPipelineConfig(
                **effective_config
            )
            data_processor: BoxOfficeClassificationDataProcessor = BoxOfficeClassificationDataProcessor(
                model_artifacts_path=artifacts_folder,
                box_office_thresholds=pipeline_config.box_office_thresholds
            )
            model_core: BoxOfficeClassificationModelCore = BoxOfficeClassificationModelCore()

            pipeline: BoxOfficeClassificationTrainingPipeline = BoxOfficeClassificationTrainingPipeline(
                data_processor=data_processor, model_core=model_core
            )

            pipeline.run(
                config=pipeline_config,
                continue_from_epoch=args.continue_from_epoch
            )
        except Exception as e:
            self._logger.error(f"Training failed: {e}", exc_info=True)
            raise RuntimeError(f"Pipeline execution failed: {e}")

    def predict(self, args: Namespace) -> None:
        """
        Makes a prediction and prints the resulting class label index.

        :param args: The arguments containing model ID, epoch, and input data source.
        """
        # 1. Load Model and Components
        try:
            artifacts_folder: Path = ProjectPaths.get_model_root_path(
                model_id=args.model_id, model_type=self._model_type
            )

            config_path: Path = artifacts_folder / "config.yaml"
            if not config_path.exists():
                raise FileNotFoundError(f"Config for model '{args.model_id}' not found.")

            original_config: dict[str, Any] = YamlFile(path=config_path).load_single_document()
            model_file_path: Path = artifacts_folder / f"{args.model_id}_{args.epoch:04d}.keras"

            data_processor: BoxOfficeClassificationDataProcessor = BoxOfficeClassificationDataProcessor(
                model_artifacts_path=artifacts_folder
            )

            if not data_processor.is_prepared:
                raise ValueError(f"Artifacts (scaler/thresholds) missing for model '{args.model_id}'.")

            model_core: BoxOfficeClassificationModelCore = BoxOfficeClassificationModelCore(model_path=model_file_path)

        except Exception as e:
            self._logger.error(f"Failed to load model components: {e}")
            raise

        # 2. Prepare Input Data
        try:
            input_data: MovieData
            if args.movie_name:
                # Use real-time search logic instead of just database retrieval
                input_data = self._fetch_movie_data_online(movie_name=args.movie_name)

                # Validation: ensure we have enough data to process
                required_weeks: int = original_config['training_week_len']
                if len(input_data.box_office) < required_weeks:
                    self._logger.warning(
                        f"Fetched data for '{args.movie_name}' only has {len(input_data.box_office)} weeks, "
                        f"but model requires {required_weeks}. Attempting to use database as fallback."
                    )
                    dataset_name: str = original_config.get('dataset_name', '')
                    dataset: BaseDataset = DatabaseDataset(name=dataset_name)
                    target_movie: Optional[MovieData] = next(
                        (m for m in dataset.movie_data if m.name == args.movie_name), None
                    )
                    if target_movie:
                        input_data = target_movie
                    else:
                        raise ValueError(
                            f"Insufficient online data and movie '{args.movie_name}' not found in database."
                        )
            else:
                input_data = self._generate_random_movie_data(weeks=original_config['training_week_len'] + 5)

            # 3. Process and Predict
            from src.models.box_office_common.box_office_data_processor import BoxOfficeDataConfig
            processing_config: BoxOfficeDataConfig = BoxOfficeDataConfig(
                training_week_len=original_config['training_week_len']
            )

            processed_input: NDArray[Any] = data_processor.process_for_prediction(
                single_input=input_data, config=processing_config
            )

            pred_params: BoxOfficeClassificationPredictParams = BoxOfficeClassificationPredictParams(verbose=0)
            # Output is softmax probabilities
            probabilities: NDArray[Any] = model_core.predict(data=processed_input, params=pred_params)

            # 4. Final Output (Label Index only)
            predicted_class_index: int = int(argmax(probabilities, axis=1)[0])
            print(predicted_class_index)

        except Exception as e:
            self._logger.error(f"Prediction failed: {e}", exc_info=True)
            raise

    def _prepare_training_config(self, args: Namespace) -> BoxOfficeClassificationConfigDict:
        """
        Loads default settings and merges them with command-line arguments.

        :param args: The command-line arguments.
        :return: A dictionary containing the final configuration.
        """
        defaults_path: Path = ProjectPaths.configs_dir / self._DEFAULT_CONFIG_FILENAME
        if not defaults_path.exists():
            raise FileNotFoundError(f"Default config not found at: {defaults_path}")

        config: BoxOfficeClassificationConfigDict = YamlFile(path=defaults_path).load_single_document()

        # Override with CLI arguments if provided
        if args.model_id:
            config['model_id'] = args.model_id
        if args.dataset_name:
            config['dataset_name'] = args.dataset_name
        if args.epochs:
            config['epochs'] = args.epochs
        if args.batch_size:
            config['batch_size'] = args.batch_size
        if hasattr(args, 'box_office_thresholds') and args.box_office_thresholds:
            config['box_office_thresholds'] = tuple(args.box_office_thresholds)

        return config

    def _fetch_movie_data_online(self, movie_name: str) -> MovieData:
        """
        Fetches movie data (box office and reviews) from online sources in real-time.

        :param movie_name: The name of the movie to fetch data for.
        :return: A MovieData instance containing the fetched data.
        """
        self._logger.info(f"Fetching online data for movie: {movie_name}")

        box_office_history: list[BoxOffice] = []
        try:
            with BoxOfficeCollector(download_mode='WEEK') as collector:
                box_office_history, _ = collector.fetch_single_movie_data(movie_name=movie_name)
        except Exception as e:
            self._logger.warning(f"Failed to fetch box office data online for '{movie_name}': {e}")

        all_public_reviews: list[PublicReview] = []
        # Fetch PTT reviews
        try:
            ptt_collector: ReviewCollector = ReviewCollector(target_website=TargetWebsite.PTT)
            all_public_reviews.extend(ptt_collector.collect_reviews_for_movie(movie_name=movie_name))
        except Exception as e:
            self._logger.warning(f"Failed to fetch PTT reviews online for '{movie_name}': {e}")

        return MovieData(
            id=-1,
            name=movie_name,
            box_office=box_office_history,
            public_reviews=all_public_reviews,
            expert_reviews=[]
        )

    def _generate_random_movie_data(self, weeks: int) -> MovieData:
        """
        Generates random movie data for testing/prediction.

        :param weeks: Number of weeks to generate.
        :return: A MovieData instance.
        """
        box_office_history: list[BoxOffice] = []
        end_date: date = date.today()
        for _ in range(weeks):
            start_date: date = end_date - timedelta(days=6)
            box_office_history.append(BoxOffice(
                start_date=start_date,
                end_date=end_date,
                amount=random.randint(100_000, 10_000_000)
            ))
            end_date = start_date - timedelta(days=1)

        box_office_history.reverse()
        return MovieData(
            id=-1,
            name="Random Movie",
            box_office=box_office_history,
            public_reviews=[],
            expert_reviews=[]
        )

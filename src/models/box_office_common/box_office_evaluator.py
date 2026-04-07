from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Generic, TypeVar

from numpy.typing import NDArray
from typing_extensions import override

from src.core.project_config import ProjectModelType, ProjectPaths
from src.data_handling.dataset import DatabaseDataset, YamlDataset
from src.data_handling.movie_collections import MovieSessionData
from src.models.base.base_model_core import BaseModelCore
from src.models.base.evaluation import (
    BaseEvaluator,
    EvaluationResultType,
    GradientEvaluationConfig
)
from src.models.box_office_common.box_office_data_processor import (
    BoxOfficeDataConfig,
    BoxOfficeDataSource,
    BoxOfficeSessionDataProcessor
)


# Define DataSourceType Enum
class DataSourceType(Enum):
    """
    Enumeration for different types of data sources.
    """
    DATABASE = "database"
    YAML = "yaml"


class BoxOfficeBaseEvaluationConfig(GradientEvaluationConfig):
    """
    A base configuration for Box Office model evaluation, extending GradientEvaluationConfig.

    This class adds the `training_week_len` parameter, which is common to both
    regression and classification box office models.
    """

    def __init__(
        self,
        *,
        training_week_len: int,
        **kwargs: Any
    ):
        """
        Initializes the BoxOfficeBaseEvaluationConfig.

        :param training_week_len: The number of past weeks used for input sequences.
        :param kwargs: Additional keyword arguments passed to the base class.
        """
        super().__init__(**kwargs)
        self.training_week_len = training_week_len


# Define generic types for the base evaluator
DataProcessorType = TypeVar('DataProcessorType', bound=BoxOfficeSessionDataProcessor)
ModelCoreType = TypeVar('ModelCoreType', bound=BaseModelCore)
BoxOfficeEvaluationConfigType = TypeVar('BoxOfficeEvaluationConfigType', bound=BoxOfficeBaseEvaluationConfig)


class BoxOfficeBaseEvaluator(
    BaseEvaluator[DataProcessorType, ModelCoreType, BoxOfficeEvaluationConfigType, EvaluationResultType],
    Generic[DataProcessorType, ModelCoreType, BoxOfficeEvaluationConfigType, EvaluationResultType]
):
    """
    A common base class for Box Office model evaluators (Regression and Classification).

    This class provides shared logic for setting up components and preparing test data,
    allowing for flexible data source selection (Database or YAML).
    """
    _data_source_type: DataSourceType

    def __init__(self, data_source_type: DataSourceType = DataSourceType.DATABASE) -> None:
        """
        Initializes the BoxOfficeBaseEvaluator.

        :param data_source_type: The type of data source to use for evaluation (Database or YAML).
        """
        super().__init__()
        self._data_source_type = data_source_type

    @property
    @abstractmethod
    def _project_model_type(self) -> ProjectModelType:
        """
        Abstract property to be implemented by subclasses, returning the specific
        ProjectModelType (e.g., BOX_OFFICE_REGRESSION, BOX_OFFICE_CLASSIFICATION).
        """
        pass

    @abstractmethod
    def _create_data_processor_instance(self, model_artifacts_path: Path) -> DataProcessorType:
        """
        Abstract method to create and return an instance of the specific DataProcessor.
        """
        pass

    @abstractmethod
    def _create_model_core_instance(self, model_file_path: Path) -> ModelCoreType:
        """
        Abstract method to create and return an instance of the specific ModelCore.
        """
        pass

    def _get_data_source(self, dataset_name: str) -> BoxOfficeDataSource:
        """
        Returns an instance of the appropriate BaseDataset subclass based on _data_source_type.

        :param dataset_name: The name of the dataset.
        :return: An instance of DatabaseDataset or YamlDataset.
        """
        if self._data_source_type == DataSourceType.YAML:
            return YamlDataset(name=dataset_name)
        return DatabaseDataset(name=dataset_name)

    @override
    def _setup_components(
        self, model_id: str, model_epoch: int
    ) -> tuple[DataProcessorType, ModelCoreType, Path]:
        """
        Sets up and loads the necessary data processor and model core for evaluation.

        This method is now generic and uses abstract methods to create specific
        DataProcessor and ModelCore instances.

        :param model_id: The unique identifier for the model series.
        :param model_epoch: The specific training epoch of the model to load.
        :return: A tuple containing the initialized data processor, model core,
                 and the path to the model artifacts' directory.
        :raises FileNotFoundError: If required artifacts (e.g., scaler, thresholds) cannot be loaded.
        """
        self.logger.debug("Loading model and data processor artifacts...")
        artifacts_path: Path = ProjectPaths.get_model_root_path(
            model_id=model_id, model_type=self._project_model_type
        )
        model_file_path: Path = artifacts_path / f"{model_id}_{model_epoch:04d}.keras"

        data_processor: DataProcessorType = self._create_data_processor_instance(model_artifacts_path=artifacts_path)
        data_processor.load_artifacts()  # Ensure artifacts are loaded for is_prepared check

        if not data_processor.is_prepared:
            raise FileNotFoundError(f"Could not load required artifacts (scaler/thresholds) from: {artifacts_path}")

        model_core: ModelCoreType = self._create_model_core_instance(model_file_path=model_file_path)
        return data_processor, model_core, artifacts_path

    @override
    def _prepare_test_data(
        self, data_processor: DataProcessorType, config: BoxOfficeEvaluationConfigType
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """
        Loads and processes data to retrieve the evaluation set.

        This method is now generic and uses the _get_data_source helper.

        :param data_processor: The initialized data processor.
        :param config: The configuration object for the evaluation run.
        :return: A tuple containing the evaluation features (x_eval) and labels (y_eval).
        :raises ValueError: If reproducibility mode is selected but split parameters are missing.
        """
        self.logger.debug("Loading and processing evaluation dataset...")

        # Use the helper method to get the correct data source type
        data_source: BoxOfficeDataSource = self._get_data_source(dataset_name=config.dataset_name)

        processing_config: BoxOfficeDataConfig = BoxOfficeDataConfig(
            training_week_len=config.training_week_len,
            split_ratios=config.split_ratios,
            random_state=config.random_state
        )

        raw_data: list[MovieSessionData] = data_processor.load_raw_data(source=data_source, config=processing_config)

        if config.evaluate_on_full_dataset:
            self.logger.debug("Evaluation mode: Processing the full dataset as the test set.")
            x_eval, y_eval = data_processor.process_for_evaluation(
                raw_data=raw_data, config=processing_config
            )
            return x_eval, y_eval
        else:
            self.logger.debug("Evaluation mode: Reproducing the original test split.")

            if config.split_ratios is None or config.random_state is None:
                raise ValueError(
                    "For reproducibility mode (evaluate_on_full_dataset=False), "
                    "'split_ratios' and 'random_state' must be provided in the configuration."
                )

            # The type hint for processed_data needs to be flexible enough for both regression and classification
            # For now, I'll use Any, but ideally, this would be a generic type from BaseDataProcessor
            processed_data: Any = data_processor.process_for_training(
                raw_data=raw_data, config=processing_config
            )
            return processed_data['x_test'], processed_data['y_test']

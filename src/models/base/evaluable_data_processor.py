from abc import abstractmethod
from typing import Generic, Optional

from numpy.typing import NDArray

from src.models.base.base_data_processor import (
    BaseDataProcessor,
    RawDataSourceType,
    RawDataType,
    ProcessedTrainingDataType,
    PredictionDataType,
    ProcessedPredictionDataType,
    DataConfigType
)


class EvaluableDataProcessor(
    BaseDataProcessor[
        RawDataSourceType,
        RawDataType,
        ProcessedTrainingDataType,
        PredictionDataType,
        ProcessedPredictionDataType,
        DataConfigType
    ],
    Generic[
        RawDataSourceType,
        RawDataType,
        ProcessedTrainingDataType,
        PredictionDataType,
        ProcessedPredictionDataType,
        DataConfigType
    ]
):
    """
    An abstract base class for data processors that support evaluation on a full, unsplit dataset.

    This class extends the BaseDataProcessor by adding an abstract method
    `process_for_evaluation`, establishing a contract for processors used
    in evaluation contexts that require handling of a complete dataset as a
    single test set.
    """

    @abstractmethod
    def process_for_evaluation(self, raw_data: RawDataType, config: Optional[DataConfigType]) \
            -> tuple[NDArray[any], NDArray[any]]:
        """
        Processes a full raw dataset for evaluation without splitting it.


        :param raw_data: The raw data to be processed for evaluation.
        :param config: An optional configuration object containing necessary parameters.
        :returns: A tuple containing the full processed features (x) and labels (y).
        """
        pass

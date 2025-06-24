from enum import Enum


class ProjectModelType(Enum):
    """
    Enum representing the types of machine learning models in the project.
    """
    PREDICTION = "box_office_prediction"
    SENTIMENT = "review_sentiment_analysis"


class ProjectDatasetType(Enum):
    """
    Enum representing the types of datasets in the project.
    """
    STRUCTURED = "structured"
    FEATURE = "feature"

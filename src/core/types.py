from enum import Enum


class ProjectModelType(Enum):
    """
    Represents the types of machine learning models in the project.

    This enumeration provides a standardized way to reference different model
    categories, ensuring consistency when creating paths, loading configurations,
    or routing logic.

    :ivar PREDICTION: Corresponds to models focused on box office revenue prediction.
    :ivar SENTIMENT: Corresponds to models focused on sentiment analysis of movie reviews.
    """
    PREDICTION = "box_office_prediction"
    SENTIMENT = "review_sentiment_analysis"


class ProjectDatasetType(Enum):
    """
    Represents the types of datasets used within the project.

    This enumeration helps differentiate between datasets at various stages of
    the data processing pipeline, such as raw structured data versus
    feature-engineered data.

    :ivar STRUCTURED: Refers to datasets that are in a structured, often tabular,
                      format but have not yet undergone feature engineering.
    :ivar FEATURE: Refers to datasets that have been processed and contain
                   engineered features ready for model consumption.
    """
    STRUCTURED = "structured"
    FEATURE = "feature"

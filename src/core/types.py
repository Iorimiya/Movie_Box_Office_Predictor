from enum import Enum


class ProjectModelType(Enum):
    """
    Represents the types of machine learning models in the project.

    This enumeration provides a standardized way to reference different model
    categories, ensuring consistency when creating paths, loading configurations,
    or routing logic.

    :ivar BOX_OFFICE_REGRESSION: Corresponds to models focused on box office prediction using regression method.
    :ivar BOX_OFFICE_CLASSIFICATION: Corresponds to models focused on box office prediction using classification method.
    """
    BOX_OFFICE_REGRESSION = "box_office_regression"
    BOX_OFFICE_CLASSIFICATION = "box_office_classification"


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


class DataSourceType(Enum):
    """
    Enumeration for different types of data sources used across the project.

    This helps in selecting between different dataset implementations like
    Database-backed or YAML-backed storage.

    :ivar DATABASE: Indicates data should be sourced from a SQL database.
    :ivar YAML: Indicates data should be sourced from local YAML files.
    """
    DATABASE = "database"
    YAML = "yaml"

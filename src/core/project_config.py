from enum import Enum
from pathlib import Path
from typing import Final


class ProjectModelType(Enum):
    """
    Enum representing the types of machine learning models in the project.
    """
    PREDICTION = "box_office_prediction"
    SENTIMENT = "review_sentiment_analysis"


class ProjectDatasetType(Enum):
    STRUCTURED = "structured"
    FEATURE = "feature"


class ProjectPaths:
    """
    A utility class providing centralized, static access to project paths.

    This class defines all essential project directories and files as class
    variables, allowing them to be accessed globally without needing to
    instantiate the class. It also provides class methods for constructing
    dynamic paths and for ensuring the directory structure exists.
    """


    @staticmethod
    def _find_project_root() -> Path:
        """
        Attempts to find the project root directory by searching upwards from the
        current file's location. It looks for a directory containing key project
        markers such as '.git' or 'src'.

        :raises FileNotFoundError: If the project root cannot be found within a reasonable search depth.
        :returns: The Path object representing the project's root directory.
        """

        current_dir: Path = Path(__file__).resolve().parent
        project_markers: list[str] = [".git", "src"]

        for _ in range(5):
            if any((current_dir / marker).exists() for marker in project_markers):
                return current_dir
            if current_dir.parent == current_dir:
                break
            current_dir = current_dir.parent

        raise FileNotFoundError(
            "Project root not found. Please ensure that this script is run from "
            "within the project directory or a subdirectory, and that the project "
            "root contains one of the following markers: " + ", ".join(project_markers)
        )

    project_root: Final[Path] = _find_project_root()

    input_dir: Final[Path] = project_root / "inputs"
    datasets_dir: Final[Path] = project_root / "datasets"
    logs_dir: Final[Path] = project_root / "logs"
    models_dir: Final[Path] = project_root / "models"
    temp_dir: Final[Path] = project_root / "temp"

    raw_index_sources_dir: Final[Path] = input_dir / "raw_index_sources"
    sentiment_analysis_resources_dir: Final[Path] = input_dir / "sentiment_analysis_resources"

    structured_datasets_dir: Final[Path] = datasets_dir / "structured"
    feature_datasets_dir: Final[Path] = datasets_dir / "feature"

    box_office_prediction_models_root: Final[Path] = models_dir / "box_office_prediction"
    review_sentiment_analysis_models_root: Final[Path] = models_dir / "review_sentiment_analysis"

    @classmethod
    def initialize_directories(cls) -> None:
        """
        Ensures that all essential project directories exist, creating them if necessary.
        """

        cls.input_dir.mkdir(parents=True, exist_ok=True)
        cls.datasets_dir.mkdir(parents=True, exist_ok=True)
        cls.logs_dir.mkdir(parents=True, exist_ok=True)
        cls.models_dir.mkdir(parents=True, exist_ok=True)
        cls.temp_dir.mkdir(parents=True, exist_ok=True)

        cls.raw_index_sources_dir.mkdir(parents=False, exist_ok=True)
        cls.sentiment_analysis_resources_dir.mkdir(parents=False, exist_ok=True)
        cls.structured_datasets_dir.mkdir(parents=False, exist_ok=True)
        cls.feature_datasets_dir.mkdir(parents=False, exist_ok=True)
        cls.box_office_prediction_models_root.mkdir(parents=False, exist_ok=True)
        cls.review_sentiment_analysis_models_root.mkdir(parents=False, exist_ok=True)

    @classmethod
    def get_dataset_path(cls, dataset_name: str, dataset_type: ProjectDatasetType) -> Path:
        match dataset_type:
            case ProjectDatasetType.STRUCTURED:
                return cls.structured_datasets_dir / dataset_name
            case ProjectDatasetType.FEATURE:
                return cls.feature_datasets_dir / dataset_name
            case _:
                raise ValueError(f"Unknown dataset_type: '{dataset_type}'. Must be a member of ProjectDatasetType.")

    @classmethod
    def get_model_root_path(cls, model_id: str, model_type: ProjectModelType) -> Path:
        match model_type:
            case ProjectModelType.PREDICTION:
                return cls.box_office_prediction_models_root / model_id
            case ProjectModelType.SENTIMENT:
                return cls.review_sentiment_analysis_models_root / model_id
            case _:
                raise ValueError(f"Unknown model_type: '{model_type}'. Must be a member of ProjectModelType.")

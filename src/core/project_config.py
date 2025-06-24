from pathlib import Path
from typing import Final

from src.core.types import ProjectDatasetType, ProjectModelType


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
        project_markers: Final[list[str]] = [".git", "src"]
        max_search_depth: Final[int] = 5

        for _ in range(max_search_depth):
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

    box_office_prediction_models_root: Final[Path] = models_dir / ProjectModelType.PREDICTION.value
    review_sentiment_analysis_models_root: Final[Path] = models_dir / ProjectModelType.SENTIMENT.value

    @classmethod
    def initialize_directories(cls) -> None:
        """
        Ensures that all essential project directories exist, creating them if necessary.
        """
        for dir_path in [
            cls.input_dir, cls.datasets_dir, cls.logs_dir, cls.models_dir, cls.temp_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        for dir_path in [
            cls.raw_index_sources_dir, cls.sentiment_analysis_resources_dir,
            cls.structured_datasets_dir, cls.feature_datasets_dir,
            cls.box_office_prediction_models_root, cls.review_sentiment_analysis_models_root
        ]:
            dir_path.mkdir(parents=False, exist_ok=True)

    @classmethod
    def get_dataset_path(cls, dataset_name: str, dataset_type: ProjectDatasetType) -> Path:
        """
        Constructs the full path for a given dataset.
        """
        match dataset_type:
            case ProjectDatasetType.STRUCTURED:
                return cls.structured_datasets_dir / dataset_name
            case ProjectDatasetType.FEATURE:
                return cls.feature_datasets_dir / dataset_name
            case _:
                raise ValueError(f"Unknown dataset_type: '{dataset_type}'. Must be a member of ProjectDatasetType.")

    @classmethod
    def get_model_root_path(cls, model_id: str, model_type: ProjectModelType) -> Path:
        """
        Constructs the root path for a specific model instance.
        """
        match model_type:
            case ProjectModelType.PREDICTION:
                return cls.box_office_prediction_models_root / model_id
            case ProjectModelType.SENTIMENT:
                return cls.review_sentiment_analysis_models_root / model_id
            case _:
                raise ValueError(f"Unknown model_type: '{model_type}'. Must be a member of ProjectModelType.")

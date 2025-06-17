from pathlib import Path
from typing import Final, Literal, Optional


class ProjectConfig:
    """
    Centralized configuration class for managing all project paths.

    This class provides structured access to all essential directories and files
    within the project, including inputs, datasets, models, logs, and
    temporary files. All path attributes are initialized as `pathlib.Path`
    objects and are intended to be read-only after instantiation.

    :ivar project_root: The root directory of the project.
    :ivar input_dir: Base directory for all raw input data.
    :ivar datasets_dir: Base directory for all processed datasets.
    :ivar logs_dir: Base directory for general application logs.
    :ivar models_dir: Base directory for all trained machine learning models.
    :ivar temp_dir: Base directory for temporary files.
    :ivar raw_index_sources_dir: Directory for raw data used to create the index, located within `input_dir`.
    :ivar sentiment_analysis_resources_dir: Directory for raw sentiment analysis training data, located within `input_dir`.
    :ivar box_office_prediction_datasets_root: Root directory for processed box office prediction datasets,
                                               located within `datasets_dir`.
    :ivar review_sentiment_analysis_datasets_root: Root directory for processed review sentiment analysis datasets,
                                                   located within `datasets_dir`.
    :ivar box_office_prediction_models_root: Root directory for box office prediction models, located within `models_dir`.
    :ivar review_sentiment_analysis_models_root: Root directory for review sentiment analysis models,
                                                 located within `models_dir`.
    """

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initializes the ProjectConfig with the project's root directory.

        If `project_root` is not provided, it will automatically attempt to find
        the project root by searching upwards from the current file's location.

        :param project_root: The root directory of the project. If None, it will be automatically detected.
        """

        self.project_root: Final[Path] = project_root if project_root else self._find_project_root()

        self.input_dir: Final[Path] = self.project_root / "inputs"
        self.datasets_dir: Final[Path] = self.project_root / "datasets"
        self.logs_dir: Final[Path] = self.project_root / "logs"
        self.models_dir: Final[Path] = self.project_root / "models"
        self.temp_dir: Final[Path] = self.project_root / "temp"

        self.raw_index_sources_dir: Final[Path] = self.input_dir / "raw_index_sources"
        self.sentiment_analysis_resources_dir: Final[Path] = self.input_dir / "sentiment_analysis_resources"

        self.box_office_prediction_datasets_root: Final[Path] = self.datasets_dir / "box_office_prediction"
        self.review_sentiment_analysis_datasets_root: Final[Path] = self.datasets_dir / "review_sentiment_analysis"

        self.box_office_prediction_models_root: Final[Path] = self.models_dir / "box_office_prediction"
        self.review_sentiment_analysis_models_root: Final[Path] = self.models_dir / "review_sentiment_analysis"

        self._ensure_directories_exist()

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

    def _ensure_directories_exist(self) -> None:
        """
        Ensures that all essential project directories exist, creating them if necessary.
        """

        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.raw_index_sources_dir.mkdir(parents=False, exist_ok=True)
        self.sentiment_analysis_resources_dir.mkdir(parents=False, exist_ok=True)
        self.box_office_prediction_datasets_root.mkdir(parents=False, exist_ok=True)
        self.review_sentiment_analysis_datasets_root.mkdir(parents=False, exist_ok=True)
        self.box_office_prediction_models_root.mkdir(parents=False, exist_ok=True)
        self.review_sentiment_analysis_models_root.mkdir(parents=False, exist_ok=True)

    def get_processed_box_office_dataset_path(self, dataset_name: str) -> Path:
        """
        Constructs and returns the path for a specific processed box office prediction dataset.

        :param dataset_name: The name of the processed dataset (e.g., 'cleaned_data_2024').
        :returns: The Path object for the specified processed box office dataset.
        """
        return self.box_office_prediction_datasets_root / dataset_name

    def get_model_root_path(self, model_type: Literal["box_office_prediction", "review_sentiment_analysis"],
                            model_instance_name: str) -> Path:
        """
        Constructs and returns the root path for a specific model instance.

        :param model_type: The type of the machine learning model.
        :param model_instance_name: The specific instance or version name of the model (e.g., 'bert_v1', 'lstm_final').
        :returns: The Path object for the model instance's root directory.
        :raises ValueError: If the provided `model_type` is not recognized.
        """
        if model_type == "box_office_prediction":
            return self.box_office_prediction_models_root / model_instance_name
        elif model_type == "review_sentiment_analysis":
            return self.review_sentiment_analysis_models_root / model_instance_name
        else:

            valid_types = ["box_office_prediction", "review_sentiment_analysis"]
            raise ValueError(f"Unknown model_type: '{model_type}'. Must be one of {valid_types}.")

# TODO: create ProjectPaths class，將所有的變數設為類別變數，所有的function除staticmethod外皆設為classmethod

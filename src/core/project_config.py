from pathlib import Path
from typing import Optional, Literal, Final # 導入 Final

class ProjectConfig:
    """
    Centralized configuration class for managing all project paths.

    This class provides structured access to all essential directories and files
    within the project, including inputs, datasets, models, logs, source code,
    and temporary files. All path attributes are marked as Final, indicating
    they should not be reassigned after initialization.

    :ivar project_root: The root directory of the project.
    :ivar input_dir: Base directory for all raw input data.
    :ivar datasets_dir: Base directory for all processed datasets.
    :ivar logs_dir: Base directory for general application logs.
    :ivar models_dir: Base directory for all trained machine learning models.
    :ivar temp_dir: Base directory for temporary files.

    # Specific input subdirectories
    :ivar raw_index_sources_dir: Directory for raw data used to create the index.
    :ivar sentiment_analysis_resources_dir: Directory for raw sentiment analysis training data.

    # Specific processed datasets subdirectories
    :ivar box_office_prediction_datasets_root: Root for processed box office prediction datasets.
    :ivar review_sentiment_analysis_datasets_root: Root for processed review sentiment analysis datasets.

    # Specific models subdirectories
    :ivar box_office_prediction_models_root: Root for box office prediction models.
    :ivar review_sentiment_analysis_models_root: Root for review sentiment analysis models.
    """


    def __init__(self, project_root: Optional[Path] = None):
        """
        Initializes the ProjectConfig with the project's root directory.

        If project_root is not provided, it will automatically attempt to find
        the project root by searching upwards from the current file's location.

        :param project_root: The root directory of the project. If None, it will be automatically detected.
        """


        self.project_root: Final[Path] = project_root if project_root is None else self._find_project_root()
        self.input_dir = self.project_root / "inputs"
        self.datasets_dir = self.project_root / "datasets"
        self.logs_dir = self.project_root / "logs"
        self.models_dir = self.project_root / "models"
        self.temp_dir = self.project_root / "temp"

        # Initialize specific input/dataset/model directories
        self.raw_index_sources_dir = self.input_dir / "raw_index_sources"
        self.sentiment_analysis_resources_dir = self.input_dir / "sentiment_analysis_resources"

        self.box_office_prediction_datasets_root = self.datasets_dir / "box_office_prediction"
        self.review_sentiment_analysis_datasets_root = self.datasets_dir / "review_sentiment_analysis"

        self.box_office_prediction_models_root = self.models_dir / "box_office_prediction"
        self.review_sentiment_analysis_models_root = self.models_dir / "review_sentiment_analysis"

        # Ensure all necessary directories exist
        self._ensure_directories_exist()

    @staticmethod
    def _find_project_root() -> Path:
        """
        Attempts to find the project root directory by searching upwards from the
        current file's location. It looks for a directory containing key project
        markers like '.git', 'src', 'models', 'datasets'.

        :raises FileNotFoundError: If the project root cannot be found within a reasonable depth.
        :returns: The Path object representing the project's root directory.
        """
        # Start from the directory where this project_config.py file is located (src/config/)
        current_dir: Path = Path(__file__).resolve().parent

        # Define common project root markers
        # Using '.git' is generally the most reliable for Git-managed projects
        # You can add 'src', 'models', 'datasets' if '.git' isn't always present or reliable
        project_markers: list[str] = [".git", "src"]

        # Search upwards for the project root, limiting the search depth to prevent infinite loops
        for _ in range(5):  # Search up to 5 levels (e.g., src/config -> src -> project_root)
            if any((current_dir / marker).exists() for marker in project_markers):
                return current_dir
            if current_dir.parent == current_dir:  # Reached filesystem root
                break
            current_dir = current_dir.parent

        raise FileNotFoundError(
            "Project root not found. Please ensure that this script is run from "
            "within the project directory or a subdirectory, and that the project "
            "root contains one of the following markers: " + ", ".join(project_markers)
        )

    def _ensure_directories_exist(self) -> None:
        """
        Ensures that all essential directories for the project exist.
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

    # --- Dataset Specific Methods ---
    def get_processed_box_office_dataset_path(self, dataset_name: str) -> Path:
        """
        Returns the root path for a specific processed box office prediction dataset.

        :param dataset_name: The name of the processed dataset (e.g., 'dataset_2024').
        :returns: The Path object for the specified processed box office dataset.
        """
        return self.box_office_prediction_datasets_root / dataset_name


    def get_model_root_path(self, model_type: Literal["box_office_prediction", "review_sentiment_analysis"], model_instance_name: str) -> Path:
        """
        Returns the root path for a specific model instance (e.g., 'model1', 'model2').

        :param model_type: The type of the machine learning model.
        :param model_instance_name: The specific instance or version name of the model (e.g., 'model1', 'model2').
        :returns: The Path object for the model instance's root directory.
        """
        if model_type == "box_office_prediction":
            return self.box_office_prediction_models_root / model_instance_name
        elif model_type == "review_sentiment_analysis":
            return self.review_sentiment_analysis_models_root / model_instance_name
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
from pathlib import Path
from typing import Final

from yaml import YAMLError

from src.core.types import ProjectDatasetType, ProjectModelType
from src.data_handling.database_client import DatabaseConfig
from src.data_handling.file_io import YamlFile


class ProjectPaths:
    """
    Provides centralized, static access to project-wide paths and constants.

    This utility class defines essential project directories (e.g., `datasets_dir`,
    `models_dir`) and file-related constants as class variables. This allows for
    consistent and easy access to file system locations throughout the application
    without needing to instantiate the class. It also includes helper methods for
    constructing dynamic paths and initializing the directory structure.

    :ivar project_root: The root directory of the project.
    :ivar input_dir: The directory for input data and resources.
    :ivar datasets_dir: The root directory for all datasets.
    :ivar logs_dir: The directory for log files.
    :ivar models_dir: The root directory for saved models.
    :ivar temp_dir: The directory for temporary files.
    :ivar configs_dir: The directory for configuration files.
    :ivar raw_index_sources_dir: The directory for raw CSV files used to create dataset indexes.
    :ivar structured_datasets_dir: The directory for structured datasets, organized by name.
    :ivar feature_datasets_dir: The directory for feature-engineered datasets.
    :ivar box_office_regression_models_root: The root directory for Box Office Regression Models.
    :ivar BOX_OFFICE_SUBFOLDER_NAME: The standard subfolder name for box office data.
    :ivar PUBLIC_REVIEWS_SUBFOLDER_NAME: The standard subfolder name for public review data.
    :ivar EXPERT_REVIEWS_SUBFOLDER_NAME: The standard subfolder name for expert review data.
    :ivar INDEX_FILE_NAME: The standard name for the index file within a structured dataset.
    """

    @staticmethod
    def _find_project_root() -> Path:
        """
        Finds the project root directory by searching upwards from this file's location.

        The method identifies the root by looking for a directory containing key
        project markers such as '.git' or 'src'.

        :raises FileNotFoundError: If the project root cannot be located within a
                                  reasonable search depth.
        :return: The path to the project's root directory.
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
    configs_dir: Final[Path] = project_root / "configs"

    raw_index_sources_dir: Final[Path] = input_dir / "raw_index_sources"

    structured_datasets_dir: Final[Path] = datasets_dir / "structured"
    feature_datasets_dir: Final[Path] = datasets_dir / "feature"

    box_office_regression_models_root: Final[Path] = models_dir / ProjectModelType.BOX_OFFICE_REGRESSION.value
    box_office_classification_models_root: Final[Path] = models_dir / ProjectModelType.BOX_OFFICE_CLASSIFICATION.value

    BOX_OFFICE_SUBFOLDER_NAME: Final[str] = "box_office"
    PUBLIC_REVIEWS_SUBFOLDER_NAME: Final[str] = "public_reviews"
    EXPERT_REVIEWS_SUBFOLDER_NAME: Final[str] = "expert_reviews"
    INDEX_FILE_NAME: Final[str] = "index.csv"

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
            cls.raw_index_sources_dir,
            cls.box_office_regression_models_root
        ]:
            dir_path.mkdir(parents=False, exist_ok=True)

    @classmethod
    def get_yaml_dataset_path(cls, dataset_name: str, dataset_type: ProjectDatasetType) -> Path:
        """
        Constructs the full path for a given dataset directory.

        :param dataset_name: The name of the dataset directory.
        :param dataset_type: The type of the dataset, which determines its base location
                             (e.g., structured or feature).
        :raises ValueError: If an unknown `dataset_type` is provided.
        :return: The full path to the specified dataset directory.
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
        Constructs the root directory path for a specific model instance.

        All artifacts for a given model (e.g., checkpoints, configs, plots)
        are stored within this directory.

        :param model_id: The unique identifier for the model.
        :param model_type: The type of the model (e.g., BOX_OFFICE_REGRESSION).
        :raises ValueError: If an unknown `model_type` is provided.
        :return: The full path to the specified model's root directory.
        """
        match model_type:
            case ProjectModelType.BOX_OFFICE_REGRESSION:
                return cls.box_office_regression_models_root / model_id
            case ProjectModelType.BOX_OFFICE_CLASSIFICATION:
                return cls.box_office_classification_models_root / model_id
            case _:
                raise ValueError(f"Unknown model_type: '{model_type}'. Must be a member of ProjectModelType.")

    @classmethod
    def get_config_path(cls, config_name: str) -> Path:
        """
        Constructs the full path for a given configuration file.

        :param config_name: The name of the configuration file (e.g., "box_office_regression_defaults.yaml").
        :return: The full path to the configuration file within the project's config directory.
        """
        return cls.configs_dir / config_name

    @classmethod
    def get_model_plots_path(cls, model_id: str, model_type: ProjectModelType) -> Path:
        """
        Constructs the path for a model's evaluation plots directory.

        This path is located within the model's root artifact directory,
        ensuring that all outputs for a model run are co-located.

        :param model_id: The unique identifier of the model.
        :param model_type: The type of the model (e.g., BOX_OFFICE_REGRESSION).
        :raises ValueError: If an unknown `model_type` is provided.
        :return: The full path to the evaluation plots directory for the model.
        """
        model_root: Path = cls.get_model_root_path(model_id=model_id, model_type=model_type)
        return model_root / "evaluation_plots"

    @classmethod
    def get_db_initial_schema_path(cls) -> Path:
        return cls.project_root / 'src' / 'data_handling' / 'sql' / 'init_schema.sql'


def _load_database_config(user_type: str = 'user') -> DatabaseConfig:
    """
    Loads the database configuration from the YAML file.

    :param user_type: The type of user to load credentials for ('user' or 'admin').
    :return: A dictionary containing database connection details.
    :raises FileNotFoundError: If the 'database_defaults.yaml' file cannot be found.
    :raises TypeError: If the root of the YAML file is not a dictionary.
    :raises yaml.YAMLError: If the YAML file is malformed.
    """
    config_path = ProjectPaths.get_config_path('database_defaults.yaml')

    if not config_path.exists():
        raise FileNotFoundError(
            f"Database configuration file not found at: {config_path}"
        )

    try:
        yaml_file = YamlFile(path=config_path)
        config_data = yaml_file.load_single_document()

        if not isinstance(config_data, dict):
            raise TypeError(
                f"Expected database configuration to be a dictionary, "
                f"but got {type(config_data).__name__} in {config_path}"
            )

        server_config = config_data.get('server', {})
        user_config = config_data.get(user_type, {})

        # 這裡可以加入更嚴格的檢查，例如確認 'name' 和 'password' 是否存在
        if 'name' not in user_config or 'password' not in user_config:
            raise KeyError(f"Missing 'name' or 'password' for user type '{user_type}' in {config_path}")

        return {
            'address': server_config.get('address', 'localhost'),
            'port': str(server_config.get('port', 3306)),
            'user': user_config['name'],
            'password': user_config['password']
        }
    except (YAMLError, TypeError, KeyError) as e:
        # 重新拋出錯誤，並附加上下文
        raise type(e)(f"Failed to load database configuration from {config_path}: {e}") from e


class ProjectConfig:
    """
    Provides centralized, static access to project-wide configurations.
    """
    # Database Configuration
    # Reads from configs/database_defaults.yaml
    DEFAULT_DATABASE_CONFIG: Final[DatabaseConfig] = _load_database_config(user_type='user')
    ROOT_DATABASE_CONFIG: Final[DatabaseConfig] = _load_database_config(user_type='admin')

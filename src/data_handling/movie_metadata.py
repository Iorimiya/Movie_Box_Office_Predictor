from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Optional, TypedDict

from src.core.logging_manager import LoggingManager


class MovieMetadataRawData(TypedDict, total=False):
    """
    Represents the raw data structure for movie metadata.

    All fields in this TypedDict are optional due to `total=False`, meaning
    they may not be present in the raw data.

    :ivar id: The raw identifier of the movie, typically a string.
    :ivar name: The raw name of the movie.
    """
    id: str
    name: str


@dataclass(kw_only=True)
class MovieMetadata:
    """
    Represents structured movie metadata.

    :ivar id: The unique integer identifier of the movie.
    :ivar name: The name of the movie.
    """
    id: int
    name: str

    @classmethod
    def from_csv_raw_data(cls, source: MovieMetadataRawData) -> Optional['MovieMetadata']:
        """
        Creates a MovieMetadata instance from raw dictionary data.

        This method attempts to parse and validate the raw data. If essential
        fields like 'id' or 'name' are missing, empty, or if 'id' cannot be
        converted to an integer, a warning is logged, and None is returned.

        :param cls: The class itself.
        :param source: A dictionary containing the raw movie metadata.
        :return: An instance of MovieMetadata if parsing is successful, otherwise None.
        """
        logger: Logger = LoggingManager().get_logger('root')
        raw_id: Optional[str] = source.get('id')
        raw_name: Optional[str] = source.get('name')

        if not raw_id:  # 檢查 None 或空字串
            logger.warning(f"Movie 'id' is missing or empty in raw data: {source}. Skipping.")
            return None
        if not raw_name:  # 檢查 None 或空字串
            logger.warning(
                f"Movie 'name' is missing or empty for potential id '{raw_id}': {source}. Skipping.")
            return None

        try:
            movie_id: int = int(raw_id)  # 將 id 從 str 轉換為 int
        except ValueError:
            logger.warning(f"Invalid movie 'id' format '{raw_id}' in raw data: {source}. Skipping.")
            return None

        try:
            return cls(id=movie_id, name=raw_name)
        except Exception as e:  # 捕獲路徑構建或 dataclass 初始化時的潛在錯誤
            logger.error(f"Error creating MovieMetadata for id '{raw_id}': {e}. Raw data: {source}")
            return None


@dataclass(kw_only=True)
class MoviePathMetadata(MovieMetadata):
    """
    Represents movie metadata including paths to associated data files.

    Inherits 'id' and 'name' from :class:`~.MovieMetadata`.

    :ivar box_office_file_path: Path to the movie's box office data file.
    :ivar public_reviews_file_path: Path to the movie's public reviews data file.
    :ivar expert_reviews_file_path: Path to the movie's expert reviews data file.
    """
    box_office_file_path: Path
    public_reviews_file_path: Path
    expert_reviews_file_path: Path

    @classmethod
    def from_metadata(cls, source: MovieMetadata, dataset_root_path: Path) -> 'MoviePathMetadata':
        """
        Creates a MoviePathMetadata instance from a MovieMetadata instance and a root path.

        This method constructs file paths for box office and review data based on
        the movie's ID and the provided dataset root path.

        :param cls: The class itself.
        :param source: An instance of MovieMetadata containing the base movie data.
        :param dataset_root_path: The root directory path for the dataset.
        :return: An instance of MoviePathMetadata with populated file paths.
        """
        return cls(id=source.id,
                   name=source.name,
                   box_office_file_path=dataset_root_path / 'box_office' / f"{source.id}.yaml",
                   public_reviews_file_path=dataset_root_path / 'public_reviews' / f"{source.id}.yaml",
                   expert_reviews_file_path=dataset_root_path / 'expert_reviews' / f"{source.id}.yaml")

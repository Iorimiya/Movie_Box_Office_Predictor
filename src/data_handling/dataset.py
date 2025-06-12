from dataclasses import dataclass
from functools import cached_property
from logging import Logger
from pathlib import Path
from typing import cast, Literal

from src.core.logging_manager import LoggingManager
from src.core.project_config import ProjectConfig
from src.data_handling.box_office import BoxOffice
from src.data_handling.file_io import CsvFile
from src.data_handling.movie_collections import MovieData
from src.data_handling.movie_metadata import MovieMetadata, MovieMetadataRawData, MoviePathMetadata
from src.data_handling.reviews import PublicReview, ExpertReview


@dataclass(kw_only=True)
class Dataset:
    name: str

    @property
    def dataset_path(self) -> Path:
        return ProjectConfig().get_processed_box_office_dataset_path(self.name)

    @property
    def index_file_path(self) -> Path:
        return self.dataset_path / 'index.csv'

    @property
    def index_file(self) -> CsvFile:
        return CsvFile(path=self.index_file_path)

    @cached_property
    def movies_metadata(self) -> list[MovieMetadata]:
        logger: Logger = LoggingManager().get_logger('root')
        logger.info(
            f"Attempting to create MovieSourceInfo objects for dataset '{self.name}' from index file: '{self.index_file_path}'.")

        raw_movie_data_from_csv: list[dict[str, str]]
        try:
            # 在嘗試載入前，明確檢查索引檔案是否存在
            if not self.index_file_path.exists():
                logger.error(f"Index file not found: '{self.index_file_path}' for dataset '{self.name}'.")
                return []

            raw_movie_data_from_csv = self.index_file.load()  # CsvFile.load() 返回 list[dict[str, str]]

            if not raw_movie_data_from_csv:  # 檢查檔案是否為空或沒有返回任何資料
                logger.info(
                    f"No movie data found or index file is empty: '{self.index_file_path}' for dataset '{self.name}'.")
                return []
        except FileNotFoundError:
            # 這個 catch 主要是為了 CsvFile 初始化或 load 內部如果也拋出 FileNotFoundError 的情況
            # 但由於上面已經有 self.index_file_path.exists() 檢查，此處被觸發的機率較低
            # 如果被觸發，錯誤已經在上面記錄過了
            return []
        except Exception as e:  # 捕獲 CsvFile.load() 可能引發的其他錯誤
            logger.error(f"Error loading index file '{self.index_file_path}' for dataset '{self.name}': {e}")
            return []

        return [
            movie_metadata for raw_movie_data in raw_movie_data_from_csv
            if (movie_metadata := MovieMetadata.from_csv_raw_data(
                source=cast(MovieMetadataRawData, raw_movie_data)
            )) is not None
        ]

    def load_movie_source_info(self) -> list[MoviePathMetadata]:
        """
        從 Dataset 的 index_file 載入並創建 MovieSourceInfo 物件列表。

        此方法會讀取資料集的索引檔案（通常是 index.csv），
        對於檔案中的每一行（代表一部電影的原始資訊），
        它會嘗試使用 MovieSourceInfo.from_csv_source 工廠方法來創建一個
        MovieSourceInfo 物件。這個物件包含了電影的 ID、名稱，以及
        根據資料集路徑和電影 ID 構建的完整檔案路徑（票房、公開評論、專家評論）。

        如果索引檔案不存在、為空，或在載入過程中發生錯誤，將返回空列表。
        對於索引檔案中無效的單行資料（例如，缺少 ID 或名稱），
        MovieSourceInfo.from_csv_source 會記錄警告並跳過該行。

        :returns: 一個包含成功創建的 MovieSourceInfo 物件的列表。
                  如果發生錯誤或沒有有效的資料，則返回空列表。
        """
        return [MoviePathMetadata.from_metadata(source=movie_metadata, dataset_root_path=self.dataset_path)
                for movie_metadata in self.movies_metadata]

    def load_all_movie_data(self,mode:Literal['ALL','META'])-> list[MovieData]:
        logger: Logger = LoggingManager().get_logger("root")
        if mode == 'ALL':
            source_infos: list[MoviePathMetadata] = self.load_movie_source_info()
            if not source_infos:
                logger.info(f"No processable movie metadata after initial validation from '{self.index_file_path}'.")
                return []

            return [
                MovieData(id=movie_meta_info.id,
                    name=movie_meta_info.name,
                    box_office=BoxOffice.create_multiple(source=movie_meta_info.box_office_file_path),
                    public_reviews=PublicReview.create_multiple(source=movie_meta_info.public_reviews_file_path),
                    expert_reviews=ExpertReview.create_multiple(source=movie_meta_info.expert_reviews_file_path)
                    ) for movie_meta_info in source_infos
            ]
        elif mode == 'Meta':
            movies_meta: list[MovieMetadata] = self.movies_metadata
            if not movies_meta:
                logger.info(f"No processable movie metadata after initial validation from '{self.index_file_path}'.")
                return []

            return [MovieData(id=movie_meta.id, name=movie_meta.name, box_office=[], public_reviews=[], expert_reviews=[])
                    for movie_meta in movies_meta]

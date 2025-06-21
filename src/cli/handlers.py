from argparse import ArgumentParser, Namespace
from logging import Logger
from pathlib import Path

from src.core.project_config import ProjectPaths, ProjectDatasetType,ProjectModelType
from src.core.logging_manager import LoggingManager
from src.data_handling.dataset import Dataset

class DatasetHandler:
    """
    處理與資料集相關的 CLI 命令。

    :成員變數:
        - parser (ArgumentParser): 用於錯誤報告的 ArgumentParser 實例。
    """

    def __init__(self, parser: ArgumentParser) -> None:
        self.__logger: Logger = LoggingManager().get_logger()
        self.__parser = parser

    def create_index(self, args: Namespace) -> None:
        """
        處理 'dataset index' 命令。

        :輸入變數:
            - args (Namespace): 解析後的命令列引數。
        """
        self.__logger.info(f"執行: 建立資料集索引，資料集名稱: {args.structured_dataset_name}")
        if Path(args.source_file).exists():
            Dataset(name = args.structured_dataset_name).initialize_index_file(source_csv=args.source_file)
        else:
            raise AttributeError

    def collect_box_office(self, args: Namespace) -> None:
        """
        處理 'dataset collect box-office' 命令。

        :輸入變數:
            - args (Namespace): 解析後的命令列引數。
        """
        if args.structured_dataset_name:
            self.__logger.info(f"執行: 為資料集 '{args.structured_dataset_name}' 蒐集票房資料。")
            structured_dataset_name = args.structured_dataset_name
            if Path(
                ProjectPaths.get_dataset_path(
                    dataset_name=structured_dataset_name,dataset_type=ProjectDatasetType.STRUCTURED
                )
            ).exists():
                Dataset(name=structured_dataset_name).collect_box_office()
            else:
                raise AttributeError
        elif args.movie_name:
            self.__logger.info(f"執行: 為電影 '{args.movie_name}' 蒐集票房資料。")
            # TODO


    def collect_ptt_review(self, args: Namespace) -> None:
        """
        處理 'dataset collect ptt-review' 命令。

        :輸入變數:
            - args (Namespace): 解析後的命令列引數。
        """
        if args.structured_dataset_name:
            self.__logger.info(f"執行: 為資料集 '{args.structured_dataset_name}' 蒐集 PTT 評論。")
            structured_dataset_name = args.structured_dataset_name
            if Path(
                ProjectPaths.get_dataset_path(
                    dataset_name=structured_dataset_name, dataset_type=ProjectDatasetType.STRUCTURED
                )
            ).exists():
                Dataset(name=structured_dataset_name).collect_public_review(target_website='PTT')
            else:
                raise AttributeError
        elif args.movie_name:
            self.__logger.info(f"執行: 為電影 '{args.movie_name}' 蒐集 PTT 評論。")
            # TODO

    def collect_dcard_review(self, args: Namespace) -> None:
        """
        處理 'dataset collect dcard-review' 命令。

        :輸入變數:
            - args (Namespace): 解析後的命令列引數。
        """
        if args.structured_dataset_name:
            self.__logger.info(f"執行: 為資料集 '{args.structured_dataset_name}' 蒐集 Dcard 評論。")
            structured_dataset_name = args.structured_dataset_name
            if Path(
                ProjectPaths.get_dataset_path(
                    dataset_name=structured_dataset_name, dataset_type=ProjectDatasetType.STRUCTURED
                )
            ).exists():
                Dataset(name=structured_dataset_name).collect_public_review(target_website='DCARD')
        elif args.movie_name:
            self.__logger.info(f"執行: 為電影 '{args.movie_name}' 蒐集 Dcard 評論。")
            # TODO

    def compute_sentiment(self, args: Namespace) -> None:
        """
        處理 'dataset compute_sentiment' 命令。

        :輸入變數:
            - args (Namespace): 解析後的命令列引數。
        """
        self.__logger.info(
            f"執行: 計算資料集 '{args.structured_dataset_name}' 的情感分數，使用模型 '{args.model_id}' (epoch: {args.epoch})。")
        structured_dataset_name = args.structured_dataset_name
        if Path(ProjectPaths.get_dataset_path(
            dataset_name=structured_dataset_name, dataset_type=ProjectDatasetType.STRUCTURED
        )).exists() and Path(ProjectPaths.get_model_root_path(
            model_id=args.model_id,model_type=ProjectModelType.PREDICTION
        )).exists():
            Dataset(name=structured_dataset_name).compute_sentiment(model_id=args.model_id,model_epoch=args.epoch)



class SentimentModelHandler:
    """
    處理與情感分數模型相關的 CLI 命令。

    :成員變數:
        - parser (ArgumentParser): 用於錯誤報告的 ArgumentParser 實例。
    """

    def __init__(self, parser: ArgumentParser) -> None:
        self.__logger: Logger = LoggingManager().get_logger()
        self.__parser = parser

    def train(self, args: Namespace) -> None:
        """
        處理 'sentiment-score-model train' 命令。

        :輸入變數:
            - args (Namespace): 解析後的命令列引數。
        """
        source_type = ""
        if args.feature_dataset_name:
            source_type = f"特徵資料集: {args.feature_dataset_name}"
        elif args.structured_dataset_name:
            source_type = f"結構化資料集: {args.structured_dataset_name}"
        elif args.random_data:
            source_type = "隨機生成資料"

        self.__logger.info(f"執行: 訓練情感分數模型 '{args.model_id}'。來源: {source_type}。")
        self.__logger.info(
            f"舊 Epoch: {args.old_epoch if args.old_epoch else '無'}，目標 Epoch: {args.target_epoch}，檢查點間隔: {args.checkpoint_interval if args.checkpoint_interval else '無'}")

    def test(self, args: Namespace) -> None:
        """
        處理 'sentiment-score-model test' 命令。

        :輸入變數:
            - args (Namespace): 解析後的命令列引數。
        """
        self.__logger.info(
            f"執行: 測試情感分數模型 '{args.model_id}' (epoch: {args.epoch})，輸入句子: '{args.input_sentence}'")

    def plot_graph(self, args: Namespace) -> None:
        """
        處理 'sentiment-score-model evaluate plot' 命令。

        :輸入變數:
            - args (Namespace): 解析後的命令列引數。
        :引發錯誤:
            - SystemExit: 如果沒有選擇任何繪圖選項。
        """
        if not (args.training_loss or args.validation_loss or args.f1_score):
            self.__parser.error("繪圖時，請至少選擇 --training-loss, --validation-loss, 或 --f1-score 其中一個旗標。")

        self.__logger.info(f"執行: 繪製情感分數模型 '{args.model_id}' 的評估圖表。")
        if args.training_loss:
            self.__logger.info("  - 繪製訓練損失曲線。")
        if args.validation_loss:
            self.__logger.info("  - 繪製驗證損失曲線。")
        if args.f1_score:
            self.__logger.info("  - 繪製 F1 分數曲線。")

    def get_metrics(self, args: Namespace) -> None:
        """
        處理 'sentiment-score-model evaluate get-metrics' 命令。

        :輸入變數:
            - args (Namespace): 解析後的命令列引數。
        :引發錯誤:
            - SystemExit: 如果沒有選擇任何指標選項。
        """
        if not (args.training_loss or args.validation_loss or args.f1_score):
            self.__parser.error("獲取指標時，請至少選擇 --training-loss, --validation-loss, 或 --f1-score 其中一個旗標。")

        self.__logger.info(f"執行: 獲取情感分數模型 '{args.model_id}' (epoch: {args.epoch}) 的評估指標。")
        if args.training_loss:
            self.__logger.info("  - 獲取訓練損失。")
        if args.validation_loss:
            self.__logger.info("  - 獲取驗證損失。")
        if args.f1_score:
            self.__logger.info("  - 獲取 F1 分數。")


# 類似地，為 PredictionModel 創建一個類別
class PredictionModelHandler:
    """
    處理與預測模型相關的 CLI 命令。

    :成員變數:
        - parser (ArgumentParser): 用於錯誤報告的 ArgumentParser 實例。
    """

    def __init__(self, parser: ArgumentParser) -> None:
        self.__logger: Logger = LoggingManager().get_logger()
        self.__parser = parser

    def train(self, args: Namespace) -> None:
        """
        處理 'prediction-model train' 命令。

        :輸入變數:
            - args (Namespace): 解析後的命令列引數。
        """
        source_type = ""
        if args.feature_dataset_name:
            source_type = f"特徵資料集: {args.feature_dataset_name}"
        elif args.structured_dataset_name:
            source_type = f"結構化資料集: {args.structured_dataset_name}"
        elif args.random_data:
            source_type = "隨機生成資料"

        self.__logger.info(f"執行: 訓練預測模型 '{args.model_id}'。來源: {source_type}。")
        self.__logger.info(
            f"舊 Epoch: {args.old_epoch if args.old_epoch else '無'}，目標 Epoch: {args.target_epoch}，檢查點間隔: {args.checkpoint_interval if args.checkpoint_interval else '無'}")

    def test(self, args: Namespace) -> None:
        """
        處理 'prediction-model test' 命令。

        :輸入變數:
            - args (Namespace): 解析後的命令列引數。
        """
        if args.movie_name:
            self.__logger.info(
                f"執行: 測試預測模型 '{args.model_id}' (epoch: {args.epoch})，針對電影: '{args.movie_name}'。")
        elif args.random:
            self.__logger.info(f"執行: 測試預測模型 '{args.model_id}' (epoch: {args.epoch})，使用隨機資料")

    def plot_graph(self, args: Namespace) -> None:
        """
        處理 'prediction-model evaluate plot' 命令。

        :輸入變數:
            - args (Namespace): 解析後的命令列引數。
        :引發錯誤:
            - SystemExit: 如果沒有選擇任何繪圖選項。
        """
        if not (args.training_loss or args.validation_loss or args.f1_score):
            self.__parser.error("繪圖時，請至少選擇 --training-loss, --validation-loss, 或 --f1-score 其中一個旗標。")

        self.__logger.info(f"執行: 繪製情感分數模型 '{args.model_id}' 的評估圖表。")
        if args.training_loss:
            self.__logger.info("  - 繪製訓練損失曲線。")
        if args.validation_loss:
            self.__logger.info("  - 繪製驗證損失曲線。")
        if args.f1_score:
            self.__logger.info("  - 繪製 F1 分數曲線。")

    def get_metrics(self, args: Namespace) -> None:
        """
        處理 'prediction-model evaluate get-metrics' 命令。

        :輸入變數:
            - args (Namespace): 解析後的命令列引數。
        :引發錯誤:
            - SystemExit: 如果沒有選擇任何指標選項。
        """
        if not (args.training_loss or args.validation_loss or args.f1_score):
            self.__parser.error("獲取指標時，請至少選擇 --training-loss, --validation-loss, 或 --f1-score 其中一個旗標。")

        self.__logger.info(f"執行: 獲取情感分數模型 '{args.model_id}' (epoch: {args.epoch}) 的評估指標。")
        if args.training_loss:
            self.__logger.info("  - 獲取訓練損失。")
        if args.validation_loss:
            self.__logger.info("  - 獲取驗證損失。")
        if args.f1_score:
            self.__logger.info("  - 獲取 F1 分數。")

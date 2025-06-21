from argparse import ArgumentParser, Namespace
from typing import cast, Optional, TypeVar

from src.cli.handlers import DatasetHandler, SentimentModelHandler, PredictionModelHandler

CliParser = TypeVar("CliParser", bound=ArgumentParser)
SubcommandAction = TypeVar("SubcommandAction", bound=ArgumentParser)


class ArgumentParserBuilder:

    def __init__(self) -> None:
        """Initializes the ArgumentParserBuilder."""
        self.parser: CliParser = ArgumentParser(
            prog="movie_predictor",
            description="Movie Box Office Prediction Tool"
        )
        self._subparsers_action: SubcommandAction = cast(SubcommandAction, self.parser.add_subparsers(
            dest="command_group",
            required=True,
            title="Available command groups",
            description="Select a command group to see its specific commands."
        ))
        # Arguments Parent Parser
        self.__structured_dataset_name_args_parser: CliParser = self.__create_structured_dataset_name_args_parser()
        self.__model_id_args_parser: CliParser = self.__create_model_id_args_parser()
        self.__model_file_args_parser: CliParser = self.__create_model_file_args_parser()

        # Common Behavior Parser
        self.__collect_common_behavior_parser: CliParser = self.__create_collect_common_behavior_parser()
        self.__train_common_behavior_parser: CliParser = self.__create_train_common_behavior_parser()
        self.__evaluate_common_behavior_parser: CliParser = self.__create_evaluate_common_behavior_parser()
        self.__plot_common_behavior_parser: CliParser = self.__create_plot_common_behavior_parser()
        self.__get_metrics_common_behavior_parser: CliParser = self.__create_get_metrics_common_behavior_parser()

        # Handler
        self.__dataset_handler = DatasetHandler(self.parser)
        self.__sentiment_model_handler = SentimentModelHandler(self.parser)
        self.__prediction_model_handler = PredictionModelHandler(self.parser)

    @staticmethod
    def __create_structured_dataset_name_args_parser() -> ArgumentParser:
        parser: CliParser = ArgumentParser(add_help=False)
        parser.add_argument("--structured-dataset-name", type=str, required=True, help="")
        return parser

    @staticmethod
    def __create_model_id_args_parser() -> ArgumentParser:
        parser: CliParser = ArgumentParser(add_help=False)
        parser.add_argument('--model-id', type=str, required=True, help="The model ID.")
        return parser

    @staticmethod
    def __create_model_file_args_parser() -> ArgumentParser:
        parser: CliParser = ArgumentParser(add_help=False)
        model_file_group = parser.add_argument_group(
            '模型檔案識別',  # 群組的標題
            description='請提供模型系列名稱與具體訓練輪次以精確定位模型檔案。'  # 群組的描述
        )
        model_file_group.add_argument('--model-id', type=str, required=True, help='The model ID.')
        model_file_group.add_argument('--epoch', type=int, required=True, help='')
        return parser

    @staticmethod
    def __create_collect_common_behavior_parser() -> ArgumentParser:
        parser: CliParser = ArgumentParser(add_help=False)
        target_group = parser.add_mutually_exclusive_group(required=True)
        target_group.add_argument('--structured-dataset-name', type=str, help='為此資料集內的電影蒐集資料。')
        target_group.add_argument('--movie_name', type=str, help='為此電影蒐集資料。')
        return parser

    def __create_train_common_behavior_parser(self) -> ArgumentParser:
        parser: CliParser = ArgumentParser(add_help=False, parents=[self.__model_id_args_parser])
        source_group = parser.add_mutually_exclusive_group(required=True)
        source_group.add_argument('--feature-dataset-name', type=str, help='指定訓練資料集名稱。')
        source_group.add_argument('--structured-dataset-name', type=str, help='指定原始來源資料名稱。')
        source_group.add_argument('--random-data', action='store_true', help='使用隨機生成資料進行訓練。')

        # 通用訓練參數
        parser.add_argument('--old-epoch', type=int, required=False, help='基於舊模型繼續訓練的名稱。')
        parser.add_argument('--target-epoch', type=int, required=True, help='目標訓練 epoch 數量。')
        parser.add_argument('--checkpoint-interval', type=int, required=False, help='儲存模型的 epoch 間隔。')
        return parser

    @staticmethod
    def __create_evaluate_common_behavior_parser() -> ArgumentParser:
        parser: CliParser = ArgumentParser(add_help=False)

        # --- 關鍵點：建立一個 ArgumentGroup 來組織相關旗標 ---
        plot_options_group = parser.add_argument_group(
            '圖表選項',  # Group 的標題
            description='請選擇至少一個要繪製的圖表類型。可多選。'  # Group 的描述
        )
        # 將旗標添加到這個群組中
        plot_options_group.add_argument('--training-loss', action='store_true', help='繪製訓練損失曲線。')
        plot_options_group.add_argument('--validation-loss', action='store_true', help='繪製驗證損失曲線。')
        plot_options_group.add_argument('--f1-score', action='store_true', help='繪製 F1 分數曲線。')
        return parser

    def __create_plot_common_behavior_parser(self) -> ArgumentParser:
        return ArgumentParser(
            add_help=False, parents=[self.__evaluate_common_behavior_parser, self.__model_id_args_parser]
        )

    def __create_get_metrics_common_behavior_parser(self) -> ArgumentParser:
        return ArgumentParser(
            add_help=False, parents=[self.__evaluate_common_behavior_parser, self.__model_file_args_parser]
        )

    def __setup_dataset_subparser(self) -> None:
        dataset_parser: CliParser = self._subparsers_action.add_parser("dataset", help="", )
        dataset_subparsers: SubcommandAction = cast(
            SubcommandAction, dataset_parser.add_subparsers(dest="dataset", required=True, help="")
        )
        index_parser: CliParser = dataset_subparsers.add_parser(
            "index", help="", parents=[self.__structured_dataset_name_args_parser]
        )
        index_parser.add_argument("--source-file", type=str, required=True, help="")
        index_parser.set_defaults(func=self.__dataset_handler.create_index)
        collect_parser: CliParser = dataset_subparsers.add_parser("collect", help="")
        collect_subparsers: SubcommandAction = cast(
            SubcommandAction, collect_parser.add_subparsers(dest="collect", required=True, help="")
        )

        box_office_parser: CliParser = collect_subparsers.add_parser(
            "box-office", help="", parents=[self.__collect_common_behavior_parser]
        )
        box_office_parser.set_defaults(func=self.__dataset_handler.collect_box_office)
        ptt_review_parser: CliParser = collect_subparsers.add_parser(
            "ptt-review", help="", parents=[self.__collect_common_behavior_parser]
        )
        ptt_review_parser.set_defaults(func=self.__dataset_handler.collect_ptt_review)
        dcard_review_parser: CliParser = collect_subparsers.add_parser(
            "dcard-review", help="", parents=[self.__collect_common_behavior_parser]
        )
        dcard_review_parser.set_defaults(func=self.__dataset_handler.collect_dcard_review)

        compute_sentiment_parser: CliParser = dataset_subparsers.add_parser(
            "compute_sentiment",
            help="",
            parents=[self.__structured_dataset_name_args_parser, self.__model_file_args_parser]
        )
        compute_sentiment_parser.set_defaults(func=self.__dataset_handler.compute_sentiment)

    def __setup_sentiment_score_model_subparser(self) -> None:
        sentiment_parser: CliParser = self._subparsers_action.add_parser("sentiment-score-model", help="")
        sentiment_subparsers: SubcommandAction = cast(
            SubcommandAction, sentiment_parser.add_subparsers(dest="sentiment_score_model", required=True, help="")
        )
        train_parser: CliParser = sentiment_subparsers.add_parser(
            'train',
            help='訓練情感分數模型。',
            parents=[self.__train_common_behavior_parser]
        )
        train_parser.set_defaults(func=self.__sentiment_model_handler.train)
        test_parser: CliParser = sentiment_subparsers.add_parser(
            'test',
            help="",
            parents=[self.__model_file_args_parser]
        )
        test_parser.add_argument("--input-sentence", type=str, required=True, help="")
        test_parser.set_defaults(func=self.__sentiment_model_handler.test)
        evaluate_parser: CliParser = sentiment_subparsers.add_parser('evaluate', help='')
        evaluate_subparsers: SubcommandAction = cast(
            SubcommandAction, evaluate_parser.add_subparsers(dest="evaluate", required=True, help="")
        )
        plot_parser: CliParser = evaluate_subparsers.add_parser(
            'plot', help="", parents=[self.__plot_common_behavior_parser]
        )
        plot_parser.set_defaults(func=self.__sentiment_model_handler.plot_graph)
        get_metrics_parser: CliParser = evaluate_subparsers.add_parser(
            'get-metrics', help="", parents=[self.__get_metrics_common_behavior_parser]
        )
        get_metrics_parser.set_defaults(func=self.__sentiment_model_handler.get_metrics)

    def __setup_prediction_model_subparser(self) -> None:
        prediction_parser: CliParser = self._subparsers_action.add_parser("prediction-model", help="")
        prediction_subparsers: SubcommandAction = cast(
            SubcommandAction, prediction_parser.add_subparsers(dest="prediction-model", required=True, help="")
        )
        train_parser: CliParser = prediction_subparsers.add_parser(
            'train',
            help='訓練情感分數模型。',
            parents=[self.__train_common_behavior_parser]
        )
        train_parser.set_defaults(func=self.__prediction_model_handler.train)
        test_parser: CliParser = prediction_subparsers.add_parser(
            'test',
            help="",
            parents=[self.__model_file_args_parser]
        )
        source_options_group = test_parser.add_mutually_exclusive_group(required=True)
        # 將旗標添加到這個群組中
        source_options_group.add_argument('--movie_name', type=str, help='')
        source_options_group.add_argument('--random', action='store_true', help='')

        test_parser.set_defaults(func=self.__prediction_model_handler.test)
        evaluate_parser: CliParser = prediction_subparsers.add_parser('evaluate', help='')
        evaluate_subparsers: SubcommandAction = cast(
            SubcommandAction, evaluate_parser.add_subparsers(dest="evaluate", required=True, help="")
        )
        plot_parser: CliParser = evaluate_subparsers.add_parser(
            'plot', help="", parents=[self.__plot_common_behavior_parser]
        )
        plot_parser.set_defaults(func=self.__prediction_model_handler.plot_graph)
        get_metrics_parser: CliParser = evaluate_subparsers.add_parser(
            'get-metrics', help="", parents=[self.__get_metrics_common_behavior_parser]
        )
        get_metrics_parser.set_defaults(func=self.__prediction_model_handler.get_metrics)

    def build(self) -> ArgumentParser:
        """
        Builds the complete argument parser by setting up all subparsers.

        :returns: The fully configured ArgumentParser instance.
        """

        # And then use this where appropriate, or adjust the main _dataset_name_parser.
        # For now, the example keeps the original _dataset_name_parser as required=True
        # and expects the calling logic in main.py to handle cases like --use_random_data.

        self.__setup_dataset_subparser()
        self.__setup_sentiment_score_model_subparser()
        self.__setup_prediction_model_subparser()
        return self.parser

    def parse_args(self, args: Optional[list[str]] = None) -> Namespace:
        """
        Parses command-line arguments using the built parser.

        This is a convenience method.

        :param args: Optional list of strings to parse. Defaults to sys.argv[1:].
        :returns: An object holding the parsed arguments.
        """
        # The build method should ideally be called once.
        # If parse_args can be called multiple times, ensure build() isn't re-executed wastefully.
        # A simple flag or checking if subparsers exist can manage this.
        # For this structure, build() is called before returning self.parser,
        # so self.parser is always the fully built one.
        return self.parser.parse_args(args=args)

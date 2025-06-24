# noinspection PyProtectedMember
from argparse import ArgumentParser, Namespace, _SubParsersAction
from typing import Optional

from src.cli.handlers import DatasetHandler, SentimentModelHandler, PredictionModelHandler

class ArgumentParserBuilder:
    """
    Builds and configures the command-line interface using argparse.

    This class encapsulates the entire CLI structure, including commands,
    sub-commands, and arguments, using a set of reusable parent parsers
    to avoid code duplication.
    """

    _MODEL_ID_KWARGS: dict[str, type|bool|str] = {
        "type": str,
        "required": True,
        "help": "The unique identifier for the model series."
    }

    def __init__(self) -> None:
        """Initializes the ArgumentParserBuilder."""
        self.parser: ArgumentParser = ArgumentParser(
            prog="movie_predictor",
            description="A command-line tool for movie box office prediction and analysis."
        )
        self._subparsers_action: _SubParsersAction = self.parser.add_subparsers(
            dest="command_group",
            required=True,
            title="Available command groups",
            description="Select a command group to see its specific commands."
        )
        self._built = False

        self._initialize_parent_parsers()
        self._initialize_handlers()

    def _initialize_parent_parsers(self) -> None:
        """Creates and stores reusable parent parsers for common arguments."""
        # Parent parser for arguments identifying a specific model file
        self.__model_file_args_parser: ArgumentParser = self.__create_model_file_args_parser()

        # Parent parser for commands that operate on a collection target
        self.__collect_common_behavior_parser: ArgumentParser = self.__create_collect_common_behavior_parser()

        # Parent parser for common training arguments
        self.__train_common_behavior_parser: ArgumentParser = self.__create_train_common_behavior_parser()

        # Parent parsers for evaluation-related commands
        self.__evaluate_common_behavior_parser: ArgumentParser = self.__create_evaluate_common_behavior_parser()
        self.__plot_common_behavior_parser: ArgumentParser = self.__create_plot_common_behavior_parser()
        self.__get_metrics_common_behavior_parser: ArgumentParser = self.__create_get_metrics_common_behavior_parser()

    def _initialize_handlers(self) -> None:
        """Initializes handlers for processing parsed commands."""
        self.__dataset_handler = DatasetHandler(self.parser)
        self.__sentiment_model_handler = SentimentModelHandler(self.parser)
        self.__prediction_model_handler = PredictionModelHandler(self.parser)

    @staticmethod
    def __create_model_file_args_parser() -> ArgumentParser:
        """Creates a parent parser for identifying a model by its ID and epoch."""
        parser: ArgumentParser = ArgumentParser(add_help=False)
        model_file_group = parser.add_argument_group(
            'Model File Identifier',
            description='Provide both model ID and epoch to locate a specific model file.'
        )
        model_file_group.add_argument('--model-id', **ArgumentParserBuilder._MODEL_ID_KWARGS)
        model_file_group.add_argument(
            '--epoch',
            type=int,
            required=True,
            help='The specific training epoch of the model to use.'
        )
        return parser

    @staticmethod
    def __create_collect_common_behavior_parser() -> ArgumentParser:
        """Creates a parent parser for specifying a data collection target."""
        parser: ArgumentParser = ArgumentParser(add_help=False)
        target_group = parser.add_mutually_exclusive_group(required=True)
        target_group.add_argument(
            '--structured-dataset-name',
            type=str,
            help='Target an entire structured dataset for data collection.'
        )
        target_group.add_argument(
            '--movie-name',
            type=str,
            help='Target a single movie for data collection.'
        )
        return parser

    def __create_train_common_behavior_parser(self) -> ArgumentParser:
        """Creates a parent parser with common arguments for model training."""
        parser: ArgumentParser = ArgumentParser(add_help=False)
        parser.add_argument('--model-id', **self._MODEL_ID_KWARGS)

        source_group = parser.add_mutually_exclusive_group(required=True)
        source_group.add_argument(
            '--feature-dataset-name',
            type=str,
            help='Specify the name of the feature dataset for training.'
        )
        source_group.add_argument(
            '--structured-dataset-name',
            type=str,
            help='Specify the name of the structured dataset to use as a source.'
        )
        source_group.add_argument(
            '--random-data',
            action='store_true',
            help='Use randomly generated data for training.'
        )

        parser.add_argument(
            '--old-epoch',
            type=int,
            required=False,
            help='The epoch of an existing model to continue training from.'
        )
        parser.add_argument(
            '--target-epoch',
            type=int,
            required=True,
            help='The target number of training epochs.'
        )
        parser.add_argument(
            '--checkpoint-interval',
            type=int,
            required=False,
            help='The interval in epochs at which to save model checkpoints.'
        )
        return parser

    @staticmethod
    def __create_evaluate_common_behavior_parser() -> ArgumentParser:
        """Creates a parent parser for selecting which metrics to evaluate or plot."""
        parser: ArgumentParser = ArgumentParser(add_help=False)
        plot_options_group = parser.add_argument_group(
            'Evaluation Options',
            description='Select at least one metric. Multiple selections are allowed.'
        )
        plot_options_group.add_argument('--training-loss', action='store_true',
                                        help='Evaluate or plot the training loss.')
        plot_options_group.add_argument('--validation-loss', action='store_true',
                                        help='Evaluate or plot the validation loss.')
        plot_options_group.add_argument('--f1-score', action='store_true', help='Evaluate or plot the F1-score.')
        return parser

    def __create_plot_common_behavior_parser(self) -> ArgumentParser:
        """Creates a parent parser for plotting graphs, combining evaluation and model ID parsers."""
        model_id_parser = ArgumentParser(add_help=False)
        model_id_parser.add_argument('--model-id', **self._MODEL_ID_KWARGS)
        return ArgumentParser(
            add_help=False, parents=[self.__evaluate_common_behavior_parser, model_id_parser]
        )

    def __create_get_metrics_common_behavior_parser(self) -> ArgumentParser:
        """Creates a parent parser for getting metrics, combining evaluation and model file parsers."""
        return ArgumentParser(
            add_help=False, parents=[self.__evaluate_common_behavior_parser, self.__model_file_args_parser]
        )

    def __setup_dataset_subparser(self) -> None:
        """Configures the 'dataset' command group and its sub-commands."""
        dataset_parser: ArgumentParser = self._subparsers_action.add_parser(
            "dataset",
            help="Commands for dataset creation and data collection."
        )
        dataset_subparsers: _SubParsersAction = dataset_parser.add_subparsers(
            dest="dataset_command", required=True, help="Available dataset commands."
        )

        # Command: dataset index
        index_parser: ArgumentParser = dataset_subparsers.add_parser(
            "index",
            help="Create an index file for a new structured dataset from a source CSV."
        )
        index_parser.add_argument(
            "--structured-dataset-name", type=str, required=True, help="The name for the new structured dataset."
        )
        index_parser.add_argument(
            "--source-file", type=str, required=True, help="Path to the source CSV file."
        )
        index_parser.set_defaults(func=self.__dataset_handler.create_index)

        # Command group: dataset collect
        collect_parser: ArgumentParser = dataset_subparsers.add_parser(
            "collect", help="Collect data (e.g., box office, reviews) for a dataset."
        )
        collect_subparsers: _SubParsersAction = collect_parser.add_subparsers(
            dest="collect_command", required=True, help="Specify the type of data to collect."
        )
        collect_subparsers.add_parser(
            "box-office", help="Collect box office data.", parents=[self.__collect_common_behavior_parser]
        ).set_defaults(func=self.__dataset_handler.collect_box_office)
        collect_subparsers.add_parser(
            "ptt-review", help="Collect PTT reviews.", parents=[self.__collect_common_behavior_parser]
        ).set_defaults(func=self.__dataset_handler.collect_ptt_review)
        collect_subparsers.add_parser(
            "dcard-review", help="Collect Dcard reviews.", parents=[self.__collect_common_behavior_parser]
        ).set_defaults(func=self.__dataset_handler.collect_dcard_review)

        # Command: dataset compute-sentiment
        compute_sentiment_parser: ArgumentParser = dataset_subparsers.add_parser(
            "compute-sentiment",
            help="Compute sentiment scores for reviews in a dataset using a trained model.",
            parents=[self.__model_file_args_parser]
        )
        compute_sentiment_parser.add_argument(
            "--structured-dataset-name", type=str, required=True, help="The dataset to process."
        )
        compute_sentiment_parser.set_defaults(func=self.__dataset_handler.compute_sentiment)

    def __setup_sentiment_model_subparser(self) -> None:
        """Configures the 'sentiment-model' command group and its sub-commands."""
        sentiment_parser: ArgumentParser = self._subparsers_action.add_parser(
            "sentiment-model", help="Commands for the sentiment analysis model."
        )
        sentiment_subparsers: _SubParsersAction = sentiment_parser.add_subparsers(
            dest="sentiment_subcommand", required=True, help="Available sentiment model commands."
        )

        sentiment_subparsers.add_parser(
            'train', help='Train a sentiment analysis model.', parents=[self.__train_common_behavior_parser]
        ).set_defaults(func=self.__sentiment_model_handler.train)

        test_parser: ArgumentParser = sentiment_subparsers.add_parser(
            'test', help="Test the sentiment model with a sentence.", parents=[self.__model_file_args_parser]
        )
        test_parser.add_argument("--input-sentence", type=str, required=True, help="The sentence to analyze.")
        test_parser.set_defaults(func=self.__sentiment_model_handler.test)

        evaluate_parser: ArgumentParser = sentiment_subparsers.add_parser(
            'evaluate', help='Evaluate the sentiment model.'
        )
        evaluate_subparsers: _SubParsersAction = evaluate_parser.add_subparsers(
            dest="evaluate_command", required=True
        )
        evaluate_subparsers.add_parser(
            'plot', help="Plot evaluation graphs.", parents=[self.__plot_common_behavior_parser]
        ).set_defaults(func=self.__sentiment_model_handler.plot_graph)
        evaluate_subparsers.add_parser(
            'get-metrics', help="Get specific evaluation metrics.", parents=[self.__get_metrics_common_behavior_parser]
        ).set_defaults(func=self.__sentiment_model_handler.get_metrics)

    def __setup_prediction_model_subparser(self) -> None:
        """Configures the 'prediction-model' command group and its sub-commands."""
        prediction_parser: ArgumentParser = self._subparsers_action.add_parser(
            "prediction-model", help="Commands for the box office prediction model."
        )
        prediction_subparsers: _SubParsersAction = prediction_parser.add_subparsers(
            dest="prediction_subcommand", required=True, help="Available prediction model commands."
        )

        prediction_subparsers.add_parser(
            'train', help='Train a box office prediction model.', parents=[self.__train_common_behavior_parser]
        ).set_defaults(func=self.__prediction_model_handler.train)

        test_parser: ArgumentParser = prediction_subparsers.add_parser(
            'test', help="Test the prediction model.", parents=[self.__model_file_args_parser]
        )
        source_group = test_parser.add_mutually_exclusive_group(required=True)
        source_group.add_argument('--movie-name', type=str, help='The name of the movie to test a prediction on.')
        source_group.add_argument('--random', action='store_true', help='Use random data for the prediction test.')
        test_parser.set_defaults(func=self.__prediction_model_handler.test)

        evaluate_parser: ArgumentParser = prediction_subparsers.add_parser(
            'evaluate', help='Evaluate the prediction model.'
        )
        evaluate_subparsers: _SubParsersAction = evaluate_parser.add_subparsers(
            dest="evaluate_command", required=True
        )
        evaluate_subparsers.add_parser(
            'plot', help="Plot evaluation graphs.", parents=[self.__plot_common_behavior_parser]
        ).set_defaults(func=self.__prediction_model_handler.plot_graph)
        evaluate_subparsers.add_parser(
            'get-metrics', help="Get specific evaluation metrics.", parents=[self.__get_metrics_common_behavior_parser]
        ).set_defaults(func=self.__prediction_model_handler.get_metrics)

    def build(self) -> ArgumentParser:
        """
        Builds the complete argument parser by setting up all subparsers.
        This method is idempotent.

        :returns: The fully configured ArgumentParser instance.
        """
        if self._built:
            return self.parser

        self.__setup_dataset_subparser()
        self.__setup_sentiment_model_subparser()
        self.__setup_prediction_model_subparser()
        self._built = True
        return self.parser

    def parse_args(self, args: Optional[list[str]] = None) -> Namespace:
        """
        Builds the parser if not already built, then parses command-line arguments.

        This is a convenience method that simplifies the user-facing API.

        :param args: Optional list of strings to parse. Defaults to sys.argv[1:].
        :returns: An object holding the parsed arguments.
        """
        self.build()
        return self.parser.parse_args(args=args)

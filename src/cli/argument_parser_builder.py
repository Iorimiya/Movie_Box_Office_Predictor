# noinspection PyProtectedMember
from argparse import ArgumentParser, Namespace, _SubParsersAction
from pathlib import Path
from typing import Optional

from src.cli.handlers import DatasetHandler, SentimentModelHandler, PredictionModelHandler, BaseModelHandler


class ArgumentParserBuilder:
    """
    Builds and configures the ArgumentParser for the application's command-line interface.

    This class encapsulates the entire CLI structure, including command groups,
    sub-commands, and their respective arguments. It uses a builder pattern to
    construct the parser lazily and ensures that the setup logic is organized
    and reusable through parent parsers and handler classes.

    :cvar _MODEL_ID_KWARGS: A dictionary of common keyword arguments for the '--model-id' CLI option.
    :ivar parser: The root ArgumentParser instance.
    :ivar _subparsers_action: The subparsers action object for adding command groups.
    :ivar _built: A flag to prevent re-building the parser.
    :ivar __model_file_args_parser: A parent parser for identifying a specific model file.
    :ivar __collect_common_behavior_parser: A parent parser for data collection commands.
    :ivar __train_common_behavior_parser: A parent parser for model training commands.
    :ivar __evaluate_common_behavior_parser: A parent parser for common evaluation options.
    :ivar __plot_common_behavior_parser: A parent parser for plotting evaluation graphs.
    :ivar __get_metrics_common_behavior_parser: A parent parser for fetching evaluation metrics.
    :ivar __dataset_handler: The handler for 'dataset' command logic.
    :ivar __sentiment_model_handler: The handler for 'sentiment-model' command logic.
    :ivar __prediction_model_handler: The handler for 'prediction-model' command logic.
    """
    _MODEL_ID_KWARGS: dict[str, type|bool|str] = {
        "type": str,
        "required": True,
        "help": "The unique identifier for the model series."
    }

    def __init__(self) -> None:
        """
        Initializes the ArgumentParserBuilder.

        Sets up the main parser, its subparsers, and then calls initialization
        methods for parent parsers and command handlers.
        """
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
        """
        Creates and initializes all parent parsers for reusable argument groups.

        Parent parsers define common sets of arguments (e.g., for identifying a model,
        training, or evaluation) that can be inherited by multiple sub-commands,
        reducing code duplication.
        """
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
        """
        Initializes the command handler instances.

        Each handler is responsible for the logic associated with a specific command
        group (e.g., 'dataset', 'sentiment-model').
        """
        self.__dataset_handler = DatasetHandler(self.parser)
        self.__sentiment_model_handler = SentimentModelHandler(self.parser)
        self.__prediction_model_handler = PredictionModelHandler(self.parser)

    @staticmethod
    def __create_model_file_args_parser() -> ArgumentParser:
        """
        Creates a parent parser for arguments that identify a specific model file.

        This parser includes the '--model-id' and '--epoch' arguments.

        :returns: An ArgumentParser configured with model file identifier arguments.
        """
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
        """
        Creates a parent parser for common arguments in data collection commands.

        This parser defines a mutually exclusive group for targeting either an
        entire dataset ('--structured-dataset-name') or a single movie ('--movie-name').

        :returns: An ArgumentParser configured with data collection target arguments.
        """
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
        """
        Creates a parent parser for common arguments in model training commands.

        This includes the required '--model-id', optional configuration overrides
        (via file or individual parameters), and options for continuing training
        from a checkpoint.

        :returns: An ArgumentParser configured with common training arguments.
        """
        parser: ArgumentParser = ArgumentParser(add_help=False)

        # --- Required Argument ---
        parser.add_argument('--model-id', **self._MODEL_ID_KWARGS)

        # --- Optional Override Method 1: File-based ---
        file_override_group = parser.add_argument_group(
            'File-based Parameter Override (Optional)',
            description='Override default parameters using a configuration file. '
                        'Note: This is mutually exclusive with individual parameter overrides below.'
        )
        file_override_group.add_argument(
            '--config-override',
            type=Path,
            required=False,
            help='Path to a YAML file with parameters to override the defaults.'
        )

        # --- Optional Override Method 2: Individual Parameters ---
        params_override_group = parser.add_argument_group(
            'Individual Parameter Overrides (Optional)',
            description='Override specific default parameters directly. '
                        'Note: This is mutually exclusive with the file-based override above.'
        )
        params_override_group.add_argument(
            '--dataset-file-name', type=str, required=False, help='Override the source dataset CSV file name.'
        )
        params_override_group.add_argument(
            '--epochs', type=int, required=False, help='Override the number of training epochs.'
        )
        params_override_group.add_argument(
            '--batch-size', type=int, required=False, help='Override the batch size for training.'
        )
        params_override_group.add_argument(
            '--vocabulary-size', type=int, required=False, help='Override the vocabulary size.'
        )
        params_override_group.add_argument(
            '--embedding-dim', type=int, required=False, help='Override the embedding dimension.'
        )
        params_override_group.add_argument(
            '--lstm-units', type=int, required=False, help='Override the number of LSTM units.'
        )
        params_override_group.add_argument(
            '--split-ratios', type=int, nargs=3, metavar=('TRAIN', 'VAL', 'TEST'),
            required=False, help='Override the data split ratios (e.g., 8 1 1 for 80/10/10).'
        )
        params_override_group.add_argument(
            '--random-state', type=int, required=False, help='Override the random state for data splitting.'
        )
        params_override_group.add_argument(
            '--checkpoint-interval', type=int, required=False, help='Override the model checkpoint interval.'
        )

        continue_group = parser.add_argument_group(
            'Continue Training (Optional)',
            description='Options to continue training from a previously saved checkpoint. '
                        'If used, a new model will not be created. The configuration '
                        'from the original model run will be used.'
        )

        continue_group.add_argument(
            '--continue-from-epoch',
            type=int,
            required=False,
            help='The epoch number of the checkpoint to load and continue training from.'
        )

        return parser

    @staticmethod
    def __create_evaluate_common_behavior_parser() -> ArgumentParser:
        """
        Creates a parent parser for common arguments in model evaluation commands.

        This parser provides boolean flags for selecting which metrics to evaluate
        or plot, such as '--training-loss', '--validation-loss', and '--f1-score'.

        :returns: An ArgumentParser configured with common evaluation metric-selection arguments.
        """
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
        """
        Creates a parent parser for plotting evaluation graphs.

        This parser combines the common evaluation metric-selection arguments with
        the '--model-id' argument.

        :returns: An ArgumentParser configured for plotting commands.
        """
        model_id_parser = ArgumentParser(add_help=False)
        model_id_parser.add_argument('--model-id', **self._MODEL_ID_KWARGS)
        return ArgumentParser(
            add_help=False, parents=[self.__evaluate_common_behavior_parser, model_id_parser]
        )

    def __create_get_metrics_common_behavior_parser(self) -> ArgumentParser:
        """
        Creates a parent parser for fetching specific evaluation metrics.

        This parser combines the common evaluation metric-selection arguments with
        the arguments for identifying a specific model file (model ID and epoch).

        :returns: An ArgumentParser configured for getting specific metrics.
        """
        return ArgumentParser(
            add_help=False, parents=[self.__evaluate_common_behavior_parser, self.__model_file_args_parser]
        )

    def __add_evaluate_subcommands(
        self,
        parent_subparsers: _SubParsersAction,
        handler: BaseModelHandler,
        model_type_name: str
    ) -> None:
        """
        Adds the 'evaluate' command and its sub-commands ('plot', 'get-metrics')
        to a parent subparser.

        This helper method centralizes the creation of the evaluation command
        structure to avoid code duplication across different model types.

        :param parent_subparsers: The subparsers action object to which the 'evaluate' command will be added.
        :param handler: The model-specific handler instance that contains the logic for plotting and getting metrics.
        :param model_type_name: The name of the model type (e.g., 'sentiment'), used for help messages.
        """
        evaluate_parser: ArgumentParser = parent_subparsers.add_parser(
            'evaluate', help=f'Evaluate the {model_type_name} model.'
        )
        evaluate_subparsers: _SubParsersAction = evaluate_parser.add_subparsers(
            dest="evaluate_command", required=True
        )
        evaluate_subparsers.add_parser(
            'plot', help="Plot evaluation graphs.", parents=[self.__plot_common_behavior_parser]
        ).set_defaults(func=handler.plot_graph)
        evaluate_subparsers.add_parser(
            'get-metrics', help="Get specific evaluation metrics.", parents=[self.__get_metrics_common_behavior_parser]
        ).set_defaults(func=handler.get_metrics)

    def __setup_dataset_subparser(self) -> None:
        """
        Sets up the 'dataset' command group and its sub-commands.

        This defines the following command structure:
        - `dataset index`: To create a dataset index.
        - `dataset collect <type>`: To collect data (box-office, ptt-review, etc.).
        - `dataset compute-sentiment`: To run sentiment analysis on a dataset.
        """
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
        """
        Sets up the 'sentiment-model' command group and its sub-commands.

        This defines the following command structure:
        - `sentiment-model train`: To train a new model.
        - `sentiment-model predict`: To test the model with a sentence.
        - `sentiment-model evaluate plot`: To plot evaluation graphs.
        - `sentiment-model evaluate get-metrics`: To get specific metric values.
        """
        sentiment_parser: ArgumentParser = self._subparsers_action.add_parser(
            "sentiment-model", help="Commands for the sentiment analysis model."
        )
        sentiment_subparsers: _SubParsersAction = sentiment_parser.add_subparsers(
            dest="sentiment_subcommand", required=True, help="Available sentiment model commands."
        )

        sentiment_subparsers.add_parser(
            'train', help='Train a sentiment analysis model.', parents=[self.__train_common_behavior_parser]
        ).set_defaults(func=self.__sentiment_model_handler.train)

        predict_parser: ArgumentParser = sentiment_subparsers.add_parser(
            'predict', help="Test the sentiment model with a sentence.", parents=[self.__model_file_args_parser]
        )
        predict_parser.add_argument(
            "--input-sentence", type=str, required=True, help="The sentence to analyze."
        )
        predict_parser.set_defaults(func=self.__sentiment_model_handler.predict)

        self.__add_evaluate_subcommands(
            parent_subparsers=sentiment_subparsers,
            handler=self.__sentiment_model_handler,
            model_type_name="sentiment"
        )

    def __setup_prediction_model_subparser(self) -> None:
        """
        Sets up the 'prediction-model' command group and its sub-commands.

        This defines the following command structure:
        - `prediction-model train`: To train a new model.
        - `prediction-model predict`: To test the model.
        - `prediction-model evaluate plot`: To plot evaluation graphs.
        - `prediction-model evaluate get-metrics`: To get specific metric values.
        """
        prediction_parser: ArgumentParser = self._subparsers_action.add_parser(
            "prediction-model", help="Commands for the box office prediction model."
        )
        prediction_subparsers: _SubParsersAction = prediction_parser.add_subparsers(
            dest="prediction_subcommand", required=True, help="Available prediction model commands."
        )

        prediction_subparsers.add_parser(
            'train', help='Train a box office prediction model.', parents=[self.__train_common_behavior_parser]
        ).set_defaults(func=self.__prediction_model_handler.train)

        predict_parser: ArgumentParser = prediction_subparsers.add_parser(
            'predict', help="Test the prediction model.", parents=[self.__model_file_args_parser]
        )
        source_group = predict_parser.add_mutually_exclusive_group(required=True)
        source_group.add_argument(
            '--movie-name', type=str, help='The name of the movie to predict a prediction on.'
        )
        source_group.add_argument(
            '--random', action='store_true', help='Use random data for the prediction predict.'
        )
        predict_parser.set_defaults(func=self.__prediction_model_handler.predict)

        self.__add_evaluate_subcommands(
            parent_subparsers=prediction_subparsers,
            handler=self.__prediction_model_handler,
            model_type_name="prediction"
        )

    def build(self) -> ArgumentParser:
        """
        Constructs and returns the complete ArgumentParser.

        This method orchestrates the setup of all command groups and their
        sub-commands. It ensures the parser is only built once.

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

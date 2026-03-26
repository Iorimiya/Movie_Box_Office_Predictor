# noinspection PyProtectedMember
from argparse import ArgumentParser, Namespace, _SubParsersAction
from pathlib import Path
from typing import Any, Callable, Optional

from src.cli.handlers.base_model_handler import BaseModelHandler
from src.cli.handlers.box_office_regression_model_handler import BoxOfficeRegressionModelHandler
from src.cli.handlers.dataset_handler import DatasetHandler


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
    :ivar __box_office_regression_model_handler: The handler for 'box-office-regression-model' command logic.
    """
    _MODEL_ID_KWARGS: dict[str, type | bool | str] = {
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
            description="A command-line tool for movie box office box_office_regression and analysis."
        )
        self._subparsers_action: _SubParsersAction = self.parser.add_subparsers(
            dest="command_group",
            required=True,
            title="Available command groups",
            description="Select a command group to see its specific commands."
        )
        self._built: bool = False

        self._initialize_parent_parsers()
        self._initialize_handlers()

    @staticmethod
    def __build_subcommand(
        parent_subparsers: _SubParsersAction,
        name: str,
        help_text: str,
        handler_function: Callable[..., None],
        parent_parsers: list[ArgumentParser],
        customizer: Optional[Callable[[ArgumentParser], None]] = None
    ) -> None:
        """
        A generic factory method to build a single, executable sub-command.

        This is the fundamental building block for creating a leaf-node command
        in the CLI tree. It encapsulates the common pattern of adding a parser,
        inheriting arguments from specified parents, applying command-specific
        customizations, and setting the default handler function to be executed.

        :param parent_subparsers: The subparsers action object to which the new command will be added.
        :param name: The name of the sub-command (e.g., 'train', 'plot').
        :param help_text: The help message displayed for the sub-command.
        :param handler_function: The function to be executed when this command is invoked.
        :param parent_parsers: A list of parent parsers from which to inherit common arguments.
        :param customizer: An optional function that takes the newly created parser
                           as an argument to add unique, command-specific arguments.
        """
        parser: ArgumentParser = parent_subparsers.add_parser(
            name, help=help_text, parents=parent_parsers
        )
        if customizer:
            customizer(parser)
        parser.set_defaults(func=handler_function)

    @staticmethod
    def __add_command_group(
        *,
        parent_subparsers: _SubParsersAction,
        name: str,
        help_text: str,
        child_command_specs: list[dict],
        customizer: Optional[Callable[[_SubParsersAction], None]] = None
    ) -> None:
        """
        A generic factory to create a command group with its own sub-commands.

        This method creates a "container" command (like 'evaluate' or 'collect')
        that holds its own set of sub-commands, defined by a list of specs.
        It supports creating both leaf-node commands and nested command groups.

        :param parent_subparsers: The subparsers action to add the new group to.
        :param name: The name of the command group.
        :param help_text: The help message for the command group.
        :param child_command_specs: A list of dictionaries, where each dict
                                    defines a child command or a nested group.
        :param customizer: An optional function that takes the new subparsers
                           action and adds more custom commands to it.
        """
        group_parser: ArgumentParser = parent_subparsers.add_parser(name, help=help_text)
        group_subparsers: _SubParsersAction = group_parser.add_subparsers(
            dest=f"{name}_command", required=True
        )
        for spec in child_command_specs:
            spec_type: str = spec.pop('type', 'command')  # Default to 'command'

            match spec_type:
                case 'command':
                    ArgumentParserBuilder.__build_subcommand(
                        parent_subparsers=group_subparsers,
                        **spec
                    )
                case 'group':
                    ArgumentParserBuilder.__add_command_group(
                        parent_subparsers=group_subparsers,
                        **spec
                    )
                case _:
                    pass
        if customizer:
            customizer(group_subparsers)

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
        self.__dataset_handler: DatasetHandler = DatasetHandler(self.parser)
        self.__box_office_regression_model_handler: BoxOfficeRegressionModelHandler = BoxOfficeRegressionModelHandler(
            self.parser)

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

        # Required Argument
        parser.add_argument('--model-id', **self._MODEL_ID_KWARGS)

        # Optional Override Method 1: File-based
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

        # Optional Override Method 2: Individual Parameters
        params_override_group = parser.add_argument_group(
            'Individual Parameter Overrides (Optional)',
            description='Override specific default parameters directly. '
                        'Note: This is mutually exclusive with the file-based override above.'
        )
        params_override_group.add_argument(
            '--dataset-name', type=str, required=False, help='Override the name of the dataset to use for training.'
        )
        params_override_group.add_argument(
            '--epochs', type=int, required=False, help='Override the number of training epochs.'
        )
        params_override_group.add_argument(
            '--checkpoint-interval', type=int, required=False, help='Override the model checkpoint interval.'
        )
        params_override_group.add_argument(
            '--training-week-len', type=int, required=False, help='Override the model training week length.'
        )
        params_override_group.add_argument(
            '--split-ratios', type=int, nargs=3, metavar=('TRAIN', 'VAL', 'TEST'),
            required=False, help='Override the data split ratios (e.g., 8 1 1 for 80/10/10).'
        )
        params_override_group.add_argument(
            '--random-state', type=int, required=False, help='Override the random state for data splitting.'
        )
        params_override_group.add_argument(
            '--lstm-units', type=int, required=False, help='Override the number of LSTM units.'
        )
        params_override_group.add_argument(
            '--dropout-rate', type=float, required=False, help='Override the dropout rate.'
        )
        params_override_group.add_argument(
            '--batch-size', type=int, required=False, help='Override the batch size for training.'
        )
        params_override_group.add_argument(
            '--early-stopping-patience',
            type=int,
            required=False,
            help='Override the patience for early stopping.'
        )
        params_override_group.add_argument(
            '--early-stopping-monitor',
            type=str,
            required=False,
            help="Override the metric to monitor for early stopping (e.g., 'val_loss', 'val_f1_score')."
        )
        params_override_group.add_argument(
            '--early-stopping-min-delta',
            type=float,  # Explicitly define the type as float
            required=False,
            help='Override the minimum delta for early stopping (e.g., 1e-5).'
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

        This parser provides flags for selecting which metrics to evaluate or plot.
        It follows an "action-centric" design, where a main flag triggers a type
        of evaluation, and other flags control what to display from the results.

        :returns: An ArgumentParser configured with common evaluation arguments.
        """
        parser: ArgumentParser = ArgumentParser(add_help=False)

        # General Metrics
        general_metrics_group = parser.add_argument_group(
            'General Metrics',
            description='Flags to display general metrics like loss values.'
        )
        general_metrics_group.add_argument(
            '--training-loss', action='store_true', help='Display or plot the training loss from history.'
        )
        general_metrics_group.add_argument(
            '--validation-loss', action='store_true', help='Display or plot the validation loss from history.'
        )
        general_metrics_group.add_argument(
            '--test-loss', action='store_true', help='Calculate and display the loss on the test set.'
        )

        # Classification Evaluation (Action-Centric)
        class_eval_group = parser.add_argument_group(
            'Classification Evaluation',
            description='Options to run a classification-based evaluation and display its results.'
        )
        class_eval_group.add_argument(
            '--classification-report',
            type=str,
            choices=['range', 'trend'],
            required=False,
            help="Trigger a classification evaluation. Choose 'range' for box-office ranges or 'trend' for up/down trend."
        )
        class_eval_group.add_argument(
            '--show-f1-score',
            action='store_true',
            help="Display the F1-score from the classification report. Requires --classification-report."
        )
        class_eval_group.add_argument(
            '--show-confusion-matrix',
            action='store_true',
            help="Display the confusion matrix from the classification report. Requires --classification-report."
        )

        # Evaluation Context
        context_group = parser.add_argument_group(
            'Evaluation Context',
            description='Options to control the dataset used for evaluation.'
        )
        context_group.add_argument(
            '--dataset-name',
            type=str,
            help='Optional. Specify a new dataset to evaluate on. '
                 'If provided, this triggers \'exploratory mode\' to test on the full, unsplit dataset. '
                 'If omitted (default), the original training dataset is used to reproduce the exact test set.'
        )
        return parser

    def __create_plot_common_behavior_parser(self) -> ArgumentParser:
        """
        Creates a parent parser for plotting evaluation graphs.

        This parser combines the common evaluation metric-selection arguments with
        the '--model-id' argument.

        :returns: An ArgumentParser configured for plotting commands.
        """
        model_id_parser: ArgumentParser = ArgumentParser(add_help=False)
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

    def __create_evaluate_specs(self, handler: BaseModelHandler) -> list[dict[str, Any]]:
        """
        Creates a list of default specifications for the 'evaluate' sub-commands.

        This method acts as a template provider, returning a standard configuration
        for 'plot' and 'get-metrics' commands. The calling function is responsible
        for further customization and for passing these specs to a group builder.

        :param handler: The model-specific handler containing the logic for the commands.
        :returns: A list of dictionaries, where each dict defines a sub-command.
        """
        return [
            {
                "name": 'plot',
                "help_text": "Plot evaluation graphs.",
                "handler_function": handler.plot_graph,
                "parent_parsers": [self.__plot_common_behavior_parser]

            },
            {
                "name": 'get-metrics',
                "help_text": "Get specific evaluation metrics.",
                "handler_function": handler.get_metrics,
                "parent_parsers": [self.__get_metrics_common_behavior_parser]
            }
        ]

    def __setup_dataset_subparser(self) -> None:
        """
        Sets up the 'dataset' command group and its sub-commands.

        This defines the following command structure:
        - `dataset index`: To create a dataset index.
        - `dataset collect <type>`: To collect data (box-office, ptt-review, etc.).
        - `dataset compute-sentiment`: To run sentiment analysis on a dataset.
        """

        def customize_index_args(parser: ArgumentParser) -> None:
            """
            Adds arguments specific to the 'dataset index' command.

            :param parser: The ArgumentParser instance to which arguments will be added.
            """
            parser.add_argument(
                "--structured-dataset-name", type=str, required=True, help="The name for the new structured dataset."
            )
            parser.add_argument(
                "--source-file", type=str, required=True, help="Path to the source CSV file."
            )

        def customize_sentiment_args(parser: ArgumentParser) -> None:
            """
            Adds arguments specific to the 'dataset compute-sentiment' command.

            :param parser: The ArgumentParser instance to which arguments will be added.
            """
            parser.add_argument('--model-id', **self._MODEL_ID_KWARGS)
            parser.add_argument(
                "--structured-dataset-name", type=str, required=True, help="The dataset to process."
            )

        dataset_specs: list[dict] = [
            {
                "type": "command",
                "name": "index",
                "help_text": "Create an index file for a new structured dataset from a source CSV.",
                "handler_function": self.__dataset_handler.create_index,
                "parent_parsers": [],
                "customizer": customize_index_args
            },
            {
                "type": "group",
                "name": "collect",
                "help_text": "Collect data (e.g., box office, reviews) for a dataset.",
                "child_command_specs": [
                    {
                        "type": "command",
                        "name": "box-office",
                        "help_text": "Collect box office data.",
                        "handler_function": self.__dataset_handler.collect_box_office,
                        "parent_parsers": [self.__collect_common_behavior_parser]
                    },
                    {
                        "type": "command",
                        "name": "ptt-review",
                        "help_text": "Collect PTT reviews.",
                        "handler_function": self.__dataset_handler.collect_ptt_review,
                        "parent_parsers": [self.__collect_common_behavior_parser]
                    },
                    {
                        "type": "command",
                        "name": "dcard-review",
                        "help_text": "Collect Dcard reviews.",
                        "handler_function": self.__dataset_handler.collect_dcard_review,
                        "parent_parsers": [self.__collect_common_behavior_parser]
                    }
                ]
            },
            {
                "type": "command",
                "name": "compute-sentiment",
                "help_text": "Compute sentiment scores for reviews in a dataset using a model.",
                "handler_function": self.__dataset_handler.compute_sentiment,
                "parent_parsers": [],
                "customizer": customize_sentiment_args
            }
        ]

        ArgumentParserBuilder.__add_command_group(
            parent_subparsers=self._subparsers_action,
            name="dataset",
            help_text="Commands for dataset creation and data collection.",
            child_command_specs=dataset_specs
        )

    def __setup_box_office_regression_model_subparser(self) -> None:
        """
        Sets up the 'box-office-regression-model' command group and its sub-commands.

        This defines the following command structure:
        - `box-office-regression-model train`: To train a new model.
        - `box-office-regression-model predict`: To test the model.
        - `box-office-regression-model evaluate plot`: To plot evaluation graphs.
        - `box-office-regression-model evaluate get-metrics`: To get specific metric values.
        """

        def add_prediction_predict_args(parser: ArgumentParser) -> None:
            """
            Adds arguments specific to the 'box-office-regression-model predict' command.

            This includes a mutually exclusive group for specifying the prediction
            source, either by movie name or by using random data.

            :param parser: The ArgumentParser instance to which arguments will be added.
            """
            source_group = parser.add_mutually_exclusive_group(required=True)
            source_group.add_argument(
                '--movie-name', type=str, help='The name of the movie to predict a prediction on.'
            )
            source_group.add_argument(
                '--random', action='store_true', help='Use random data for the prediction predict.'
            )

        def add_f1_metrics_args(parser: ArgumentParser) -> None:
            """
            Adds F1-score related arguments for evaluation commands.

            This customizer adds a group of arguments for fine-tuning the F1-score
            calculation. It includes options for the classification conversion
            method ('--f1-method'), the averaging strategy for multi-class
            results ('--f1-average-method'), and the specific ranges to use
            when the 'range' method is selected ('--box-office-ranges').

            :param parser: The ArgumentParser instance to which arguments will be added.
            """

            parser.add_argument(
                '--f1-average-method',
                type=str,
                required=False,
                choices=['micro', 'macro', 'weighted', 'none'],
                help='The averaging method for F1 score calculation.'
                     'Options: "micro", "macro", "weighted", "none".'
            )
            parser.add_argument(
                '--box-office-ranges',
                type=int,
                nargs='+',
                metavar='RANGE',
                required=False,
                help='Ranges for classifying box office when using the "range" F1-method. '
                     'Example: --box-office-ranges 1000000 5000000 10000000'
            )

        evaluate_child_specs: list[dict] = self.__create_evaluate_specs(
            handler=self.__box_office_regression_model_handler)
        for spec in evaluate_child_specs:
            spec['customizer'] = add_f1_metrics_args

        box_office_regression_model_specs: list[dict] = [
            {
                "type": "command",
                "name": 'train',
                "help_text": 'Train a box office regression model.',
                "handler_function": self.__box_office_regression_model_handler.train,
                "parent_parsers": [self.__train_common_behavior_parser],
                "customizer": add_f1_metrics_args
            },
            {
                "type": "command",
                "name": 'predict',
                "help_text": 'Test the regression model.',
                "handler_function": self.__box_office_regression_model_handler.predict,
                "parent_parsers": [self.__model_file_args_parser],
                "customizer": add_prediction_predict_args
            },
            {
                "type": "group",
                "name": 'evaluate',
                "help_text": 'Evaluate the regression model.',
                "child_command_specs": evaluate_child_specs
            }
        ]

        ArgumentParserBuilder.__add_command_group(
            parent_subparsers=self._subparsers_action,
            name="box-office-regression-model",
            help_text="Commands for the box office regression model.",
            child_command_specs=box_office_regression_model_specs
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
        self.__setup_box_office_regression_model_subparser()
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

from logging import Logger
from pathlib import Path
from argparse import ArgumentParser, Namespace
from typing import Optional

from src.utilities.util import recreate_folder
from src.core.constants import Constants
from utilities.plot import plot_training_loss, plot_validation_loss, \
    plot_trend_accuracy, plot_range_accuracy
from src.core.logging_manager import LoggingManager, LogLevel, HandlerSettings
from src.data_handling.movie_data import load_index_file, MovieData
from data_collection.review_collector import ReviewCollector
from data_collection.box_office_collector import BoxOfficeCollector
from src.models.box_office_prediction import MoviePredictionModel
from src.models.review_sentiment_analysis import ReviewSentimentAnalyseModel


def set_argument_parser() -> Namespace:
    """
    Sets up the argument parser for the program.

    Returns:
        Namespace: Parsed command-line arguments.
    """
    parser: ArgumentParser = ArgumentParser(prog=None, usage=None, description=None, epilog=None)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-u", "--user", action="store_true", help="execute program as a user.")
    group.add_argument("-d", "--developer", action="store_true", help="execute program as a developer.")
    group.add_argument("-f", "--function", type=str,
                       choices=["collect_box_office", "collect_ptt_review", "collect_dcard_review",
                                "train_review_sentiment_model", "test_review_sentiment_model",
                                "add_sentiment_score_to_saved_data",
                                "train_movie_prediction_model", "train_movie_prediction_model_with_checkpointing",
                                "train_movie_prediction_model_with_randomly_generated_data",
                                "test_movie_prediction_model_with_randomly_generated_data",
                                "movie_prediction_model_trend_evaluation", "movie_prediction_model_range_evaluation",
                                "plot_training_loss_curve", "plot_validation_loss_curve",
                                "plot_trend_accuracy_curve", "plot_range_accuracy_curve"],
                       help="unit test")

    parser.add_argument("--movie_name", type=str, required=False,
                        help="the movie name that user want to get rating result, or the target movie name that search in unit test.")
    parser.add_argument("--model_name", type=str, required=False,
                        help="the model name as saving training result, or existed model for evaluation.")
    parser.add_argument("--old_model_name", type=str, required=False, help="the old model name for continue training.")
    parser.add_argument("--target_epoch", type=int, required=False, help="target training epoch of model.")
    parser.add_argument("--saving_epoch", type=int, required=False, help="saving interval in epochs.")
    parser.add_argument("--path", type=str, required=False, help="file path for unit test.")
    parser.add_argument("--input", type=str, required=False, help="the input of unit test.")

    return parser.parse_args()


if __name__ == "__main__":
    main_logger: Logger = LoggingManager.create_predefined_manager().get_logger('root')

    args = set_argument_parser()

    if args.user:
        if args.movie_name:
            main_logger.info("Collecting box office, reviews and predicting box office next week.")
            main_logger.info(f"Movie name inputted: {args.movie_name}.")
            download_temp_folder = Constants.SCRAPING_DATA_FOLDER.joinpath("temp")
            recreate_folder(path=download_temp_folder)
            movie_data: MovieData = MovieData(movie_name=args.movie_name, movie_id=0)
            BoxOfficeCollector(download_mode=BoxOfficeCollector.Mode.WEEK,
                               box_office_data_folder=download_temp_folder).download_single_box_office_data(
                movie_data=movie_data)
            ReviewCollector(target_website=ReviewCollector.TargetWebsite.PTT).search_review_with_single_movie(
                movie_data=movie_data)
            analyzer: ReviewSentimentAnalyseModel = ReviewSentimentAnalyseModel(
                model_path=Constants.REVIEW_SENTIMENT_ANALYSIS_MODEL_PATH,
                tokenizer_path=Constants.REVIEW_SENTIMENT_ANALYSIS_TOKENIZER_PATH)
            for review in movie_data.public_reviews:
                review.sentiment_score = analyzer.predict(review.content)
            movie_data.save_public_review(Constants.PUBLIC_REVIEW_FOLDER)
            MoviePredictionModel(model_path=Constants.BOX_OFFICE_PREDICTION_MODEL_PATH,
                                 training_setting_path=Constants.BOX_OFFICE_PREDICTION_SETTING_PATH,
                                 transform_scaler_path=Constants.BOX_OFFICE_PREDICTION_SCALER_PATH).simple_predict(
                input_data=movie_data)
        else:
            raise AttributeError("You must specify a movie name.")
    elif args.developer:
        if args.target_epoch:
            main_logger.info("Collecting box office, reviews and training model for prediction.")
            main_logger.info(f"Epoch inputted: {args.target_epoch}.")
            input_epoch: int = int(args.target_epoch)
            BoxOfficeCollector(download_mode=BoxOfficeCollector.Mode.WEEK).download_multiple_box_office_data()
            ReviewCollector(target_website=ReviewCollector.TargetWebsite.PTT).search_review_with_multiple_movie()
            ReviewSentimentAnalyseModel(
                model_path=Constants.REVIEW_SENTIMENT_ANALYSIS_MODEL_PATH,
                tokenizer_path=Constants.REVIEW_SENTIMENT_ANALYSIS_TOKENIZER_PATH).simple_predict(input_data=None)
            MoviePredictionModel().simple_train(input_data=Constants.INDEX_PATH, epoch=input_epoch)
        else:
            raise AttributeError("You must specify value of epoch.")
    elif args.function:
        match args.function:
            case "collect_box_office":
                main_logger.info("Collecting box office.")
                main_logger.info(f"Path inputted: {args.path}.")
                with BoxOfficeCollector(download_mode=BoxOfficeCollector.Mode.WEEK) as collector:
                    collector.download_multiple_box_office_data(
                        input_file_path=Path(args.path) if args.path else None)
            case "collect_ptt_review":
                main_logger.info("Collecting ptt review.")
                target_website: ReviewCollector.TargetWebsite = ReviewCollector.TargetWebsite.PTT
                if args.movie_name:
                    main_logger.info(f"Movie name inputted: {args.movie_name}.")
                    print(
                        ReviewCollector(target_website=target_website).search_review_with_single_movie(args.movie_name))
                else:
                    ReviewCollector(target_website=target_website).search_review_with_multiple_movie()
            case "collect_dcard_review":
                main_logger.info("Collecting dcard review.")
                target_website: ReviewCollector.TargetWebsite = ReviewCollector.TargetWebsite.DCARD
                if args.movie_name:
                    main_logger.info(f"Movie name inputted: {args.movie_name}.")
                    print(
                        ReviewCollector(target_website=target_website).search_review_with_single_movie(args.movie_name))
                else:
                    ReviewCollector(target_website=target_website).search_review_with_multiple_movie()
            case "train_review_sentiment_model":
                if args.target_epoch:
                    main_logger.info('Training review sentiment model.')
                    main_logger.info(f"Epoch inputted: {args.target_epoch}.")
                    input_epoch: int = int(args.target_epoch)
                    ReviewSentimentAnalyseModel().simple_train(
                        input_data=Path("data/review_sentiment_analysis/dataset/review_sentiment_analysis_dataset.csv"),
                        epoch=input_epoch, model_save_name=args.model_name if args.model_name else 'test')
                else:
                    raise AttributeError("You must specify value of epoch.")
            case "test_review_sentiment_model":
                if args.input:
                    main_logger.info('Testing review sentiment model.')
                    main_logger.info(f"Review content inputted: {args.input}.")
                    input_review = args.input
                    default_model_path = Constants.REVIEW_SENTIMENT_ANALYSIS_MODEL_PATH.with_stem('test_10')
                    defaults_tokenizer_path = Constants.REVIEW_SENTIMENT_ANALYSIS_TOKENIZER_PATH
                    print(ReviewSentimentAnalyseModel(model_path=default_model_path,
                                                      tokenizer_path=defaults_tokenizer_path).predict(
                        input_review))

                else:
                    raise AttributeError("You must enter review content.")
            case "add_sentiment_score_to_saved_data":
                main_logger.info("Adding sentiment score to saved data.")
                analyzer: ReviewSentimentAnalyseModel = ReviewSentimentAnalyseModel(
                    model_path=Constants.REVIEW_SENTIMENT_ANALYSIS_MODEL_PATH,
                    tokenizer_path=Constants.REVIEW_SENTIMENT_ANALYSIS_TOKENIZER_PATH)
                for movie in load_index_file():
                    movie.load_public_review()
                    for review in movie.public_reviews:
                        review.sentiment_score = analyzer.predict(review.content)
                    movie.save_public_review(Constants.PUBLIC_REVIEW_FOLDER)

            case "train_movie_prediction_model" | "train_movie_prediction_model_with_randomly_generated_data" | "train_movie_prediction_model_with_checkpointing":
                ml_logger: Logger = LoggingManager().get_logger('machine_learning')
                match args.function:
                    case "train_movie_prediction_model":
                        if args.target_epoch:
                            model_name: str = args.model_name if args.model_name else Constants.BOX_OFFICE_PREDICTION_MODEL_NAME
                            input_epoch: int = int(args.target_epoch)
                            LoggingManager().add_handler(HandlerSettings(
                                name='machine_learning',
                                level=LogLevel.INFO,
                                output=Constants.BOX_OFFICE_PREDICTION_FOLDER / f"{model_name}_{input_epoch}.logs"))
                            LoggingManager().link_handler_to_logger(logger_name='machine_learning',
                                                                    handler_name='machine_learning')
                            ml_logger.info('Training prediction model.')
                            ml_logger.info(f"Target epoch inputted: {args.target_epoch}.")
                            MoviePredictionModel().simple_train(input_data=Constants.INDEX_PATH, epoch=input_epoch,
                                                                model_name=model_name,
                                                                old_model_path=Constants.BOX_OFFICE_PREDICTION_FOLDER.joinpath(
                                                                    args.old_model_name, f"{args.old_model_name}.keras") \
                                                                    if args.old_model_name else None)
                        else:
                            raise AttributeError("You must specify value of epoch.")
                    case "train_movie_prediction_model_with_randomly_generated_data":
                        if args.target_epoch:
                            model_name: str = args.model_name if args.model_name else 'gen_data'
                            input_epoch: int = int(args.target_epoch)
                            LoggingManager().add_handler(HandlerSettings(
                                name='machine_learning',
                                level=LogLevel.INFO,
                                output=Constants.BOX_OFFICE_PREDICTION_FOLDER / f"{model_name}_{input_epoch}.logs"))
                            LoggingManager().link_handler_to_logger(logger_name='machine_learning',
                                                                    handler_name='machine_learning')
                            ml_logger.info('Training prediction model with generated data.')
                            ml_logger.info(f"Target epoch inputted: {args.target_epoch}.")
                            MoviePredictionModel().simple_train(
                                input_data=None, epoch=input_epoch,
                                model_name=model_name,
                                old_model_path=Constants.BOX_OFFICE_PREDICTION_FOLDER.joinpath(args.old_model_name,
                                                                                               f"{args.old_model_name}.keras") \
                                    if args.old_model_name else None)
                        else:
                            raise AttributeError("You must specify value of epoch.")
                    case "train_movie_prediction_model_with_checkpointing":
                        if args.target_epoch and args.saving_epoch:
                            input_epoch: int = int(args.target_epoch)
                            saving_interval: int = int(args.saving_epoch)
                            if args.old_model_name:
                                model_name: str = args.model_name if args.model_name else \
                                    args.old_model_name.rsplit('_', 1)[0]
                                LoggingManager().add_handler(HandlerSettings(
                                    name='machine_learning',
                                    level=LogLevel.INFO,
                                    output=Constants.BOX_OFFICE_PREDICTION_FOLDER / f"{model_name}_{input_epoch}.logs"))
                                LoggingManager().link_handler_to_logger(logger_name='machine_learning',
                                                                        handler_name='machine_learning')
                                ml_logger.info(f"Continue training from model {args.old_model_name}.")
                                if args.model_name:
                                    ml_logger.info(f"New model name: {args.model_name}.")
                                ml_logger.info(f"Target epoch inputted: {args.target_epoch}.")
                                ml_logger.info(f"Saving model every {args.saving_epoch} epoch.")

                                init_epoch: int = int(args.old_model_name.rsplit('_', 1)[1])
                                old_model_path: Optional[Path] = Constants.BOX_OFFICE_PREDICTION_FOLDER.joinpath(
                                    args.old_model_name,
                                    f"{args.old_model_name}.keras")
                            else:
                                model_name: str = args.model_name if args.model_name else Constants.BOX_OFFICE_PREDICTION_MODEL_NAME
                                LoggingManager().add_handler(HandlerSettings(
                                    name='machine_learning',
                                    level=LogLevel.INFO,
                                    output=Constants.BOX_OFFICE_PREDICTION_FOLDER / f"{model_name}_{input_epoch}.logs"))
                                LoggingManager().link_handler_to_logger(logger_name='machine_learning',
                                                                        handler_name='machine_learning')
                                ml_logger.info(f"Training new model.")
                                ml_logger.info(f"Target epoch inputted: {args.target_epoch}.")
                                ml_logger.info(f"Saving model every {args.saving_epoch} epoch.")
                                init_epoch: int = 0
                                old_model_path: Optional[Path] = None
                            MoviePredictionModel().simple_train(input_data=Constants.INDEX_PATH, model_name=model_name,
                                                                epoch=(input_epoch, saving_interval),
                                                                old_model_path=old_model_path)

                        else:
                            raise AttributeError("You must specify value of epoch.")

            case "movie_prediction_model_trend_evaluation" | "movie_prediction_model_range_evaluation" | "test_movie_prediction_model_with_randomly_generated_data":
                model_path = Constants.BOX_OFFICE_PREDICTION_FOLDER.joinpath(args.model_name,
                                                                             f"{args.model_name}.keras") \
                    if args.model_name else Constants.BOX_OFFICE_PREDICTION_MODEL_PATH
                setting_path = Constants.BOX_OFFICE_PREDICTION_FOLDER.joinpath(args.model_name, 'setting.yaml') \
                    if args.model_name else Constants.BOX_OFFICE_PREDICTION_SETTING_PATH
                scaler_path = Constants.BOX_OFFICE_PREDICTION_FOLDER.joinpath(args.model_name, f'scaler.gz') \
                    if args.model_name else Constants.BOX_OFFICE_PREDICTION_SCALER_PATH
                test_data_folder_path = Constants.BOX_OFFICE_PREDICTION_FOLDER.joinpath(args.model_name) \
                    if args.model_name else Constants.BOX_OFFICE_PREDICTION_DEFAULT_MODEL_FOLDER
                match args.function:
                    case "movie_prediction_model_trend_evaluation":
                        main_logger.info('Evaluating prediction model using trend method.')
                        MoviePredictionModel(model_path=model_path, training_setting_path=setting_path,
                                             transform_scaler_path=scaler_path) \
                            .evaluate_trend(test_data_folder_path=test_data_folder_path)
                    case "movie_prediction_model_range_evaluation":
                        main_logger.info('Evaluating prediction model using range method.')
                        MoviePredictionModel(model_path=model_path, training_setting_path=setting_path,
                                             transform_scaler_path=scaler_path) \
                            .evaluate_range(test_data_folder_path=test_data_folder_path)
                    case "test_movie_prediction_model_with_randomly_generated_data":
                        main_logger.info('Testing prediction model with generated data.')
                        MoviePredictionModel(model_path=model_path, training_setting_path=setting_path,
                                             transform_scaler_path=scaler_path) \
                            .simple_predict(input_data=None)

            case "plot_training_loss_curve":
                if args.path:
                    plot_training_loss(Path(args.path))
                else:
                    raise AttributeError("You must specify path for log.")
            case "plot_validation_loss_curve":
                if args.model_name:
                    plot_validation_loss(args.model_name)
                else:
                    raise AttributeError("You must specify model name.")
            case "plot_trend_accuracy_curve":
                if args.model_name:
                    plot_trend_accuracy(args.model_name)
                else:
                    raise AttributeError("You must specify model name.")
            case "plot_range_accuracy_curve":
                if args.model_name:
                    plot_range_accuracy(args.model_name)
                else:
                    raise AttributeError("You must specify model name.")
            case _:
                raise ValueError
    else:
        raise ValueError("Argument error.")

import logging
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser, Namespace
from typing import Optional

from tools.util import recreate_folder
from tools.constant import Constants
from tools.plot import plot_training_loss, plot_validation_loss, \
    plot_trend_accuracy, plot_range_accuracy
from movie_data import load_index_file, MovieData
from web_scraper.review_collector import ReviewCollector
from web_scraper.box_office_collector import BoxOfficeCollector
from machine_learning_model.box_office_prediction import MoviePredictionModel
from machine_learning_model.review_sentiment_analysis import ReviewSentimentAnalyseModel


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


def set_logging_setting(display_level: int, file_path: Path) -> None:
    """
    Sets up the logging configuration for the program.

    Args:
        display_level (int): The logging level to display.
        file_path (Path): The path to the log file.
    """
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=display_level, format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        filename=file_path, filemode='w', encoding='utf-8'
    )
    return


if __name__ == "__main__":
    # setting logging information
    logging_level: int = logging.INFO
    set_logging_setting(
        display_level=logging_level,
        file_path=Path(__file__).resolve(strict=True).parent.
        joinpath("log", f"{datetime.now().strftime('%Y-%m-%dT%H：%M：%S%Z')}_{logging.getLevelName(logging_level)}.log"),
    )

    args = set_argument_parser()

    if args.user:
        if args.movie_name:
            logging.info("collecting box office, reviews and predicting box office next week.")
            logging.info(f"name inputted: {args.movie_name}")
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
            logging.info("collecting box office, reviews and training model for prediction.")
            logging.info(f"epoch inputted: {args.target_epoch}")
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
                logging.info("Collecting box office.")
                logging.info(f"path inputted: {args.path}.")
                with BoxOfficeCollector(download_mode=BoxOfficeCollector.Mode.WEEK) as collector:
                    collector.download_multiple_box_office_data(
                        input_file_path=Path(args.path) if args.path else None)
            case "collect_ptt_review":
                logging.info("Collecting ptt review.")
                target_website: ReviewCollector.TargetWebsite = ReviewCollector.TargetWebsite.PTT
                if args.movie_name:
                    logging.info(f"name inputted: {args.movie_name}.")
                    print(
                        ReviewCollector(target_website=target_website).search_review_with_single_movie(args.movie_name))
                else:
                    ReviewCollector(target_website=target_website).search_review_with_multiple_movie()
            case "collect_dcard_review":
                logging.info("Collecting dcard review.")
                target_website: ReviewCollector.TargetWebsite = ReviewCollector.TargetWebsite.DCARD
                if args.movie_name:
                    logging.info(f"name inputted: {args.movie_name}.")
                    print(
                        ReviewCollector(target_website=target_website).search_review_with_single_movie(args.movie_name))
                else:
                    ReviewCollector(target_website=target_website).search_review_with_multiple_movie()
            case "train_review_sentiment_model":
                if args.target_epoch:
                    logging.info('training sentiment model.')
                    logging.info(f"epoch inputted: {args.target_epoch}")
                    input_epoch: int = int(args.target_epoch)
                    ReviewSentimentAnalyseModel().simple_train(
                        input_data=Path("data/review_sentiment_analysis/dataset/review_sentiment_analysis_dataset.csv"),
                        epoch=input_epoch, model_save_name=args.model_name if args.model_name else 'test')
                else:
                    raise AttributeError("You must specify value of epoch.")
            case "test_review_sentiment_model":
                if args.input:
                    logging.info('testing sentiment model.')
                    logging.info(f"review content inputted: {args.input}")
                    input_review = args.input
                    default_model_path = Constants.REVIEW_SENTIMENT_ANALYSIS_MODEL_PATH.with_stem('test_10')
                    defaults_tokenizer_path = Constants.REVIEW_SENTIMENT_ANALYSIS_TOKENIZER_PATH
                    print(ReviewSentimentAnalyseModel(model_path=default_model_path,
                                                      tokenizer_path=defaults_tokenizer_path).predict(
                        input_review))

                else:
                    raise AttributeError("You must enter review content.")
            case "add_sentiment_score_to_saved_data":
                logging.info("adding sentiment score to saved data.")
                analyzer: ReviewSentimentAnalyseModel = ReviewSentimentAnalyseModel(
                    model_path=Constants.REVIEW_SENTIMENT_ANALYSIS_MODEL_PATH,
                    tokenizer_path=Constants.REVIEW_SENTIMENT_ANALYSIS_TOKENIZER_PATH)
                for movie in load_index_file():
                    movie.load_public_review()
                    for review in movie.public_reviews:
                        review.sentiment_score = analyzer.predict(review.content)
                    movie.save_public_review(Constants.PUBLIC_REVIEW_FOLDER)
            case "train_movie_prediction_model":
                if args.target_epoch:
                    logging.info('training prediction model.')
                    logging.info(f"epoch inputted: {args.target_epoch}")
                    input_epoch: int = int(args.target_epoch)
                    MoviePredictionModel().simple_train(input_data=Constants.INDEX_PATH, epoch=input_epoch,
                                                        model_name=args.model_name if args.model_name else Constants.BOX_OFFICE_PREDICTION_MODEL_NAME,
                                                        old_model_path=Constants.BOX_OFFICE_PREDICTION_FOLDER.joinpath(
                                                            args.old_model_name, f"{args.old_model_name}.keras") \
                                                            if args.old_model_name else None)
                else:
                    raise AttributeError("You must specify value of epoch.")
            case "train_movie_prediction_model_with_randomly_generated_data":
                if args.target_epoch:
                    logging.info('training prediction model with generated data.')
                    logging.info(f"epoch inputted: {args.target_epoch}")
                    input_epoch: int = int(args.target_epoch)
                    MoviePredictionModel().simple_train(
                        input_data=None, epoch=input_epoch,
                        model_name=args.model_name if args.model_name else 'gen_data',
                        old_model_path=Constants.BOX_OFFICE_PREDICTION_FOLDER.joinpath(args.old_model_name,
                                                                                       f"{args.old_model_name}.keras") \
                            if args.old_model_name else None)
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
                        logging.info('evaluating prediction model using trend method.')
                        MoviePredictionModel(model_path=model_path, training_setting_path=setting_path,
                                             transform_scaler_path=scaler_path) \
                            .evaluate_trend(test_data_folder_path=test_data_folder_path)
                    case "movie_prediction_model_range_evaluation":
                        logging.info('evaluating prediction model using range method.')
                        MoviePredictionModel(model_path=model_path, training_setting_path=setting_path,
                                             transform_scaler_path=scaler_path) \
                            .evaluate_range(test_data_folder_path=test_data_folder_path)
                    case "test_movie_prediction_model_with_randomly_generated_data":
                        logging.info('testing prediction model with generated data.')
                        MoviePredictionModel(model_path=model_path, training_setting_path=setting_path,
                                             transform_scaler_path=scaler_path) \
                            .simple_predict(input_data=None)
            case "train_movie_prediction_model_with_checkpointing":
                if args.target_epoch and args.saving_epoch:
                    input_epoch: int = int(args.target_epoch)
                    loop_epoch: int = int(args.saving_epoch)
                    if args.old_model_name:
                        logging.info(f"continue training with model {args.old_model_name}.")
                        logging.info(f"epoch inputted: {args.target_epoch}")
                        logging.info(f"loop epoch inputted: {args.saving_epoch}")
                        model_name: str = args.model_name if args.model_name else args.old_model_name.rsplit('_', 1)[0]
                        init_epoch: int = int(args.old_model_name.rsplit('_', 1)[1])
                        old_model_path: Optional[Path] = Constants.BOX_OFFICE_PREDICTION_FOLDER.joinpath(
                            args.old_model_name,
                            f"{args.old_model_name}.keras")
                    else:
                        logging.info(f"training new model.")
                        logging.info(f"epoch inputted: {args.target_epoch}")
                        logging.info(f"loop epoch inputted: {args.saving_epoch}")
                        model_name: str = args.model_name if args.model_name else Constants.BOX_OFFICE_PREDICTION_MODEL_NAME
                        init_epoch: int = 0
                        old_model_path: Optional[Path] = None
                    for loop_index in range(init_epoch, input_epoch, loop_epoch):
                        if loop_index != init_epoch:
                            old_model_path = Constants.BOX_OFFICE_PREDICTION_FOLDER.joinpath(
                                f"{model_name}_{loop_index}", f"{model_name}_{loop_index}.keras")
                        MoviePredictionModel().simple_train(input_data=Constants.INDEX_PATH, model_name=model_name,
                                                            epoch=loop_epoch,
                                                            old_model_path=old_model_path)

                else:
                    raise AttributeError("You must specify value of epoch.")
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

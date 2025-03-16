import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from datetime import datetime

from web_scraper.box_office_collector import BoxOfficeCollector
from web_scraper.review_collector import ReviewCollector
from machine_learning_model.review_sentiment_analysis import ReviewSentimentAnalyseModel
from movie_data import load_index_file, PublicReview
from tools.util import *
from machine_learning_model.box_office_prediction import MoviePredictionModel


def set_argument_parser() -> Namespace:
    parser: ArgumentParser = ArgumentParser(prog=None, usage=None, description=None, epilog=None)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-u", "--user", action="store_true", help="execute program as a user.")
    group.add_argument("-d", "--developer", action="store_true", help="execute program as a developer.")
    group.add_argument("-f", "--function", type=str,
                       choices=["collect_box_office", "collect_ptt_review", "collect_dcard_review",
                                "review_sentiment_model_train", "review_sentiment_model_test", "movie_prediction_train",
                                "movie_prediction_test", "movie_prediction_train_gen_data",
                                "movie_prediction_test_gen_data"],
                       help="unit test")
    parser.add_argument("-n", "--name", type=str, required=False,
                        help="the movie name that user want to get rating result.")
    parser.add_argument("-i", "--input", type=str, required=False, help="the input of unit test.")

    return parser.parse_args()


def set_logging_setting(display_level: int, file_path: Path) -> None:
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
        if args.name:
            # TODO
            pass
        else:
            raise AttributeError("You must specify a movie name.")
    elif args.developer:
        # TODO
        pass
    elif args.function:
        match args.function:
            case "collect_box_office":
                with BoxOfficeCollector(download_mode=BoxOfficeCollector.Mode.WEEK) as collector:
                    collector.get_box_office_data(input_file_path=Path(args.input) if args.input else None)
            case "collect_ptt_review":
                target_website: ReviewCollector.TargetWebsite = ReviewCollector.TargetWebsite.PTT
                if args.input:
                    print(ReviewCollector(target_website=target_website).search_review_by_single_movie(args.input))
                else:
                    ReviewCollector(target_website=target_website).scrap_train_review_data()
            case "collect_dcard_review":
                target_website: ReviewCollector.TargetWebsite = ReviewCollector.TargetWebsite.DCARD
                if args.input:
                    print(ReviewCollector(target_website=target_website).search_review_by_single_movie(args.input))
                else:
                    ReviewCollector(target_website=target_website).scrap_train_review_data()
            case "review_sentiment_model_train":
                input_epoch: int = int(args.input) if args.input else 1000
                ReviewSentimentAnalyseModel().train(
                    data_path=Path("data/review_sentiment_analysis/dataset/review_sentiment_analysis_dataset.csv"),
                    epoch=input_epoch)
            case "review_sentiment_model_test":
                if args.input:
                    input_review = args.input
                    default_model_path = Constants.REVIEW_SENTIMENT_ANALYSIS_MODEL_PATH
                    defaults_tokenizer_path = Constants.REVIEW_SENTIMENT_ANALYSIS_TOKENIZER_PATH
                    print(ReviewSentimentAnalyseModel(model_path=default_model_path,
                                                      tokenizer_path=defaults_tokenizer_path).test(
                        input_review))
                else:
                    analyzer: ReviewSentimentAnalyseModel = ReviewSentimentAnalyseModel(
                        model_path=Constants.REVIEW_SENTIMENT_ANALYSIS_MODEL_PATH,
                        tokenizer_path=Constants.REVIEW_SENTIMENT_ANALYSIS_TOKENIZER_PATH)
                    for movie in load_index_file():
                        movie.load_public_review()
                        for review in movie.public_reviews:
                            review.sentiment_score = analyzer.test(review.content)
                        movie.save_public_review(Constants.PUBLIC_REVIEW_FOLDER)
            case "movie_prediction_train":
                input_epoch: int = int(args.input) if args.input else 1000
                MoviePredictionModel().movie_train(epoch=input_epoch)
            case "movie_prediction_test":
                # TODO
                pass
            case "movie_prediction_train_gen_data":
                input_epoch: int = int(args.input) if args.input else 1000
                MoviePredictionModel().train_with_auto_generated_data(epoch=input_epoch)
            case "movie_prediction_test_gen_data":
                MoviePredictionModel(model_path=Constants.BOX_OFFICE_PREDICTION_MODEL_PATH,
                                     training_setting_path=Constants.BOX_OFFICE_PREDICTION_SETTING_PATH,
                                     transform_scaler_path=Constants.BOX_OFFICE_PREDICTION_SCALER_PATH).test_with_auto_generated_data()
            case _:
                raise ValueError
    else:
        raise ValueError("Argument error.")

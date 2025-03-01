import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from datetime import datetime

from web_scraper.box_office_collector import BoxOfficeCollector
from web_scraper.review_collector import ReviewCollector
from machine_learning_model.emotion_analyser import EmotionAnalyser


def set_argument_parser() -> Namespace:
    parser: ArgumentParser = ArgumentParser(prog=None, usage=None, description=None, epilog=None)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-u", "--user", action="store_true", help="execute program as a user.")
    group.add_argument("-d", "--developer", action="store_true", help="execute program as a developer.")
    group.add_argument("-t", "--test", type=str,
                       choices=["collect_box_office", "collect_ptt_review", "collect_dcard_review",
                                "train_emotion_analysis", "test_emotion_analysis"],
                       help="unit test with procedure for testing")
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
            pass
        else:
            raise AttributeError("You must specify a movie name.")
    elif args.developer:
        pass
    elif args.test:
        match args.test:
            case "collect_box_office":
                with BoxOfficeCollector(download_mode=BoxOfficeCollector.Mode.WEEK) as collector:
                    collector.get_box_office_data(input_file_path=Path(args.input) if args.input else None)
            case "collect_ptt_review":
                target_website: ReviewCollector.TargetWebsite = ReviewCollector.TargetWebsite.PTT
                if args.input:
                    print(ReviewCollector(target_website=target_website).search_review(args.input))
                else:
                    ReviewCollector(target_website=target_website).scrap_train_review_data()
            case "collect_dcard_review":
                target_website: ReviewCollector.TargetWebsite = ReviewCollector.TargetWebsite.DCARD
                if args.input:
                    print(ReviewCollector(target_website=target_website).search_review(args.input))
                else:
                    ReviewCollector(target_website=target_website).scrap_train_review_data()
            case "train_emotion_analysis":
                input_epoch: int = int(args.input)
                defaults_model_save_folder: Path = Path("./data/emotion_analysis/model")
                defaults_model_save_name: str = "emotion_analysis_model"
                defaults_tokenizer_save_folder: Path = Path("./data/emotion_analysis/dataset")
                defaults_tokenizer_save_name: str = "tokenizer.pickle"
                EmotionAnalyser().train(
                    data_path=Path("./data/emotion_analysis/dataset/emotion_analyse_dataset.csv"),
                    tokenizer_save_folder=defaults_tokenizer_save_folder,
                    tokenizer_save_name=defaults_tokenizer_save_name,
                    model_save_folder=defaults_model_save_folder,
                    model_save_name=defaults_model_save_name,
                    epoch=input_epoch)
            case "test_emotion_analysis":
                input_review = args.input
                default_model_path = Path("./data/emotion_analysis/model/emotion_analysis_model_1000.keras")
                defaults_tokenizer_path = Path("./data/emotion_analysis/dataset/tokenizer.pickle")
                print(EmotionAnalyser(model_path=default_model_path, tokenizer_path=defaults_tokenizer_path).test(
                    input_review))

            case _:
                raise ValueError
    else:
        raise ValueError("Argument error.")

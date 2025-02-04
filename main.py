from box_office_collector import BoxOfficeCollector
from review_collector import ReviewCollector
from movie_review import MovieReview

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from datetime import datetime


def set_argument_parser() -> Namespace:
    parser: ArgumentParser = ArgumentParser(prog=None, usage=None, description=None, epilog=None)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-u", "--user", action="store_true", help="execute program as a user.")
    group.add_argument("-d", "--developer", action="store_true", help="execute program as a developer.")
    group.add_argument("-t", "--test", type=str,
                       choices=["collect_box_office", "collect_ptt_review", "collect_dcard_review"],
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
    logging_level: int = logging.DEBUG
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
        if args.input:
            match args.test:
                case "collect_box_office":
                    input_file_path: str = args.input
                    with BoxOfficeCollector(download_mode=BoxOfficeCollector.Mode.WEEK) as collector:
                        collector.get_box_office_data(input_file_path=input_file_path)
                case "collect_ptt_review":
                    input_title: str = args.input
                    target_website: ReviewCollector.TargetWebsite = ReviewCollector.TargetWebsite.PPT
                    searcher = ReviewCollector(target_website=ReviewCollector.TargetWebsite.DCARD)
                    reviews: list[MovieReview] = searcher.search_review(movie_name=input_title)
                    print(reviews)
                case "collect_dcard_review":
                    input_title: str = args.input
                    target_website: ReviewCollector.TargetWebsite = ReviewCollector.TargetWebsite.DCARD
                    searcher = ReviewCollector(target_website=ReviewCollector.TargetWebsite.DCARD)
                    reviews: list[MovieReview] = searcher.search_review(movie_name=input_title)
                    print(reviews)
                case _:
                    raise ValueError
        else:
            raise ValueError("argumant \"test\" need \"input\" for parameter. ")
    else:
        raise ValueError("Argument error.")

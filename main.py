from movie_weekly_box_office_collector import MovieWeeklyBoxOfficeCollector

import logging
from pathlib import Path
from datetime import datetime


def set_logging_setting(display_level: int, file_path: Path) -> None:
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=display_level, format="%(asctime)s %(filename)s %(levelname)s:%(message)s",
        filename=file_path, filemode='w', encoding='utf-8'
    )
    return


if __name__ == "__main__":
    # setting logging information
    logging_level = logging.DEBUG
    set_logging_setting(
        display_level=logging_level,
        file_path=Path(__file__).resolve(strict=True).parent.
        joinpath("log", f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{logging.getLevelName(logging_level)}.log"),
    )

    # unit test
    input_file_path = "data/input/the_movie_list_of_box_office_10,000,000.csv"
    with MovieWeeklyBoxOfficeCollector(page_changing_waiting_time=2, download_waiting_time=1) as collector:
        collector.get_weekly_box_office_data(csv_file_path=Path(input_file_path))

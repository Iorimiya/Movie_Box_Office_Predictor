from movie_weekly_box_office_collector import MovieWeeklyBoxOfficeCollector

from datetime import datetime
import logging
from pathlib import Path


def set_logging_setting(display_level: int, file_path: Path) -> None:
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=display_level, format="%(asctime)s %(filename)s %(levelname)s:%(message)s",
        filename=file_path, filemode='w', encoding='utf-8'
    )
    return


if __name__ == "__main__":
    logging_level = logging.DEBUG
    set_logging_setting(
        display_level=logging_level,
        file_path=Path(__file__).resolve(strict=True).parent.
        joinpath("log", f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{logging.getLevelName(logging_level)}.log"),
    )

    movie_name = f"影子籃球員-冬季選拔賽總集篇 門的彼端"
    with MovieWeeklyBoxOfficeCollector(page_changing_waiting_time=5, download_waiting_time=3) as collector:
        collector.get_weekly_box_office_data(movie_name=movie_name)

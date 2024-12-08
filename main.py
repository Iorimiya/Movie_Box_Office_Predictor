from box_office_collector import BoxOfficeCollector

import logging
from pathlib import Path
from datetime import datetime


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
        joinpath("log", f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{logging.getLevelName(logging_level)}.log"),
    )

    # unit test
    input_file_path: str = "data/input/the_movie_list_of_box_office_10,000,000.csv"
    with BoxOfficeCollector(page_changing_waiting_time=2, download_waiting_time=1,
                            download_mode=BoxOfficeCollector.Mode.WEEK) as collector:
        collector.get_box_office_data()
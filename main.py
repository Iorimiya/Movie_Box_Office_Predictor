from box_office_collector import BoxOfficeCollector
from format_tranfer_tool import FormatTransferTool

import logging
from enum import Enum
from pathlib import Path
from datetime import datetime


class Mode(Enum):
    COLLECT_BOX_OFFICE = 1
    COLLECT_REVIEW = 2
    TRANSFER_BOX_OFFICE_DATA_FORMAT = 3


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
    operation_mode: Mode = Mode.COLLECT_REVIEW
    # unit test
    if operation_mode == Mode.COLLECT_BOX_OFFICE:
        input_file_path: str = "data/input/the_movie_list_of_box_office_10,000,000.csv"
        with BoxOfficeCollector(page_changing_waiting_time=2, download_waiting_time=1,
                                download_mode=BoxOfficeCollector.Mode.WEEK) as collector:
            collector.get_box_office_data()
    elif operation_mode == Mode.COLLECT_REVIEW:
        pass

    elif operation_mode == Mode.TRANSFER_BOX_OFFICE_DATA_FORMAT:
        input_path = 'data/weekly_box_office_data/by_movie_name'
        output_path = 'data/weekly_box_office_data/all_data.yaml'
        FormatTransferTool(input_path, output_path).transfer_data()

from browser import Browser, TimeoutType
from movie_data import MovieData

import csv
import logging
from enum import Enum
from math import log10
from pathlib import Path
from typing import TypeAlias
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.common import exceptions as selenium_exceptions
from selenium.webdriver.support import expected_conditions as ec
from urllib3.exceptions import ReadTimeoutError

DownloadFinishCondition: TypeAlias = Browser.DownloadFinishCondition
PageChangeCondition: TypeAlias = Browser.PageChangeCondition
WaitingMethodSetting: TypeAlias = Browser.WaitingMethodSetting


class BoxOfficeCollector:
    class Mode(Enum):
        WEEK = 1
        WEEKEND = 2

    class UpdateType(Enum):
        URL = 1
        FILE_PATH = 2

    def __init__(self, download_mode: Mode = Mode.WEEK, page_loading_timeout: float = 30,
                 download_timeout: float = 60) -> None:

        # download mode amd type settings
        self.__download_mode: BoxOfficeCollector.Mode = download_mode
        self.__download_type: str = 'json'
        logging.info(f"use {self.__download_mode.name} mode to download data.")

        # path of folders
        self.__data_path: Path = Path("data")
        self.__box_office_data_folder: Path = self.__data_path.joinpath("box_office")
        self.__download_target_folder: Path = self.__box_office_data_folder.joinpath("by_id")

        # path of files
        self.__index_file_path: Path = self.__data_path.joinpath("index.csv")
        self.__progress_file_path: Path = self.__box_office_data_folder.joinpath("download_progress.csv")
        self.__temporary_file_downloaded_path: Path = self.__data_path.joinpath(
            f"各週{'' if self.__download_mode == self.Mode.WEEK else '週末'}票房資料匯出.{self.__download_type}")

        # initialize path before browser create to avoid resolve error
        self.__initialize_paths()

        # browser setting
        self.__browser: Browser = Browser(download_path=self.__data_path.resolve(strict=True),
                                          page_loading_timeout=page_loading_timeout, download_timeout=download_timeout)
        # url
        self.__searching_url: str = "https://boxofficetw.tfai.org.tw/search/0"
        self.__defaults_url: str = "https://google.com"

        # csv
        self.__input_csv_file_header: str = '片名'
        self.__index_file_header: list[str] = ['id', 'name']
        self.__progress_file_header: list[str] = ['id', 'movie_page_url', 'file_path']
        return

    def __enter__(self) -> any:
        self.__browser = self.__browser.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> any:
        return self.__browser.__exit__(exc_type=exc_type, exc_val=exc_val, exc_tb=exc_tb)

    @staticmethod
    def __read_data_from_csv(path: Path) -> list:
        with open(file=path, mode='r', encoding='utf-8') as file:
            return list(csv.DictReader(file))

    @staticmethod
    def __write_data_to_csv(path: Path, data: list[dict], header: list) -> None:
        with open(file=path, mode='w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()
            writer.writerows(data)
        return

    def __initialize_paths(self) -> None:
        self.__data_path.mkdir(parents=True, exist_ok=True)
        self.__box_office_data_folder.mkdir(parents=True, exist_ok=True)
        self.__download_target_folder.mkdir(parents=True, exist_ok=True)
        return

    def __initialize_index_file(self, input_csv_path: Path) -> None:
        # get movie names from input csv
        with open(file=input_csv_path, mode='r', encoding='utf-8') as file:
            movie_names: list[str] = [row[self.__input_csv_file_header] for row in csv.DictReader(file)]
        # create index file
        self.__index_file_path.touch()
        self.__write_data_to_csv(path=self.__index_file_path,
                                 data=[{self.__index_file_header[0]: index, self.__index_file_header[1]: name} for
                                       index, name in enumerate(movie_names)],
                                 header=self.__index_file_header)
        return

    def __get_movie_data(self) -> list[MovieData]:
        return [MovieData(
            movie_id=int(movie[self.__index_file_header[0]]),
            movie_name=movie[self.__index_file_header[1]])
            for movie in self.__read_data_from_csv(self.__index_file_path)]

    def __initialize_progress_file(self, movie_datas: list[MovieData]):
        self.__progress_file_path.touch()
        self.__write_data_to_csv(path=self.__progress_file_path,
                                 data=[{self.__progress_file_header[0]: movie_data.movie_id,
                                        self.__progress_file_header[1]: '',
                                        self.__progress_file_header[2]: ''} for movie_data in movie_datas],
                                 header=self.__progress_file_header)
        return

    def __update_progress_information(self, index: int, update_type: UpdateType, new_data_value: Path | str):
        # read progress data from csv file
        progress_data = self.__read_data_from_csv(self.__progress_file_path)
        # overwrite new data
        if update_type == self.UpdateType.URL:
            progress_data[index][self.__progress_file_header[1]] = new_data_value
        elif update_type == self.UpdateType.FILE_PATH:
            progress_data[index][self.__progress_file_header[2]] = new_data_value
        # save new data to the same csv file
        self.__write_data_to_csv(path=self.__progress_file_path, header=self.__progress_file_header,
                                 data=progress_data)
        return

    def __navigate_to_movie_page(self, movie_data: MovieData) -> bool:
        movie_name = movie_data.movie_name
        searching_url: str = f"{self.__searching_url}/{movie_name}"
        try:
            # go to search page
            self.__browser.get(url=searching_url)
        except selenium_exceptions.TimeoutException:
            logging.warning("navigate to search url failed.")
            return False
        # find the drop-down list element from page and compare the text of each element and pick the first one matched the movie name
        logging.info(
            f"trying to find the button element which displayed the movie name {movie_name} in drop-down list.")
        target_element = next((button for button in
                               self.__browser.find_elements(by=By.CSS_SELECTOR,
                                                            value='#film-searcher button.result-item') if
                               button.find_element(by=By.CSS_SELECTOR, value="span.name").text == movie_name), None)

        if target_element is None:
            logging.warning(f"Searching {movie_name} failed, none movie title drop-down list found.")
            return False
        logging.info(f"button element of movie {movie_name} found.")
        if not self.__browser.click(button_locator=target_element,
                                    post_method=WaitingMethodSetting(
                                        method=PageChangeCondition(searching_url=searching_url),
                                        error_message="No page changing detect.", timeout=0,
                                        timeout_type=TimeoutType.PAGE_LOADING)):
            return False
        logging.debug(f"goto url: {self.__browser.current_url}")
        self.__update_progress_information(index=movie_data.movie_id,
                                           update_type=self.UpdateType.URL,
                                           new_data_value=self.__browser.current_url)
        return True

    def __click_download_button(self, trying_times: int) -> bool:
        # by defaults, the page is show the weekend data
        if self.__download_mode == self.Mode.WEEK:
            # to use week mode, the additional step is click the "本週" button
            logging.info(f"with download mode is \"WEEK\" mode, trying to click \"本週\" button.")
            week_button_selector = "button#weeks-tab"
            if not self.__browser.click(button_locator=week_button_selector,
                                        pre_method=WaitingMethodSetting(
                                            method=ec.element_to_be_clickable((By.CSS_SELECTOR, week_button_selector)),
                                            error_message="", timeout=0, timeout_type=TimeoutType.PAGE_LOADING)):
                logging.warning("click \"本週\" button failed.")
                return False
        button: WebElement = self.__browser.find_button(
            button_selector_path=f"div#export-button-container button[data-ext='{self.__download_type}']")
        logging.info(f"trying to search download button with {self.__download_type} format.")
        if button:
            if self.__browser.click(
                    button_locator=button,
                    post_method=WaitingMethodSetting(
                        method=DownloadFinishCondition(download_file_path=self.__temporary_file_downloaded_path),
                        error_message="waiting time too long.", timeout=float(3 * trying_times),
                        timeout_type=TimeoutType.DOWNLOAD)):
                return True
        return False

    def __rename_downloaded_file(self, target_file_path: Path, movie_id: int) -> bool:
        self.__temporary_file_downloaded_path.replace(target_file_path)
        if target_file_path.exists():
            # update progress with the path and downloaded flag
            self.__update_progress_information(index=movie_id,
                                               update_type=self.UpdateType.FILE_PATH,
                                               new_data_value=target_file_path)
            return True
        return False

    def __search_box_office_data(self, movie_data: MovieData, progress: dict, trying_times: int = 10,
                                 max_digit: int = 0) -> None:
        movie_name = movie_data.movie_name
        movie_id = movie_data.movie_id
        logging.info(f"Searching box office of {movie_name}.")
        download_target_file_path = self.__download_target_folder.joinpath(
            f"{movie_id:0{max_digit}}.{self.__download_type}")

        if progress[self.__progress_file_header[2]] and download_target_file_path.exists():
            if progress[self.__progress_file_header[1]]:
                logging.info(f"movie url, data file and record in progress correct, skip to next movie,")
            else:
                logging.info(f"movie url not found, search it,")
                for trying_index in range(trying_times):
                    if self.__navigate_to_movie_page(movie_data):
                        break
            return

        for trying_index in range(trying_times):
            current_trying_times = trying_index + 1
            # to avoid the strange error when page switching, go to defaults url for the start
            try:
                self.__browser.get(self.__defaults_url)
            except (ReadTimeoutError, selenium_exceptions.UnexpectedAlertPresentException):
                continue
            # if progress shows the url has been recorded, skip navigating and get it from file.
            if progress[self.__progress_file_header[1]]:
                logging.info(f"only movie url found, download again,")
                self.__browser.get(progress[self.__progress_file_header[1]])
            else:
                logging.info(f"none data found, search and download")
                if not self.__navigate_to_movie_page(movie_data):
                    continue
            if not self.__click_download_button(trying_times=trying_index):
                logging.warning(f"The {current_trying_times} times of searching box office data failed.")
                continue
            if self.__rename_downloaded_file(download_target_file_path, movie_id=movie_id):
                break
            else:
                logging.debug("rename error"),
                logging.warning(f"The {current_trying_times} times of searching box office data failed.")
        return

    def get_box_office_data(self, input_csv_path: str | None = None) -> None:
        # delete previous searching results
        self.__temporary_file_downloaded_path.unlink(missing_ok=True)
        # read index data
        # if not exist, create it from input file
        if not self.__index_file_path.exists():
            self.__initialize_index_file(Path(input_csv_path))
        movie_data = self.__get_movie_data()

        # read_download progress
        # if not exist, create it.
        if not self.__progress_file_path.exists():
            self.__initialize_progress_file(movie_data)
        current_progress = self.__read_data_from_csv(self.__progress_file_path)
        [self.__search_box_office_data(movie_data=movie,
                                       progress=progress,
                                       max_digit=int(log10(len(movie_data))) + 1)
         for movie, progress in zip(movie_data, current_progress)]
        return

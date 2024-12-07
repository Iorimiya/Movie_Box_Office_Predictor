from multiprocessing.util import debug

from colab_browser import ColabBrowser
from movie_data import MovieData

import csv
import time
import logging
from enum import Enum
from math import log10
from pathlib import Path
from selenium.webdriver.common.by import By
from selenium.common import exceptions as selenium_exceptions
from selenium.webdriver.remote.webelement import WebElement
from urllib3.exceptions import ReadTimeoutError


# noinspection PyTypeChecker
class BoxOfficeCollector:
    class DownloadMode(Enum):
        WEEK = 1
        WEEKEND = 2

    class ProgressUpdateType(Enum):
        URL = 1
        FILE_PATH = 2

    def __init__(self, page_changing_waiting_time: float = 2, download_waiting_time: float = 1,
                 download_mode: DownloadMode = DownloadMode.WEEK) -> None:

        # download mode amd type settings
        self.__download_mode: BoxOfficeCollector.DownloadMode = download_mode
        self.__download_type: str = 'json'
        logging.info(f"use {self.__download_mode} mode to download data.")

        # path of folders
        self.__data_path: Path = Path("data")
        self.__box_office_data_folder: Path = self.__data_path.joinpath("box_office")
        self.__download_target_folder: Path = self.__box_office_data_folder.joinpath("by_id")

        # path of files
        self.__index_file_path: Path = self.__data_path.joinpath("index.csv")
        self.__progress_file_path: Path = self.__box_office_data_folder.joinpath("download_progress.csv")
        self.__temporary_file_downloaded_path: Path = self.__data_path.joinpath(
            f"各週{'' if self.__download_mode == self.DownloadMode.WEEK else '週末'}票房資料匯出.{self.__download_type}")

        # initialize path before browser create to avoid resolve error
        self.__initialize_paths()

        # browser setting
        self.__browser: ColabBrowser = ColabBrowser(download_path=self.__data_path.resolve(strict=True))
        self.__page_changing_waiting_time: float = page_changing_waiting_time
        self.__download_waiting_time: float = download_waiting_time
        self.__defaults_download_waiting_time:float = download_waiting_time

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

    def __update_progress_information(self, index: int, update_type: ProgressUpdateType, new_data_value: Path | str):
        # read progress data from csv file
        progress_data = self.__read_data_from_csv(self.__progress_file_path)
        # overwrite new data
        if update_type == self.ProgressUpdateType.URL:
            progress_data[index][self.__progress_file_header[1]] = new_data_value
        elif update_type == self.ProgressUpdateType.FILE_PATH:
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
            self.__browser.get(url=searching_url, waiting_time=self.__page_changing_waiting_time)
        except ReadTimeoutError:
            logging.debug(f"Read Timeout Error on {searching_url} caught.")
            return False
        # find the drop-down list element from page
        buttons: list[WebElement] = self.__browser.find_elements(
            by=By.XPATH,
            value='//div[@id="film-searcher"]/div[@class="body"]/button/span[@class="name"]'
        )
        # compare the text of each element and pick the first one matched the movie name
        target_element: WebElement | None = next(
            (button.find_element(by=By.XPATH, value=f"./..") for button in buttons if
             button.text == movie_name),
            None,
        )
        if target_element is None:
            logging.debug(msg=f"Searching {movie_name} failed, none movie title drop-down list found.")
            return False
        try:
            target_element.click()
            logging.info(msg=f"the drop-down list button of {movie_name} is clicked.")
        except (selenium_exceptions.ElementClickInterceptedException, AttributeError):
            logging.debug(
                msg=f"Searching failed, the drop-down list button of {movie_name} cannot be clicked.",
                exc_info=True)
            return False
        # waiting for the page changing
        time.sleep(self.__page_changing_waiting_time)
        if self.__browser.current_url == self.__searching_url:  # if page not changed
            logging.debug("No page changing detect.")
            return False
        logging.debug(msg=f"goto url: {self.__browser.current_url}")
        #
        self.__update_progress_information(index=movie_data.movie_id,
                                           update_type=self.ProgressUpdateType.URL,
                                           new_data_value=self.__browser.current_url)
        return True

    def __click_download_button(self) -> bool:
        # by defaults, the page is show the weekend data
        if self.__download_mode == self.DownloadMode.WEEK:
            # to use week mode, the additional step is click the "本週" button
            week_box_office_button: WebElement = self.__browser.find_element(by=By.XPATH,
                                                                             value='//button[@id="weeks-tab"]')
            try:
                week_box_office_button.click()
                logging.info(msg=f"weeks-tab button is clicked.")
            except (selenium_exceptions.ElementClickInterceptedException, AttributeError):
                logging.debug(msg=f"Download failed, the weeks-tab button cannot be clicked.",
                              exc_info=True)
                return False
        # find button to download file
        file_download_button: WebElement = self.__browser.find_element(
            by=By.XPATH,
            value=f'//div[@id="export-button-container"]/button[@data-ext="{self.__download_type}"]',
        )
        try:
            file_download_button.click()
            logging.info(msg=f"{self.__download_type.upper()} button is clicked.")
        except (selenium_exceptions.ElementClickInterceptedException, AttributeError):
            logging.debug(
                msg=f"Download failed, the {self.__download_type.upper()} button cannot be clicked.",
                exc_info=True)
            return False
        # waiting until the file downloaded
        time.sleep(self.__download_waiting_time)
        if not self.__temporary_file_downloaded_path.exists():
            logging.debug(f"Download time not enough.")
            time.sleep(30)
            if not self.__temporary_file_downloaded_path.exists():
                logging.debug("waiting time too long.")
                return False
        return True

    def __rename_downloaded_file(self, target_file_path: Path, movie_id: int) -> bool:
        self.__temporary_file_downloaded_path.replace(target_file_path)
        if target_file_path.exists():
            # update progress with the path and downloaded flag
            self.__update_progress_information(index=movie_id,
                                               update_type=self.ProgressUpdateType.FILE_PATH,
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
                logging.info(msg=f"movie url, data file and record in progress correct, skip to next movie,")
            else:
                logging.info(f"movie url not found, search it,")
                self.__navigate_to_movie_page(movie_data)
            return

        for trying_index in range(trying_times):
            current_trying_times = trying_index + 1
            # to avoid the strange error when page switching, go to defaults url for the start
            try:
                self.__browser.get(self.__defaults_url)
            except selenium_exceptions.UnexpectedAlertPresentException:
                logging.debug("Unexpected Alert Caught,")
                continue
            # if progress shows the url has been recorded, skip navigating and get it from file.
            if progress[self.__progress_file_header[1]]:
                logging.info(msg=f"only movie url found, download again,")
                self.__browser.get(progress[self.__progress_file_header[1]])
            else:
                logging.info(f"none data found, search and download")
                self.__navigate_to_movie_page(movie_data)
            if not self.__click_download_button():
                logging.warning(f"The {current_trying_times} times of searching box office data failed.")
                continue
            if self.__rename_downloaded_file(download_target_file_path, movie_id=movie_id):
                break
            else:
                logging, debug("rename error"),
                logging.info(f"The {current_trying_times} times of searching box office data failed.")
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

import json

import re
from logging import Logger
from datetime import datetime
from enum import Enum
from typing import TypeAlias, Final, TypedDict

from selenium.common.exceptions import *
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.expected_conditions import visibility_of_element_located, element_to_be_clickable
from tqdm import tqdm
from urllib3.exceptions import ReadTimeoutError

from web_scraper.browser import Browser
from movie_data import BoxOffice, MovieData, load_index_file
from tools.logging_manager import LoggingManager
from tools.util import *

DownloadFinishCondition: TypeAlias = Browser.DownloadFinishCondition
PageChangeCondition: TypeAlias = Browser.PageChangeCondition
WaitingCondition: TypeAlias = Browser.WaitingCondition


class BoxOfficeCollector:
    """
    A class to collect box office data from a website.
    """
    class Mode(Enum):
        """
        Enum representing the download mode.
        """
        WEEK = 1
        WEEKEND = 2

    class UpdateType(Enum):
        """
        Enum representing the update type for progress information.
        """
        URL = 1
        FILE_PATH = 2

    class ProgressData(TypedDict):
        """
        TypedDict representing the progress data.
        """
        Constants.BOX_OFFICE_DOWNLOAD_PROGRESS_HEADER[0]: int | str
        Constants.BOX_OFFICE_DOWNLOAD_PROGRESS_HEADER[1]: str
        Constants.BOX_OFFICE_DOWNLOAD_PROGRESS_HEADER[2]: str

    def __init__(self, index_file_path: Path = Constants.INDEX_PATH,
                 box_office_data_folder: Path = Constants.BOX_OFFICE_FOLDER,
                 download_mode: Mode = Mode.WEEK, page_loading_timeout: float = 30) -> None:
        """
        Initializes the BoxOfficeCollector.

        Args:
            index_file_path (Path): Path to the index file. Defaults to Constants.INDEX_PATH.
            box_office_data_folder (Path): Path to the box office data folder. Defaults to Constants.BOX_OFFICE_FOLDER.
            download_mode (Mode): Download mode (WEEK or WEEKEND). Defaults to Mode.WEEK.
            page_loading_timeout (float): Page loading timeout in seconds. Defaults to 30.
        """
        # download mode amd type settings
        self.__download_mode: Final[BoxOfficeCollector.Mode] = download_mode
        self.__logger:Logger = LoggingManager().get_logger('root')
        self.__logger.info(f"Using {self.__download_mode.name} mode to download data.")

        # constants
        self.__scrap_file_extension: Final[str] = 'json'
        self.__store_file_extension: Final[str] = Constants.DEFAULT_SAVE_FILE_EXTENSION
        self.__page_loading_timeout: Final[float] = page_loading_timeout
        self.__searching_url: Final[str] = "https://boxofficetw.tfai.org.tw/search/0"
        self.__index_file_header: Final[tuple[str]] = Constants.INDEX_HEADER
        self.__progress_file_header: Final[tuple[str]] = Constants.BOX_OFFICE_DOWNLOAD_PROGRESS_HEADER

        # path
        self.__box_office_data_folder: Final[Path] = box_office_data_folder
        self.__index_file_path: Final[Path] = index_file_path
        self.__progress_file_path: Final[Path] = self.__box_office_data_folder.joinpath("download_progress.csv")
        self.__temporary_file_downloaded_path: Final[Path] = self.__box_office_data_folder.joinpath(
            f"各週{'' if self.__download_mode == self.Mode.WEEK else '週末'}票房資料匯出.{self.__scrap_file_extension}")

        # initialize path before browser create to avoid resolve error
        self.__box_office_data_folder.mkdir(parents=True, exist_ok=True)

        # browser setting
        self.__browser: Browser = Browser(download_path=self.__box_office_data_folder.resolve(strict=True),
                                          page_loading_timeout=self.__page_loading_timeout)

        return

    def __enter__(self) -> any:
        self.__browser = self.__browser.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> any:
        return self.__browser.__exit__(exc_type=exc_type, exc_val=exc_val, exc_tb=exc_tb)

    def __update_progress_information(self, index: int, update_type: UpdateType, new_data_value: Path | str) -> None:
        """
        Updates the progress information in the progress file.

        Args:
            index (int): Index of the movie.
            update_type (UpdateType): Type of update (URL or FILE_PATH).
            new_data_value (Path | str): New data value.
        """
        # read progress data from csv file
        progress_data = read_data_from_csv(self.__progress_file_path)
        # overwrite new data
        if update_type == self.UpdateType.URL:
            progress_data[index][self.__progress_file_header[1]] = new_data_value
        elif update_type == self.UpdateType.FILE_PATH:
            progress_data[index][self.__progress_file_header[2]] = new_data_value
        # save new data to the same csv file
        write_data_to_csv(path=self.__progress_file_path, header=self.__progress_file_header,
                          data=progress_data)
        return

    def __navigate_to_movie_page(self, movie_data: MovieData) -> None:
        """
        Navigates to the movie page on the website.

        Args:
            movie_data (MovieData): Movie data.

        Raises:
            InvalidSwitchToTargetException: If navigation fails.
        """
        movie_name = movie_data.movie_name
        searching_url: str = f"{self.__searching_url}/{movie_name}"
        try:
            # go to search page
            self.__browser.get(url=searching_url)
        except TimeoutException:
            self.__logger.warning("Navigate to search url failed.", exc_info=True)
            raise InvalidSwitchToTargetException
        self.__browser.wait(WaitingCondition(condition=visibility_of_element_located(
            locator=(By.CSS_SELECTOR, '#film-searcher button.result-item')), timeout=5,
            error_message="Searching {movie_name} failed, none movie title drop-down list found."))
        # find the drop-down list element from page and compare the text of each element and pick the first one matched the movie name
        self.__logger.info(
            f"Trying to find the button element which displayed the movie name {movie_name} in drop-down list.")
        target_element = next((button for button in
                               self.__browser.find_elements(by=By.CSS_SELECTOR,
                                                            value='#film-searcher button.result-item') if
                               button.find_element(by=By.CSS_SELECTOR, value="span.name").text == movie_name), None)

        if target_element is None:
            self.__logger.warning(f"Searching {movie_name} failed, none movie title drop-down list found.")
            raise InvalidSwitchToTargetException
        self.__logger.info(f"Button element of movie {movie_name} found.")
        try:
            self.__browser.click(button_locator=target_element,
                                 post_method=WaitingCondition(
                                     condition=PageChangeCondition(searching_url=searching_url),
                                     error_message="No page changing detect.",
                                     timeout=self.__page_loading_timeout))
        except NoSuchElementException:
            raise InvalidSwitchToTargetException
        self.__logger.debug(f"Goto url: \"{self.__browser.current_url}\".")
        if self.__progress_file_path.exists():
            self.__update_progress_information(index=movie_data.movie_id,
                                               update_type=self.UpdateType.URL,
                                               new_data_value=self.__browser.current_url)
        return

    def __click_download_button(self, trying_times: int) -> None:
        """
        Clicks the download button on the page.

        Args:
            trying_times (int): Number of trying times.

        Raises:
            NoSuchElementException: If the download button is not found.
        """
        # by defaults, the page is show the weekend data
        if self.__download_mode == self.Mode.WEEK:
            # to use week mode, the additional step is click the "本週" button
            self.__logger.info(f"With download mode is \"WEEK\" mode, trying to click \"本週\" button.")
            week_button_selector = "button#weeks-tab"

            try:
                self.__browser.click(button_locator=week_button_selector,
                                     pre_method=WaitingCondition(
                                         condition=element_to_be_clickable(
                                             (By.CSS_SELECTOR, week_button_selector)),
                                         error_message="", timeout=self.__page_loading_timeout))
            except NoSuchElementException:
                self.__logger.warning("Click \"本週\" button failed.")
                raise
        self.__logger.info(f"Trying to search download button with {self.__scrap_file_extension} format.")
        try:
            button: WebElement = self.__browser.find_button(
                button_selector_path=f"div#export-button-container button[data-ext='{self.__scrap_file_extension}']")
        except NoSuchElementException:
            self.__logger.warning("Search download button failed.")
            raise
        try:
            self.__browser.click(
                button_locator=button,
                post_method=WaitingCondition(
                    condition=DownloadFinishCondition(download_file_path=self.__temporary_file_downloaded_path),
                    error_message="waiting time too long.", timeout=float(3 * trying_times)))
        except NoSuchElementException:
            raise
        else:
            return

    def __load_box_office_data_from_json_file(self, movie_data: MovieData) -> MovieData:
        """
        Loads box office data from a JSON file.

        Args:
            movie_data (MovieData): Movie data.

        Returns:
            MovieData: Updated movie data.

        Raises:
            TypeError: If an error occurs during data loading.
            ValueError: If box office data is None.
        """
        date_format: Final[str] = '%Y-%m-%d'
        date_split_pattern: Final[str] = '~'
        input_encoding: Final[str] = 'utf-8-sig'
        with open(self.__temporary_file_downloaded_path, mode='r', encoding=input_encoding) as file:
            json_data = json.load(file)
        try:
            weekly_box_office_data: list[BoxOffice] = [BoxOffice(
                start_date=datetime.strptime(re.split(date_split_pattern, week_data["Date"])[0], date_format).date(),
                end_date=datetime.strptime(re.split(date_split_pattern, week_data["Date"])[1], date_format).date(),
                box_office=int(week_data["Amount"]) if week_data["Amount"] is not None else 0)
                for week_data in json_data['Rows']]
        except TypeError:
            self.__logger.debug("An error occurred. See below.", exc_info=True)
            raise
        if not weekly_box_office_data:
            self.__logger.debug("Variable \"box_office_data\" is None.")
            raise ValueError
        movie_data.update_data(box_offices=weekly_box_office_data)
        return movie_data

    def __search_and_download_data(self, movie_data: MovieData, progress: Optional[ProgressData],
                                   trying_times: int = 10) -> None:
        """
        Searches and downloads box office data for a movie.

        Args:
            movie_data (MovieData): Movie data.
            progress (Optional[ProgressData]): Progress data.
            trying_times (int): Number of trying times. Defaults to 10.

        Raises:
            AssertionError: If download fails after multiple tries.
        """
        movie_name = movie_data.movie_name
        movie_id = movie_data.movie_id
        self.__logger.info(f"Searching box office of {movie_name}.")
        download_target_file_path = self.__box_office_data_folder.joinpath(
            f"{movie_id}.{self.__store_file_extension}")

        for trying_time in range(trying_times):
            current_trying_times = trying_time + 1
            # check progress
            if progress and progress[self.__progress_file_header[2]] and download_target_file_path.exists():
                if progress[self.__progress_file_header[1]]:
                    self.__logger.info(f"Movie url, data file and record in progress correct, skip to next movie.")
                    return
                else:
                    self.__logger.info(f"Movie url not found, search it.")
                    try:
                        self.__navigate_to_movie_page(movie_data)
                    except InvalidSwitchToTargetException:
                        continue
                    else:
                        return
            else:
                # to avoid the strange error when page switching, go to defaults url for the start
                try:
                    self.__browser.home()
                except (ReadTimeoutError, UnexpectedAlertPresentException):
                    continue
                # if progress shows the url has been recorded, skip navigating and get it from file.
                if progress and progress[self.__progress_file_header[1]]:
                    self.__logger.info(f"Only movie url found, download again.")
                    self.__browser.get(progress[self.__progress_file_header[1]])
                else:
                    self.__logger.info(f"None data found, search and download.")
                    try:
                        self.__navigate_to_movie_page(movie_data)
                    except InvalidSwitchToTargetException:
                        continue
                try:
                    self.__click_download_button(trying_times=trying_time)
                except NoSuchElementException:
                    self.__logger.warning(f"The {current_trying_times} times of searching box office data failed.")
                    continue
                try:
                    movie_data = self.__load_box_office_data_from_json_file(movie_data)
                except FileNotFoundError:
                    self.__logger.debug("File not found, is it still in downloading?")
                    self.__logger.warning(f"The {current_trying_times} times of searching box office data failed.")
                except (ValueError, TypeError):
                    self.__logger.debug("Load downloaded file failed.")
                    self.__logger.warning(f"The {current_trying_times} times of searching box office data failed.")
                    self.__temporary_file_downloaded_path.unlink()
                    continue
                try:
                    movie_data.save_box_office(self.__box_office_data_folder)
                except TypeError:
                    self.__logger.debug("Cannot save box office data.", exc_info=True)
                    continue
                self.__temporary_file_downloaded_path.unlink()
                if self.__progress_file_path.exists():
                    self.__update_progress_information(index=movie_data.movie_id,
                                                       update_type=self.UpdateType.FILE_PATH,
                                                       new_data_value=self.__box_office_data_folder.joinpath(
                                                           f"{movie_data.movie_id}.yaml"))
                return
        raise AssertionError

    def download_single_box_office_data(self, movie_data: MovieData) -> None:
        """
        Downloads box office data for a single movie.

        Args:
            movie_data (MovieData): Movie data.
        """
        # delete previous searching results
        self.__temporary_file_downloaded_path.unlink(missing_ok=True)
        self.__search_and_download_data(movie_data=movie_data,
                                        progress=None,
                                        trying_times=10)
        movie_data.load_box_office(self.__box_office_data_folder)
        delete_file_path = self.__box_office_data_folder.joinpath(
            f"{movie_data.movie_id}.{self.__store_file_extension}")
        delete_file_path.unlink()
        return

    def download_multiple_box_office_data(self, input_file_path: Optional[Path] = None,
                                          input_csv_file_header: str = Constants.INPUT_MOVIE_LIST_HEADER) -> None:
        """
        Downloads box office data for multiple movies.

        Args:
            input_file_path (Optional[Path]): Path to the input file. Defaults to None.
            input_csv_file_header (str): Header of the input CSV file. Defaults to Constants.INPUT_MOVIE_LIST_HEADER.
        """
        # delete previous searching results
        self.__temporary_file_downloaded_path.unlink(missing_ok=True)
        # read index data
        # if not exist, create it from input file
        if not self.__index_file_path.exists():
            if input_file_path:
                initialize_index_file(CSVFileData(path=input_file_path, header=input_csv_file_header),
                                      CSVFileData(path=self.__index_file_path, header=self.__index_file_header))
            else:
                self.__logger.error("No previous index file, please enter input file path.")
                exit(1)

        movie_data: list[MovieData] = load_index_file(file_path=self.__index_file_path,
                                                      index_header=self.__index_file_header)

        # read_download progress
        # if not exist, create it.
        if not self.__progress_file_path.exists():
            self.__progress_file_path.touch()
            write_data_to_csv(path=self.__progress_file_path,
                              data=[{self.__progress_file_header[0]: single_movie_data.movie_id,
                                     self.__progress_file_header[1]: '',
                                     self.__progress_file_header[2]: ''} for single_movie_data in movie_data],
                              header=self.__progress_file_header)
        current_progress: list = read_data_from_csv(self.__progress_file_path)

        with tqdm(total=len(current_progress), bar_format=Constants.STATUS_BAR_FORMAT) as pbar:
            for movie, progress in zip(movie_data, current_progress):
                try:
                    self.__search_and_download_data(movie_data=movie, progress=progress)
                except AssertionError:
                    pbar.update(1)
                    continue
                else:
                    pbar.update(1)
        return

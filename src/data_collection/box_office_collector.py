from datetime import datetime
from enum import Enum
import json
from logging import Logger
from pathlib import Path
import re
from typing import Final, Optional, TypeAlias, TypedDict

from selenium.common.exceptions import (
    InvalidSwitchToTargetException,
    NoSuchElementException,
    TimeoutException,
    UnexpectedAlertPresentException
)
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.expected_conditions import element_to_be_clickable, visibility_of_element_located
from tqdm import tqdm
from urllib3.exceptions import ReadTimeoutError

from src.core.constants import Constants
from src.core.logging_manager import LoggingManager
from src.data_collection.browser import Browser
from src.data_handling.file_io import CsvFile
from src.data_handling.movie_data import BoxOffice, load_index_file, MovieData
from src.utilities.util import CSVFileData, initialize_index_file

DownloadFinishCondition: TypeAlias = Browser.DownloadFinishCondition
PageChangeCondition: TypeAlias = Browser.PageChangeCondition
WaitingCondition: TypeAlias = Browser.WaitingCondition


class BoxOfficeCollector:
    """
    Collects box office data from a specific website.

    This class handles navigating to movie pages, downloading box office data files,
    parsing these files, and updating movie data objects. It also manages
    download progress.

    :ivar __download_mode: The mode for downloading data (e.g., weekly or weekend).
    :ivar __logger: Logger instance for logging messages.
    :ivar __scrap_file_extension: The file extension of the initially downloaded (scraped) data files.
    :ivar __store_file_extension: The file extension used for storing processed box office data.
    :ivar __page_loading_timeout: Timeout in seconds for page loading operations.
    :ivar __searching_url: The base URL for searching movies on the target website.
    :ivar __index_file_header: The expected header for the movie index CSV file.
    :ivar __progress_file_header: The expected header for the download progress CSV file.
    :ivar __box_office_data_folder: The path to the folder where box office data is stored and downloaded.
    :ivar __index_file_path: The path to the movie index file.
    :ivar __progress_file_path: The path to the download progress tracking file.
    :ivar __temporary_file_downloaded_path: The path where raw downloaded files are temporarily stored.
    :ivar __browser: An instance of the ``Browser`` class for web interactions.
    """

    class Mode(Enum):
        """
        Represents the download mode for box office data.

        :cvar WEEK: Indicates that weekly box office data should be downloaded.
        :cvar WEEKEND: Indicates that weekend box office data should be downloaded.
        """
        WEEK = 1
        WEEKEND = 2

    class UpdateType(Enum):
        """
        Represents the type of information being updated in the progress file.

        :cvar URL: Indicates that the movie's URL on the box office website is being updated.
        :cvar FILE_PATH: Indicates that the local file path of the downloaded box office data is being updated.
        """
        URL = 1
        FILE_PATH = 2

    class ProgressData(TypedDict):
        """
        Represents the structure of a single entry in the download progress data.

        Keys are dynamically set based on ``Constants.BOX_OFFICE_DOWNLOAD_PROGRESS_HEADER``.
        Typically includes movie ID, URL, and file path.
        """
        Constants.BOX_OFFICE_DOWNLOAD_PROGRESS_HEADER[0]: int | str
        Constants.BOX_OFFICE_DOWNLOAD_PROGRESS_HEADER[1]: str
        Constants.BOX_OFFICE_DOWNLOAD_PROGRESS_HEADER[2]: str

    def __init__(self, index_file_path: Path = Constants.INDEX_PATH,
                 box_office_data_folder: Path = Constants.BOX_OFFICE_FOLDER,
                 download_mode: Mode = Mode.WEEK, page_loading_timeout: float = 30) -> None:
        """
        Initializes the BoxOfficeCollector.

        Sets up paths, download mode, logging, and the web browser instance.
        Ensures the box office data folder exists.

        :param index_file_path: Path to the movie index file.
        :param box_office_data_folder: Path to the folder for storing box office data.
        :param download_mode: The mode for downloading data (WEEK or WEEKEND).
        :param page_loading_timeout: Timeout in seconds for web page loading.
        """
        # download mode amd type settings
        self.__download_mode: Final[BoxOfficeCollector.Mode] = download_mode
        self.__logger: Logger = LoggingManager().get_logger('root')
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
        """
        Enters the runtime context for the ``BoxOfficeCollector``, primarily for managing the browser resource.

        This allows the ``BoxOfficeCollector`` to be used with the ``with`` statement,
        ensuring that the browser is properly initialized.

        :returns: The ``BoxOfficeCollector`` instance itself.
        """
        self.__browser = self.__browser.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> any:
        """
        Exits the runtime context, ensuring the browser resource is properly released.

        This is called automatically when exiting a ``with`` statement, and it
        delegates to the browser's ``__exit__`` method to close the browser.

        :param exc_type: The type of the exception that caused the context to be exited, if any.
        :param exc_val: The exception instance that caused the context to be exited, if any.
        :param exc_tb: A traceback object encapsulating the call stack at the point
                       where the exception was raised, if any.
        """
        return self.__browser.__exit__(exc_type=exc_type, exc_val=exc_val, exc_tb=exc_tb)

    def __update_progress_information(self, index: int, update_type: UpdateType, new_data_value: Path | str) -> None:
        """
        Updates a specific field in the download progress CSV file for a given movie index.

        It reads the existing progress data, modifies the specified entry, and saves it back.

        :param index: The 0-based index of the movie in the progress file (corresponds to movie_id if progress file is 0-indexed by movie_id).
        :param update_type: The type of information to update (URL or FILE_PATH).
        :param new_data_value: The new value (URL string or file Path) to write.
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
        Navigates the browser to the specific movie's page on the box office website.

        It first searches for the movie by name, then selects the correct entry
        from the search results drop-down list to go to the movie's detail page.
        Updates the progress file with the movie's page URL if successful.

        :param movie_data: The ``MovieData`` object containing the movie's name and ID.
        :raises InvalidSwitchToTargetException: If navigation to the search page fails,
                                               if the movie is not found in the search results,
                                               or if clicking the movie link fails to change the page.
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
        Locates and clicks the appropriate download button on the movie's box office page.

        If the download mode is 'WEEK', it first clicks a button to switch to weekly data view.
        Then, it clicks the button to download data in the specified scrap file format (e.g., JSON).
        Waits for the download to complete.

        :param trying_times: The current attempt number, used to adjust the download timeout.
        :raises NoSuchElementException: If the '本週' button (for WEEK mode) or the
                                       main download button cannot be found or clicked.
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
        Loads box office data from the downloaded JSON file and updates the ``MovieData`` object.

        Parses the JSON file, extracts weekly box office figures, start dates, and end dates,
        then creates ``BoxOffice`` objects and updates the provided ``movie_data``.

        :param movie_data: The ``MovieData`` object to be updated.
        :returns: The updated ``MovieData`` object with loaded box office data.
        :raises TypeError: If there's an issue parsing dates or amounts from the JSON data,
                           often due to unexpected data format or ``None`` values where numbers are expected.
        :raises ValueError: If the loaded box office data list (``weekly_box_office_data``) is empty after parsing.
        :raises FileNotFoundError: If the temporary downloaded JSON file does not exist.
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
        Manages the process of searching for and downloading box office data for a single movie.

        It checks existing progress, navigates to the movie page if necessary,
        clicks the download button, loads data from the downloaded file,
        saves it in the standard format, and updates progress.
        Includes retry logic for robustness.

        :param movie_data: The ``MovieData`` object for the movie.
        :param progress: The current progress data for this movie from the progress file.
        :param trying_times: The maximum number of attempts to download the data.
        :raises AssertionError: If the download and processing fail after all ``trying_times`` attempts.
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
        Downloads and processes box office data for a single specified movie.

        This method orchestrates the download, data loading, and initial saving.
        It then loads the saved data back into the ``movie_data`` object and cleans up
        the initially saved file, assuming further processing or aggregation might occur elsewhere.
        The temporary downloaded file is also cleaned up.

        :param movie_data: The ``MovieData`` object for which to download box office data.
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
        Downloads box office data for multiple movies listed in an index file.

        It initializes the index file if it doesn't exist (using ``input_file_path`` if provided).
        It also initializes or loads a progress tracking file.
        Then, it iterates through the movies, attempting to download data for each,
        and updates a progress bar.

        :param input_file_path: Optional path to an input CSV file containing movie names,
                                used to initialize the index file if it's missing.
        :param input_csv_file_header: The header name for the movie title column in the
                                      ``input_file_path`` CSV, used if initializing the index.
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

import json
import tempfile
from logging import Logger
from pathlib import Path
from typing import Callable, Final, Literal, Optional, TypeAlias, TypedDict, Tuple

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
from yaml import YAMLError

from src.core.constants import Constants
from src.core.logging_manager import LoggingManager
from src.data_collection.browser import Browser
from src.data_handling.box_office import BoxOffice
from src.data_handling.file_io import CsvFile, YamlFile
from src.data_handling.movie_collections import MovieData
from src.data_handling.movie_metadata import MovieMetadata

DownloadFinishCondition: TypeAlias = Browser.DownloadFinishCondition
PageChangeCondition: TypeAlias = Browser.PageChangeCondition
WaitingCondition: TypeAlias = Browser.WaitingCondition


class BoxOfficeProgressEntry(TypedDict):
    """
    Represents the structure of a single entry in the box office download progress file.

    :ivar id: The unique identifier for the movie.
    :ivar url: The URL of the movie's box office data page.
    :ivar file_path: The local path where the movie's box office data is saved.
    """
    id: int
    url: str
    file_path: str


class BoxOfficeProgressFile(CsvFile):
    """
    Handles read/write operations for the box office download progress CSV file.

    This class extends CsvFile to manage a progress file that tracks the download
    status (URL, saved file path) for each movie.

    :ivar HEADER: A tuple defining the CSV header fields: ('id', 'url', 'file_path').
    """

    HEADER: Final[tuple[str, str, str]] = ('id', 'url', 'file_path')

    def __init__(self, path: Path, encoding: str = Constants.DEFAULT_ENCODING):
        """
        Initializes the BoxOfficeProgressFile handler.

        :param path: The path to the progress CSV file.
        :param encoding: The encoding of the file, defaults to 'utf-8'.
        """
        super().__init__(path=path, encoding=encoding, header=self.HEADER)
        self.__logger: Logger = LoggingManager().get_logger('root')

    def save(self, data: list[BoxOfficeProgressEntry]) -> None:
        """
        Saves a list of progress entries to the CSV file.

        This method writes the provided data, ensuring the parent directory exists.
        It uses the class's predefined HEADER for the CSV field names.

        :param data: A list of ``BoxOfficeProgressEntry`` dictionaries to save.
        """
        super().save(data=data)
        self.__logger.info(f"Successfully saved {len(data)} progress entries to '{self.path}'.")
        return

    def load(self, row_factory: Optional[Callable[[dict[str, str]], any]] = None) -> list[BoxOfficeProgressEntry]:
        """
        Loads and parses data from the progress CSV file.

        This method overrides the parent ``CsvFile.load`` to use a specific internal
        row factory (``_progress_entry_factory``) for converting rows into
        ``BoxOfficeProgressEntry`` dictionaries. A warning is logged if an external
        ``row_factory`` is provided, as it will be ignored.

        :param row_factory: This parameter is ignored. A warning will be logged if it is provided.
        :returns: A list of parsed ``BoxOfficeProgressEntry`` objects. Returns an
                  empty list if the file does not exist or is empty.
        :raises Exception: Propagates exceptions from the underlying CSV loading
                           process, except for ``FileNotFoundError``.
        """
        if not self.path.exists():
            self.__logger.info(f"Progress file not found at '{self.path}'. Returning empty list.")
            return []

        if row_factory is not None:
            self.__logger.warning(
                "BoxOfficeProgressFile.load was called with a 'row_factory' argument, "
                "but it will use its internal '_progress_entry_factory' for conversion."
            )

        try:
            loaded_entries: list[Optional[BoxOfficeProgressEntry]] = super().load(
                row_factory=self._progress_entry_factory)
            processed_data: list[BoxOfficeProgressEntry] = [entry for entry in loaded_entries if entry is not None]
            return processed_data
        except FileNotFoundError:
            self.__logger.error(f"FileNotFoundError during load after exists() check for '{self.path}'.")
            return []
        except Exception as e:
            self.__logger.error(f"Error loading progress file '{self.path}': {e}", exc_info=True)
            raise

    @staticmethod
    def _progress_entry_factory(row: dict[str, str]) -> Optional[BoxOfficeProgressEntry]:
        """
        Converts a raw CSV row into a structured ``BoxOfficeProgressEntry``.

        This factory function validates the input row, ensuring the 'id' field
        exists and is a valid integer. If the row is invalid, it logs a warning
        and returns ``None``.

        :param row: A dictionary representing a single row from the CSV file.
        :returns: A ``BoxOfficeProgressEntry`` instance if the row is valid,
                  otherwise ``None``.
        """
        logger: Logger = LoggingManager().get_logger(
            BoxOfficeProgressFile.__name__)
        movie_id_str: Optional[str] = row.get('id')
        url_str: str = row.get('url', '')
        file_path_str: str = row.get('file_path', '')

        if movie_id_str is None:
            logger.warning(f"Skipping progress entry due to missing 'id': {row}")
            return None

        try:
            movie_id: int = int(movie_id_str)
        except ValueError:
            logger.warning(
                f"Skipping progress entry due to invalid 'id' format: '{movie_id_str}' in {row}")
            return None

        return BoxOfficeProgressEntry(id=movie_id, url=url_str, file_path=file_path_str)

    def initialize_from_movie_metadata(self, movies_metadata: list[MovieMetadata]) -> None:
        """
        Creates and initializes the progress file from a list of movie metadata.

        This method generates an initial progress entry for each movie, setting the
        'id' from the metadata and leaving 'url' and 'file_path' empty.
        It will overwrite the progress file if it already exists.

        :param movies_metadata: A list of ``MovieMetadata`` objects to use for
                                initialization.
        """
        initial_data: list[BoxOfficeProgressEntry] = [
            BoxOfficeProgressEntry(id=movie.id, url='', file_path='') for movie in movies_metadata
        ]
        self.save(data=initial_data)
        self.__logger.info(f"Initialized progress file '{self.path}' with {len(initial_data)} entries.")
        return

    def update_entry(self, movie_id: int, update_field: Literal['url', 'file_path'], new_value: str) -> None:
        """
        Updates a single field for a specific movie entry in the progress file.

        This method reads the entire progress file, finds the entry matching the
        ``movie_id``, modifies the specified ``update_field`` with the ``new_value``,
        and then writes the entire dataset back to the file.

        :param movie_id: The ID of the movie entry to update.
        :param update_field: The name of the field to update (either 'url' or 'file_path').
        :param new_value: The new value to set for the field.
        :raises ValueError: If the ``movie_id`` is not found in the progress file or
                            if ``update_field`` is not a valid field name.
        :raises FileNotFoundError: If the progress file does not exist when an update
                                   is attempted.
        """
        current_progress: list[BoxOfficeProgressEntry] = self.load()
        if not current_progress and not self.exists:
            raise FileNotFoundError(
                f"Progress file '{self.path}' not found. Cannot update entry for movie ID {movie_id}.")

        target_entry: Optional[BoxOfficeProgressEntry] = None
        entry_index: int = -1
        for i, entry in enumerate(current_progress):
            if entry.get('id') == movie_id:
                target_entry = entry
                entry_index = i
                break

        if target_entry is None:
            msg: str = f"Movie ID {movie_id} not found in progress file '{self.path}'. Cannot update."
            self.__logger.error(msg)
            raise ValueError(msg)

        if update_field == 'url':
            current_progress[entry_index]['url'] = new_value
        elif update_field == 'file_path':
            current_progress[entry_index]['file_path'] = new_value
        else:
            invalid_field_msg: str = f"Invalid update_field: '{update_field}'. Must be 'url' or 'file_path'."
            self.__logger.error(invalid_field_msg)
            raise ValueError(invalid_field_msg)

        self.save(data=current_progress)
        self.__logger.debug(f"Updated {update_field} for movie ID {movie_id} in progress file.")
        return


class BoxOfficeCollector:
    """
    Collects box office data from a specific website.


    This class encapsulates the logic for navigating the website, searching for
    movies, and downloading their box office data. It is designed to be used
    as a context manager to ensure the underlying browser instance is properly
    managed.

    It supports two primary use cases:
    1. Fetching data for a single movie and returning it in memory.
    2. Batch downloading data for multiple movies, saving the results to disk
       and tracking progress to allow for resumption.

    :ivar __download_mode: The mode for downloading data (e.g., weekly or weekend).
    :ivar __logger: Logger instance for logging messages.
    :ivar __scrap_file_extension: The file extension of the initially downloaded (scraped) data files.
    :ivar __page_loading_timeout: Timeout in seconds for page loading operations.
    :ivar __SEARCHING_URL: The base URL for searching movies on the target website.
    :ivar __browser: An instance of the ``Browser`` class for web interactions. Initialized in `__enter__`.
    """

    __SEARCHING_URL: Final[str] = "https://boxofficetw.tfai.org.tw/search/0"

    def __init__(self,
                 download_mode: Literal['WEEK', 'WEEKEND'] = 'WEEK',
                 page_loading_timeout: float = 30) -> None:
        """
        Initializes the BoxOfficeCollector.

        :param download_mode: The type of box office data to download ('WEEK' for
                              weekly totals, 'WEEKEND' for weekend totals).
        :param page_loading_timeout: The maximum time in seconds to wait for web
                                     pages to load.
        """
        self.__logger: Logger = LoggingManager().get_logger('root')
        self.__download_mode: Final[Literal['WEEK', 'WEEKEND']] = download_mode
        self.__logger.info(f"Using {self.__download_mode} mode to download data.")
        self.__scrap_file_extension: Final[str] = 'json'
        self.__page_loading_timeout: Final[float] = page_loading_timeout

        self.__browser: Optional[Browser] = None
        return

    def __enter__(self) -> 'BoxOfficeCollector':
        """
        Enters the runtime context, initializing the browser resource.

        This allows the ``BoxOfficeCollector`` to be used with the ``with`` statement.

        :returns: The ``BoxOfficeCollector`` instance itself.
        """
        self.__logger.debug("Entering context, initializing browser...")
        self.__browser = Browser(
            download_path=Path(tempfile.gettempdir()),
            page_loading_timeout=self.__page_loading_timeout
        )
        self.__browser.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exits the runtime context and releases the browser resource.

        This method ensures the underlying Selenium browser instance is properly
        closed and its resources are freed, even if errors occur within the
        ``with`` block.

        :param exc_type: The exception type if an exception was raised in the ``with`` block.
        :param exc_val: The exception value if an exception was raised.
        :param exc_tb: The traceback object if an exception was raised.
        """
        if self.__browser:
            self.__logger.debug("Exiting context, closing browser...")
            self.__browser.__exit__(exc_type, exc_val, exc_tb)
        return

    def _check_browser_active(self) -> None:
        """
        Verifies that the browser has been initialized within a context manager.

        :raises RuntimeError: If the browser is not active.
        """
        if not self.__browser:
            raise RuntimeError(
                "Browser is not initialized. BoxOfficeCollector must be used within a 'with' statement."
            )

    def __navigate_to_movie_page(self, movie_name: str, known_url: Optional[str] = None) -> Optional[str]:
        """
        Navigates to a specific movie's data page on the TFAI website.

        This method first attempts to navigate directly to the ``known_url`` if one
        is provided. If direct navigation fails, is not provided, or leads to a
        redirect, it falls back to searching for the movie by ``movie_name`` and
        clicking the corresponding link in the search results.

        :param movie_name: The name of the movie to find.
        :param known_url: An optional, pre-existing URL for the movie's page to
                          attempt direct navigation.
        :returns: The final URL of the movie's page upon successful navigation,
                  or ``None`` if the page cannot be reached.
        """
        self._check_browser_active()

        # Try direct navigation if a known_url is provided
        if known_url:
            self.__logger.info(f"Attempting direct navigation to known URL for '{movie_name}': {known_url}")
            try:
                self.__browser.get(url=known_url)
                # Wait for the page to load, but don't expect a URL change if we're already there
                # Need to ensure the page content is ready
                self.__browser.wait(method_setting=WaitingCondition(
                    condition=visibility_of_element_located(
                        locator=(By.CSS_SELECTOR, 'div#export-button-container button')),
                    timeout=self.__page_loading_timeout,
                    error_message=f"Direct navigation to '{known_url}' failed to load expected elements."
                ))
                if self.__browser.current_url == known_url:
                    self.__logger.info(f"Successfully navigated directly to '{known_url}'.")
                    return known_url
                else:
                    self.__logger.warning(
                        f"Direct navigation to '{known_url}' resulted in redirect to '{self.__browser.current_url}'. Falling back to search.")
            except TimeoutException as e:
                self.__logger.warning(f"Direct navigation to '{known_url}' timed out: {e}. Falling back to search.")
            except Exception as e:
                self.__logger.warning(f"Error during direct navigation to '{known_url}': {e}. Falling back to search.")

        # Fallback to search-and-click logic
        searching_url: str = f"{self.__SEARCHING_URL}/{movie_name}"
        self.__logger.info(f"Performing search-and-click navigation for '{movie_name}' at '{searching_url}'.")
        try:
            self.__browser.get(url=searching_url)
            self.__browser.wait(method_setting=WaitingCondition(
                condition=visibility_of_element_located(
                    locator=(By.CSS_SELECTOR, '#film-searcher button.result-item')),
                timeout=5,
                error_message=f"Searching '{movie_name}' failed, no movie title drop-down list found."
            ))
        except TimeoutException as e:
            self.__logger.warning(f"Navigate to search url or find dropdown failed for '{movie_name}': {e}")
            return None

        self.__logger.info(
            f"Trying to find the button element which displayed the movie name {movie_name} in drop-down list.")
        target_element: Optional[WebElement] = next(
            (button for button in self.__browser.find_elements(
                by=By.CSS_SELECTOR, value='#film-searcher button.result-item'
            ) if button.find_element(by=By.CSS_SELECTOR, value="span.name").text == movie_name), None)

        if target_element is None:
            self.__logger.warning(f"Searching {movie_name} failed, none movie title drop-down list found.")
            return None

        self.__logger.info(f"Button element of movie {movie_name} found.")
        try:
            self.__browser.click(button_locator=target_element,
                                 post_method=WaitingCondition(
                                     condition=PageChangeCondition(searching_url=searching_url),
                                     error_message="No page changing detect.",
                                     timeout=self.__page_loading_timeout))
        except (NoSuchElementException, TimeoutException) as e:
            self.__logger.warning(f"Clicking movie link for '{movie_name}' failed: {e}")
            return None

        current_url: str = self.__browser.current_url
        self.__logger.debug(f"Goto url: \"{current_url}\".")
        return current_url

    def __click_download_button(self, temp_download_path: Path, trying_times: int) -> None:
        """
        Clicks the data download button and waits for the file to be saved.

        Depending on the collector's ``download_mode``, this method may first click
        the 'WEEK' tab. It then finds and clicks the JSON export button and waits
        for the download to complete by checking for the existence of the target file.
        The wait timeout for the download increases with ``trying_times``.

        :param temp_download_path: The expected full path of the downloaded file.
        :param trying_times: The attempt number for the download, used to calculate
                             an adaptive timeout.
        :raises NoSuchElementException: If a required UI element (e.g., tab or download
                                        button) cannot be found or clicked.
        :raises TimeoutException: If waiting for an element to become clickable or for
                                  the download to finish exceeds the timeout.
        """
        self._check_browser_active()
        if self.__download_mode == 'WEEK':
            self.__logger.info(f"With download mode is \"WEEK\" mode, trying to click \"本週\" button.")
            week_button_selector: str = "button#weeks-tab"
            try:
                self.__browser.click(button_locator=week_button_selector,
                                     pre_method=WaitingCondition(
                                         condition=element_to_be_clickable(
                                             (By.CSS_SELECTOR, week_button_selector)),
                                         error_message="Week tab button not clickable.",
                                         timeout=self.__page_loading_timeout))
            except (NoSuchElementException, TimeoutException) as e:
                self.__logger.warning(f"Clicking '本週' button failed: {e}")
                raise

        self.__logger.info(f"Trying to search download button with {self.__scrap_file_extension} format.")
        download_button_selector: str = f"div#export-button-container button[data-ext='{self.__scrap_file_extension}']"
        try:
            self.__browser.wait(method_setting=WaitingCondition(
                condition=element_to_be_clickable((By.CSS_SELECTOR, download_button_selector)),
                error_message="Download button not clickable.",
                timeout=self.__page_loading_timeout
            ))
            button: WebElement = self.__browser.find_element(by=By.CSS_SELECTOR, value=download_button_selector)
            self.__browser.click(
                button_locator=button,
                post_method=WaitingCondition(
                    condition=DownloadFinishCondition(download_file_path=temp_download_path),
                    error_message="Download did not finish in time.",
                    timeout=float(3 * (trying_times + 1))
                )
            )
        except (NoSuchElementException, TimeoutException) as e:
            self.__logger.warning(f"Searching or clicking download button failed: {e}")
            raise
        return

    def __search_and_fetch_box_office(self,
                                      movie_name: str,
                                      movie_id: Optional[int] = None,
                                      progress_file: Optional[BoxOfficeProgressFile] = None,
                                      trying_times: int = 3) -> \
        Tuple[Optional[list[BoxOffice]], Optional[str]]:
        """
        Handles the complete workflow for fetching a single movie's box office data.

        This method orchestrates the process of navigating to the movie's page
        (optimizing with a known URL from ``progress_file`` if available), clicking
        the download button, and parsing the resulting file. It uses a temporary
        directory for downloads and includes a retry mechanism to handle transient
        network or browser issues.

        :param movie_name: The name of the movie to fetch.
        :param movie_id: The unique ID of the movie, used for looking up the URL in
                         the progress file.
        :param progress_file: An optional progress file handler to read known URLs from.
        :param trying_times: The maximum number of attempts for the entire fetch process.
        :returns: A tuple containing the list of ``BoxOffice`` data objects and the
                  movie's page URL. Both values are ``None`` if all attempts fail.
        """
        self._check_browser_active()
        log_id: str = f" (ID: {movie_id})" if movie_id is not None else ""
        self.__logger.info(f"Fetching box office data for '{movie_name}'{log_id}.")

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir_path: Path = Path(temp_dir_str)

            with self.__browser.temporary_download_path(new_path=temp_dir_path):
                temp_file_stem: str
                match self.__download_mode:
                    case 'WEEK':
                        temp_file_stem = '各週票房資料匯出'
                    case 'WEEKEND':
                        temp_file_stem = '各週週末票房資料匯出'
                    case _:
                        raise ValueError(f"Invalid download_mode: '{self.__download_mode}'.")
                temp_download_file_path: Path = temp_dir_path / f"{temp_file_stem}.{self.__scrap_file_extension}"

                for attempt in range(trying_times):
                    self.__logger.info(f"Attempt {attempt + 1}/{trying_times} for '{movie_name}'.")
                    try:
                        self.__browser.home()

                        # Determine if we have a known URL from progress file (only if progress_file and movie_id are provided)
                        known_url: Optional[str] = None
                        if progress_file and movie_id is not None:
                            progress_entries: list[BoxOfficeProgressEntry] = progress_file.load()
                            current_progress: Optional[BoxOfficeProgressEntry] = next(
                                (p for p in progress_entries if p['id'] == movie_id), None)
                            if current_progress and current_progress.get('url'):
                                known_url = current_progress['url']

                        # Navigate using the potentially known URL
                        movie_url: Optional[str] = self.__navigate_to_movie_page(
                            movie_name=movie_name, known_url=known_url)
                        if not movie_url:
                            continue

                        self.__click_download_button(temp_download_path=temp_download_file_path, trying_times=attempt)

                        box_office_data: list[BoxOffice] = BoxOffice.from_json_file(
                            file_path=temp_download_file_path)
                        self.__logger.info(
                            f"Successfully fetched {len(box_office_data)} entries for '{movie_name}'.")

                        return box_office_data, movie_url

                    except (InvalidSwitchToTargetException, NoSuchElementException, TimeoutException, ReadTimeoutError,
                            UnexpectedAlertPresentException, FileNotFoundError, json.JSONDecodeError,
                            ValueError) as e:
                        self.__logger.warning(
                            f"Attempt {attempt + 1} failed for '{movie_name}': {e}"
                        )
                        if attempt + 1 == trying_times:
                            self.__logger.error(f"All {trying_times} attempts failed for '{movie_name}'.")
                            return None, None
        return None, None

    def download_box_office_data_for_movie(self, movie_name: str) -> list[BoxOffice]:
        """
        Fetches box office data for a single movie and returns it in memory.

        This is a simplified public method for retrieving data for one movie without
        persisting it to disk or using a progress file. It must be called within
        the ``with`` context of the collector.

        :param movie_name: The name of the movie to fetch data for.
        :returns: A list of ``BoxOffice`` objects representing the movie's data.
        :raises RuntimeError: If the data cannot be fetched after multiple attempts.
        """
        self._check_browser_active()

        box_office_data, _ = self.__search_and_fetch_box_office(
            movie_name=movie_name, movie_id=None, progress_file=None)
        if box_office_data is None:
            error_message: str = f"Failed to download box office data for movie '{movie_name}' after multiple attempts."
            self.__logger.error(error_message)
            raise RuntimeError(error_message)
        return box_office_data

    def download_box_office_data_for_movies(self,
                                            multiple_movie_data: list[MovieData],
                                            data_folder: Path) -> None:
        """
        Downloads and saves box office data for a list of movies.

        This method iterates through a list of movies, downloading and saving the
        box office data for each one to the specified ``data_folder``. It uses a
        progress file (``download_progress.csv``) within that folder to track
        completed downloads, allowing the process to be resumed. If a download
        fails for a movie, an empty data file is created to prevent re-attempts.

        :param multiple_movie_data: A list of ``MovieData`` objects to process.
        :param data_folder: The target directory to save the YAML data files and
                            the progress CSV file.
        """
        self._check_browser_active()
        self.__logger.info(f"Starting batch download to '{data_folder}'.")
        data_folder.mkdir(parents=True, exist_ok=True)

        progress_file: BoxOfficeProgressFile = BoxOfficeProgressFile(
            path=data_folder / "download_progress.csv")

        if not progress_file.exists:
            self.__logger.info(f"Progress file '{progress_file.path}' not found. Initializing.")
            progress_file.initialize_from_movie_metadata(
                movies_metadata=[MovieMetadata(id=md.id, name=md.name) for md in multiple_movie_data]
            )

        with tqdm(
            total=len(multiple_movie_data), bar_format=Constants.STATUS_BAR_FORMAT, desc="Collecting Box Office"
        ) as pbar:
            for movie in multiple_movie_data:
                pbar.set_postfix_str(f"Movie: {movie.name[:30]}...", refresh=True)

                # Check progress to see if we can skip based on file_path
                progress_entries: list[BoxOfficeProgressEntry] = progress_file.load()
                progress: Optional[BoxOfficeProgressEntry] = next(
                    (p for p in progress_entries if p['id'] == movie.id), None)
                if progress and progress.get('file_path') and Path(progress['file_path']).exists():
                    self.__logger.info(f"Data for movie ID {movie.id} already exists. Skipping.")
                    pbar.update(1)
                    continue

                # Fetch data, passing the progress_file for potential URL optimization
                box_office_data, movie_url = self.__search_and_fetch_box_office(
                    movie_name=movie.name, movie_id=movie.id, progress_file=progress_file)

                # Persist data and update progress if fetch was successful
                if box_office_data and movie_url:
                    movie.update_box_office(data=box_office_data, update_method='REPLACE')
                    saved_path: Path = movie.save_box_office(target_directory=data_folder)

                    # Update progress file
                    try:
                        progress_file.update_entry(movie_id=movie.id, update_field='url', new_value=movie_url)
                        progress_file.update_entry(movie_id=movie.id, update_field='file_path',
                                                   new_value=str(saved_path))
                        self.__logger.info(f"Box office data for movie ID {movie.id} processed and saved.")
                    except (ValueError, FileNotFoundError) as e:
                        self.__logger.error(f"Failed to update progress for movie ID {movie.id}: {e}")
                else:
                    empty_file_path: Path = data_folder / f"{movie.id}.yaml"
                    self.__logger.warning(
                        f"No box office data found or URL not retrieved for movie ID {movie.id}. "
                        f"Attempting to create empty file and update progress."
                    )
                    try:
                        # Attempt 1: Create empty YAML file
                        YamlFile(path=empty_file_path).save([])
                        self.__logger.info(
                            f"Created empty box office file for movie ID {movie.id} at '{empty_file_path}'.")

                        # Attempt 2: Update progress file (only if file creation succeeded)
                        progress_file.update_entry(movie_id=movie.id, update_field='url',
                                                   new_value=movie_url if movie_url else '')
                        progress_file.update_entry(movie_id=movie.id, update_field='file_path',
                                                   new_value=str(empty_file_path))
                        self.__logger.info(f"Progress file updated for movie ID {movie.id}.")

                    except (OSError, YAMLError) as e:
                        # Catches errors specifically from YamlFile operations (e.g., permissions, disk space)
                        self.__logger.error(
                            f"Error creating empty box office file for movie ID {movie.id} at '{empty_file_path}': {e}",
                            exc_info=True)
                    except (ValueError, FileNotFoundError) as e:
                        # Catches errors specifically from progress_file.update_entry (e.g., movie ID not found in progress file)
                        self.__logger.error(
                            f"Failed to update progress for movie ID {movie.id} after creating empty file: {e}",
                            exc_info=True)
                    except Exception as e:
                        # Catch any other unexpected errors during this block
                        self.__logger.error(
                            f"An unexpected error occurred for movie ID {movie.id} during empty file creation or progress update: {e}",
                            exc_info=True)
                pbar.update(1)
        return

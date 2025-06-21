import json
from logging import Logger
from pathlib import Path
from typing import Callable,Final, Literal, Optional, TypeAlias, TypedDict

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
from src.data_handling.box_office import BoxOffice
from src.data_handling.file_io import CsvFile
from src.data_handling.movie_collections import MovieData
from src.data_handling.movie_metadata import MovieMetadata

DownloadFinishCondition: TypeAlias = Browser.DownloadFinishCondition
PageChangeCondition: TypeAlias = Browser.PageChangeCondition
WaitingCondition: TypeAlias = Browser.WaitingCondition


class BoxOfficeProgressEntry(TypedDict):
    """
    Represents the structure of a single entry in the box office download progress file.
    """
    id: int
    url: str
    file_path: str


class BoxOfficeProgressFile(CsvFile):
    """
    Handles read/write operations and updates for the box office download progress CSV file.

    This file tracks the download status (URL, saved file path) for each movie.
    """

    # Define the expected header explicitly
    HEADER: Final[tuple[str, str, str]] = ('id', 'url', 'file_path')

    def __init__(self, path: Path, encoding: str = Constants.DEFAULT_ENCODING):
        """
        Initializes the BoxOfficeProgressFile handler.

        :param path: The path to the progress CSV file.
        :param encoding: The encoding of the file, defaults to 'utf-8'.
        """
        # Pass the specific header to the CsvFile constructor if needed,
        # or handle header management within the save/load operation in this class.
        # For simplicity, let's assume CsvFile handles header dynamically on save,
        # and DictReader handles it on load. We'll use our HEADER for validation/access.
        super().__init__(path=path, encoding=encoding, header=self.HEADER)
        self.__logger: Logger = LoggingManager().get_logger('root')

    def save(self, data: list[BoxOfficeProgressEntry]) -> None:
        """
        Saves a list of BoxOfficeProgressEntry dictionaries to the progress CSV file.

        Ensures the parent directory exists and uses the defined HEADER for field names.

        :param data: A list of progress entries to save.
        """
        # Convert BoxOfficeProgressEntry (TypedDict) back to list[dict[str, Any]] for CsvFile.save
        # Ensure 'id' is saved as string if CsvFile.save expects string keys/values

        # Ensure header is used explicitly by CsvFile if it doesn't handle it dynamically
        # If CsvFile.save needs fieldnames, you might need to pass self.HEADER
        # Assuming CsvFile.save handles header dynamically based on first dict keys,
        # ensure the first dict in data_to_save has all header keys.
        # A safer CsvFile.save would accept an optional fieldnames parameter.
        super().save(data=data)
        self.__logger.info(f"Successfully saved {len(data)} progress entries to '{self.path}'.")
        return

    def load(self, row_factory: Optional[Callable[[dict[str, str]], any]] = None) -> list[BoxOfficeProgressEntry]:
        """
        Loads data from the progress CSV file into a list of BoxOfficeProgressEntry dictionaries.

        This method overrides CsvFile.load. It always uses its internal `_progress_entry_factory`
        for row conversion, regardless of whether a `row_factory` argument is provided.
        If a `row_factory` is provided, a warning will be logged.

        Ensures the file exists before attempting to load.

        :param row_factory: An optional callable. If provided, it will be ignored and a warning logged,
                            as this method uses a specific internal factory.
        :return: A list of progress entries. Returns an empty list if the file is empty or not found.
        :raises Exception: Propagates exceptions from CsvFile.load() other than FileNotFoundError.
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
            # CsvFile.load now uses the row_factory to convert rows
            loaded_entries: list[Optional[BoxOfficeProgressEntry]] = super().load(row_factory=self._progress_entry_factory)
            # Filter out None entries that failed factory conversion
            processed_data: list[BoxOfficeProgressEntry] = [entry for entry in loaded_entries if entry is not None]
            return processed_data
        except FileNotFoundError: # Should be caught by exists() check, but good for robustness
            self.__logger.error(f"FileNotFoundError during load after exists() check for '{self.path}'.")
            return []
        except Exception as e:
            self.__logger.error(f"Error loading progress file '{self.path}': {e}", exc_info=True)
            raise

    @staticmethod
    def _progress_entry_factory(row: dict[str, str]) -> Optional[BoxOfficeProgressEntry]:
        """
        Factory function to convert a raw CSV row (dict[str, str]) to a BoxOfficeProgressEntry.

        Performs validation and type conversion for the 'id' field.
        Returns None if the row is invalid.

        :param row: A dictionary representing a row from the CSV file.
        :return: A BoxOfficeProgressEntry instance or None if conversion fails.
        """
        logger: Logger = LoggingManager().get_logger(
            BoxOfficeProgressFile.__name__)  # Use a more specific logger if desired
        movie_id_str: Optional[str] = row.get('id')
        url_str: str = row.get('url', '')  # Default to empty string if missing
        file_path_str: str = row.get('file_path', '')  # Default to empty string if missing

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
        Initializes the progress file with entries for a list of movies.

        Each movie gets an entry with its ID, and empty strings for URL and file_path.
        Overwrites the file if it already exists.

        :param movies_metadata: A list of MovieMetadata objects.
        """
        initial_data: list[BoxOfficeProgressEntry] = [
            BoxOfficeProgressEntry(id=movie.id, url='', file_path='') for movie in movies_metadata
        ]
        self.save(data=initial_data)
        self.__logger.info(f"Initialized progress file '{self.path}' with {len(initial_data)} entries.")
        return

    def update_entry(self, movie_id: int, update_field: Literal['url', 'file_path'], new_value: str) -> None:
        """
        Updates a specific field (URL or file_path) for a movie entry in the progress file.

        Loads the current data, finds the entry by movie ID, updates the specified field,
        and saves the data back.

        :param movie_id: The ID of the movie whose entry to update.
        :param update_field: The field to update ('url' or 'file_path').
        :param new_value: The new string value for the field.
        :raises ValueError: If the movie ID is not found in the progress file or if update_field is invalid.
        :raises FileNotFoundError: If the progress file does not exist when trying to load for update.
        """
        current_progress: list[BoxOfficeProgressEntry] = self.load()
        if not current_progress and not self.exists:  # Check if load returned empty due to non-existence
            raise FileNotFoundError(
                f"Progress file '{self.path}' not found. Cannot update entry for movie ID {movie_id}.")

        target_entry: Optional[BoxOfficeProgressEntry] = None
        entry_index: int = -1
        for i, entry in enumerate(current_progress):
            if entry.get('id') == movie_id:  # Use .get for safety, though TypedDict implies key existence
                target_entry = entry
                entry_index = i
                break

        if target_entry is None:
            msg: str = f"Movie ID {movie_id} not found in progress file '{self.path}'. Cannot update."
            self.__logger.error(msg)
            raise ValueError(msg)

        # Directly update the dictionary in the list
        if update_field == 'url':
            current_progress[entry_index]['url'] = new_value
        elif update_field == 'file_path':
            current_progress[entry_index]['file_path'] = new_value
        else:
            # This case should ideally be caught by Literal type checking,
            # but an explicit runtime check is good for robustness.
            invalid_field_msg: str = f"Invalid update_field: '{update_field}'. Must be 'url' or 'file_path'."
            self.__logger.error(invalid_field_msg)
            raise ValueError(invalid_field_msg)

        self.save(data=current_progress)
        self.__logger.debug(f"Updated {update_field} for movie ID {movie_id} in progress file.")
        return


class BoxOfficeCollector:
    """
    Collects box office data from a specific website.

    This class handles navigating to movie pages, downloading box office data files,
    parsing these files, and updating movie data objects. It also manages
    download progress using BoxOfficeProgressFile.

    :ivar __download_mode: The mode for downloading data (e.g., weekly or weekend).
    :ivar __logger: Logger instance for logging messages.
    :ivar __scrap_file_extension: The file extension of the initially downloaded (scraped) data files.
    :ivar __page_loading_timeout: Timeout in seconds for page loading operations.
    :ivar __SEARCHING_URL: The base URL for searching movies on the target website.
    :ivar __box_office_data_folder: The path to the folder where box office data is stored and downloaded.
    :ivar __progress_file: An instance of BoxOfficeProgressFile for managing download progress.
    :ivar __temporary_file_downloaded_path: The path where raw downloaded files are temporarily stored.
    :ivar __browser: An instance of the ``Browser`` class for web interactions.
    """

    __SEARCHING_URL: Final[str] = "https://boxofficetw.tfai.org.tw/search/0"

    def __init__(self, box_office_data_folder: Path, download_mode: Literal['WEEK', 'WEEKEND'] = 'WEEK',
                 page_loading_timeout: float = 30) -> None:
        """
        Initializes the BoxOfficeCollector.

        Sets up paths, download mode, logging, and the web browser instance.
        Ensures the box office data folder exists.

        :param box_office_data_folder: Path to the folder for storing box office data.
        :param download_mode: The mode for downloading data (WEEK or WEEKEND).
        :param page_loading_timeout: Timeout in seconds for web page loading.
        """

        self.__logger: Logger = LoggingManager().get_logger('root')

        # download mode amd type settings
        self.__download_mode: Final[Literal['WEEK','WEEKEND']] = download_mode
        self.__logger.info(f"Using {self.__download_mode} mode to download data.")

        # constants
        self.__scrap_file_extension: Final[str] = 'json'
        self.__page_loading_timeout: Final[float] = page_loading_timeout

        # path
        self.__box_office_data_folder: Final[Path] = box_office_data_folder
        self.__progress_file: Final[BoxOfficeProgressFile] = BoxOfficeProgressFile(
            path=self.__box_office_data_folder / "download_progress.csv")

        temp_file_stem: str
        match self.__download_mode:
            case 'WEEK':
                temp_file_stem = '各週票房資料匯出'
            case 'WEEKEND':
                temp_file_stem = '各週週末票房資料匯出'
            case _:
                raise ValueError(f"Invalid download_mode: '{self.__download_mode}'. Must be 'WEEK' or 'WEEKEND'.")

        self.__temporary_file_downloaded_path: Final[Path] = self.__box_office_data_folder.joinpath(
            f"{temp_file_stem}.{self.__scrap_file_extension}")

        # initialize path before browser create to avoid resolve error
        self.__box_office_data_folder.mkdir(parents=True, exist_ok=True)

        # browser setting
        self.__browser: Browser = Browser(download_path=self.__box_office_data_folder.resolve(strict=True),
                                          page_loading_timeout=self.__page_loading_timeout)

        return

    def __enter__(self) -> 'BoxOfficeCollector':
        """
        Enters the runtime context for the ``BoxOfficeCollector``, primarily for managing the browser resource.

        This allows the ``BoxOfficeCollector`` to be used with the ``with`` statement,
        ensuring that the browser is properly initialized.

        :returns: The ``BoxOfficeCollector`` instance itself.
        """
        self.__browser = self.__browser.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
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

    def __update_progress_information(self, movie_id: int,
                                      update_type: Literal['url', 'file_path'],
                                      new_data_value: str) -> None:
        """
        Updates a specific field in the download progress CSV file for a given movie ID.

        Delegates the update logic to the BoxOfficeProgressFile handler.

        :param movie_id: The ID of the movie whose entry to update.
        :param update_type: The type of information to update ('url' or 'file_path').
        :param new_data_value: The new string value for the field.
        """
        try:
            self.__progress_file.update_entry(
                movie_id=movie_id, update_field=update_type, new_value=str(new_data_value)
            )
        except ValueError as e:
            self.__logger.error(f"Failed to update progress for movie ID {movie_id}: {e}")
        except FileNotFoundError as e:  # Catch FileNotFoundError from update_entry
            self.__logger.error(f"Progress file not found when trying to update for movie ID {movie_id}: {e}")
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
        movie_name = movie_data.name
        searching_url: str = f"{self.__SEARCHING_URL}/{movie_name}"
        try:
            # go to search page
            self.__browser.get(url=searching_url)
        except TimeoutException:
            self.__logger.warning("Navigate to search url failed.", exc_info=True)
            raise InvalidSwitchToTargetException
        try:
            self.__browser.wait(method_setting=WaitingCondition(
                condition=visibility_of_element_located(
                    locator=(By.CSS_SELECTOR, '#film-searcher button.result-item')),
                timeout=5,
                error_message=f"Searching '{movie_name}' failed, no movie title drop-down list found."
            ))
        except TimeoutException as e:  # Catch the specific error from wait
            self.__logger.warning(str(e))
            raise InvalidSwitchToTargetException from e

        # find the drop-down list element from page and compare the text of each element and pick the first one matched the movie name
        self.__logger.info(
            f"Trying to find the button element which displayed the movie name {movie_name} in drop-down list.")
        target_element: Optional[WebElement] = next(
            (button for button in self.__browser.find_elements(
                by=By.CSS_SELECTOR, value='#film-searcher button.result-item'
            ) if button.find_element(by=By.CSS_SELECTOR, value="span.name").text == movie_name), None)

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
        except (NoSuchElementException, TimeoutException) as e:
            self.__logger.warning(f"Clicking movie link for '{movie_name}' failed: {e}")
            raise InvalidSwitchToTargetException from e
        self.__logger.debug(f"Goto url: \"{self.__browser.current_url}\".")
        self.__update_progress_information(movie_id=movie_data.id,
                                           update_type='url',
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
        :raises TimeoutException: If waiting for an element or download times out.
        """
        # by defaults, the page is show the weekend data
        if self.__download_mode == 'WEEK':
            # to use week mode, the additional step is click the "本週" button
            self.__logger.info(f"With download mode is \"WEEK\" mode, trying to click \"本週\" button.")
            week_button_selector = "button#weeks-tab"

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
                    condition=DownloadFinishCondition(download_file_path=self.__temporary_file_downloaded_path),
                    error_message="Download did not finish in time.",
                    timeout=float(3 * (trying_times + 1))  # Ensure trying_times starts from 0 for multiplier
                )
            )
        except (NoSuchElementException, TimeoutException) as e:
            self.__logger.warning(f"Searching or clicking download button failed: {e}")
            raise

        return

    def __load_box_office_data_from_json_file(self, movie_data: MovieData) -> MovieData:
        """
        Loads box office data from the downloaded JSON file using BoxOffice.from_json_file
        and updates the ``MovieData`` object.

        :param movie_data: The ``MovieData`` object to be updated.
        :returns: The updated ``MovieData`` object with loaded box office data.
        :raises FileNotFoundError: If the temporary downloaded JSON file does not exist.
        :raises json.JSONDecodeError: If the JSON file is malformed.
        :raises ValueError: If the JSON 'Rows' data is missing, empty, all items are malformed,
                           or if data conversion within BoxOffice.create_multiple fails.
        """
        self.__logger.info(f"Loading box office data from JSON file: {self.__temporary_file_downloaded_path}")
        try:
            weekly_box_office_data: list[BoxOffice] = BoxOffice.from_json_file(
                file_path=self.__temporary_file_downloaded_path
            )
            movie_data.update_box_office(data=weekly_box_office_data, update_method='REPLACE')
            self.__logger.info(
                f"Successfully loaded and updated {len(weekly_box_office_data)} box office entries for movie ID {movie_data.id}."
            )
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            # Log the specific error and re-raise to be handled by the calling method (__search_and_download_data)
            self.__logger.error(
                f"Failed to load box office data from {self.__temporary_file_downloaded_path} for movie ID {movie_data.id}: {e}"
            )
            # Ensure movie_data.box_office is cleared or handled if partial data might have been set before error
            movie_data.update_box_office(data=[], update_method='REPLACE')
            raise
        return movie_data

    def __search_and_download_data(self, movie_data: MovieData, trying_times: int = 10) -> None:
        """
        Manages the process of searching for and downloading box office data for a single movie.

        It checks existing progress, navigates to the movie page if necessary,
        clicks the download button, loads data from the downloaded file,
        saves it in the standard format, and updates progress.
        Includes retry logic for robustness.

        :param movie_data: The ``MovieData`` object for the movie.
        :param trying_times: The maximum number of attempts to download the data.
        :raises AssertionError: If the download and processing fail after all ``trying_times`` attempts.
        """
        movie_name: str = movie_data.name
        movie_id: int = movie_data.id
        self.__logger.info(f"Searching box office of '{movie_name}' (ID: {movie_id}).")
        download_target_file_path: Path = self.__box_office_data_folder.joinpath(
            f"{movie_id}.{Constants.DEFAULT_SAVE_FILE_EXTENSION}")

        all_progress_entries: list[BoxOfficeProgressEntry] = self.__progress_file.load()
        current_movie_progress: Optional[BoxOfficeProgressEntry] = next(
            (entry for entry in all_progress_entries if entry.get('id') == movie_id), None
        )

        for trying_time_idx in range(trying_times):
            current_attempt: int = trying_time_idx + 1
            self.__logger.info(f"Attempt {current_attempt}/{trying_times} for movie ID {movie_id} ('{movie_name}').")
            # check progress
            if current_movie_progress and current_movie_progress.get('file_path') and \
                Path(current_movie_progress['file_path']).exists():
                if current_movie_progress.get('url'):
                    self.__logger.info(
                        f"Movie URL and data file already recorded and exist for ID {movie_id}. Skipping download."
                    )
                    if not movie_data.box_office:
                        movie_data.load_box_office(target_directory=self.__box_office_data_folder)
                    return
                else:
                    self.__logger.info(f"Movie url not found in progress for ID {movie_id}, search it.")
                    try:
                        self.__navigate_to_movie_page(movie_data=movie_data)
                        # Ensure movie_data has the loaded data
                        if not movie_data.box_office:
                            movie_data.load_box_office(target_directory=self.__box_office_data_folder)
                        return  # Success after getting URL
                    except InvalidSwitchToTargetException:
                        self.__logger.warning(
                            f"Failed to navigate for ID {movie_id} on attempt {current_attempt}. Retrying..."
                        )
                        if current_attempt == trying_times: break  # Break if last attempt
                        continue  # To next attempt

                # to avoid the strange error when page switching, go to defaults url for the start
            try:
                self.__browser.home()
                # if progress shows the url has been recorded, skip navigating and get it from file.
                if current_movie_progress and current_movie_progress.get('url'):
                    self.__logger.info(f"Only movie url found in progress for ID {movie_id}, download again.")
                    self.__browser.get(url=current_movie_progress['url'])
                else:
                    self.__logger.info(f"None data found for ID {movie_id}, search and download.")
                    self.__navigate_to_movie_page(movie_data=movie_data)
                self.__click_download_button(trying_times=trying_time_idx)
                movie_data = self.__load_box_office_data_from_json_file(movie_data)
                movie_data.save_box_office(self.__box_office_data_folder)
                self.__update_progress_information(
                    movie_id=movie_id,
                    update_type='file_path',
                    new_data_value=str(download_target_file_path)
                )
            except (InvalidSwitchToTargetException, NoSuchElementException, TimeoutException, ReadTimeoutError,
                    UnexpectedAlertPresentException) as e_web:
                self.__logger.warning(
                    f"Web interaction failed for ID {movie_id} on attempt {current_attempt}: {e_web}"
                )
                self.__temporary_file_downloaded_path.unlink(missing_ok=True)  # Clean up if download started
            except FileNotFoundError as e_file:  # From __load_box_office_data_from_json_file
                self.__logger.warning(
                    f"Downloaded file not found for ID {movie_id} on attempt {current_attempt}: {e_file}"
                )
            except (
                ValueError, TypeError, json.JSONDecodeError) as e_parse:  # From __load_box_office_data_from_json_file
                self.__logger.warning(
                    f"Parsing downloaded file failed for ID {movie_id} on attempt {current_attempt}: {e_parse}"
                )
                self.__temporary_file_downloaded_path.unlink(missing_ok=True)  # Clean corrupted downloaded file
            except Exception as e_general:  # Catch other unexpected errors during the process
                self.__logger.error(
                    f"An unexpected error occurred for ID {movie_id} on attempt {current_attempt}: {e_general}",
                    exc_info=True)
                self.__temporary_file_downloaded_path.unlink(missing_ok=True)
            finally:
                self.__temporary_file_downloaded_path.unlink()
            if current_attempt == trying_times:  # If this was the last attempt
                break
        raise AssertionError(
            f"Failed to download and process box office data for movie ID {movie_id} ('{movie_name}') after {trying_times} attempts.")

    def download_single_box_office_data(self, movie_data: MovieData) -> None:
        """
        Downloads and processes box office data for a single specified movie.

        This method orchestrates the download, data loading, and saving of box office data.
        The temporary downloaded file is cleaned up. The final processed data remains
        in the `movie_data` object and is saved to its standard YAML file.

        :param movie_data: The ``MovieData`` object for which to download box office data.
                           This object will be updated with the new data.
        """
        # delete previous searching results
        self.__temporary_file_downloaded_path.unlink(missing_ok=True)
        try:
            self.__search_and_download_data(movie_data=movie_data, trying_times=10)
            # After __search_and_download_data, movie_data is updated and saved.
            # No need to load it again here unless __search_and_download_data's contract changes.
            # The old logic of loading and then deleting the YAML seemed redundant if the goal
            # is to have the data in movie_data and saved to its standard location.
            self.__logger.info(f"Box office data for movie ID {movie_data.id} processed and saved.")
        except AssertionError as e:
            self.__logger.error(f"Failed to download single box office data for movie ID {movie_data.id}: {e}")
            # Optionally re-raise or handle as per application requirements
        finally:
            self.__temporary_file_downloaded_path.unlink(missing_ok=True)  # Ensure cleanup
        return

    def download_multiple_box_office_data(self, multiple_movie_data: list[MovieData])->None:
        self.__temporary_file_downloaded_path.unlink(missing_ok=True)
        # read_download progress
        # if not exist, create it.
        if not self.__progress_file.exists:  # Use the 'exists' property
            self.__logger.info(f"Progress file '{self.__progress_file.path}' not found. Initializing.")
            # Ensure that MovieData can be used where MovieMetadata is expected,
            # or convert if necessary. Assuming MovieData is a subclass or compatible.
            self.__progress_file.initialize_from_movie_metadata(
                movies_metadata=[MovieMetadata(id=md.id, name=md.name) for md in multiple_movie_data]
            )

        with tqdm(
            total=len(multiple_movie_data), bar_format=Constants.STATUS_BAR_FORMAT, desc="Collecting Box Office"
        ) as pbar:
            for movie in multiple_movie_data:
                pbar.set_postfix_str(f"Movie: {movie.name[:30]}...", refresh=True)
                try:
                    self.__search_and_download_data(movie_data=movie)
                except AssertionError:
                    pass
                finally:
                    pbar.update(1)
        return

    # TODO: multiple時才需要建立ProgressFile，single時使用temp資料夾下載即可
    # TODO: multiple輸出None並存入檔案內，single輸出BoxOffice物件(ask 潔米奈)

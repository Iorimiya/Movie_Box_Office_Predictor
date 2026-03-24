import json
import tempfile
from logging import Logger
from pathlib import Path
from typing import Final, Literal, Optional, Tuple, TypeAlias
from urllib.parse import quote

from selenium.common.exceptions import (
    InvalidSwitchToTargetException,
    NoSuchElementException,
    TimeoutException,
    UnexpectedAlertPresentException
)
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.expected_conditions import (
    element_to_be_clickable,
    text_to_be_present_in_element,
    visibility_of_element_located
)
from urllib3.exceptions import ReadTimeoutError

from src.core.logging_manager import LoggingManager
from src.data_collection.browser import Browser, ElementLocator
from src.data_handling.box_office import BoxOffice

DownloadFinishCondition: TypeAlias = Browser.DownloadFinishCondition
PageChangeCondition: TypeAlias = Browser.PageChangeCondition
WaitingCondition: TypeAlias = Browser.WaitingCondition


class BoxOfficeCollector:
    """
    Collects box office data for a single movie from a specific website.

    This class encapsulates the logic for navigating the website, searching for
    a movie, and downloading its box office data. It is designed to be used
    as a context manager to ensure the underlying browser instance is properly
    managed.

    The primary method, `fetch_single_movie_data`, handles the entire
    process for one movie and returns the data in memory.

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
                 page_loading_timeout: float = 30,
                 headless: bool = False) -> None:
        """
        Initializes the BoxOfficeCollector.

        :param download_mode: The type of box office data to download ('WEEK' for
                              weekly totals, 'WEEKEND' for weekend totals).
        :param page_loading_timeout: The maximum time in seconds to wait for web
                                     pages to load.
        :param headless: If ``True`` (default), runs the underlying browser in
                         headless mode. Set to ``False`` to run with a visible GUI.
        """
        self.__logger: Logger = LoggingManager().get_logger('root')
        self.__download_mode: Final[Literal['WEEK', 'WEEKEND']] = download_mode
        self.__logger.info(f"Using {self.__download_mode} mode to download data.")
        self.__scrap_file_extension: Final[str] = 'json'
        self.__page_loading_timeout: Final[float] = page_loading_timeout
        self.__headless: Final[bool] = headless

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
            headless=self.__headless,
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

        def _get_safe_xpath_text_query(text_to_match: str) -> str:
            """
            Generates a safe XPath query string for matching text content.

            This method handles cases where the text contains single quotes, double quotes,
            or both, by constructing a robust `concat()` expression. This prevents
            InvalidSelectorException when building dynamic XPaths.

            :param text_to_match: The string to be safely embedded in an XPath text() query.
            :returns: A safe XPath expression part, e.g., "text()=concat('a', \"'\", 'b')".
            """
            if "'" not in text_to_match:
                # If no single quotes, we can safely use them to wrap the text.
                return f"text()='{text_to_match}'"

            if '"' not in text_to_match:
                # If no double quotes, we can safely use them to wrap the text.
                return f'text()="{text_to_match}"'

            # If both single and double quotes are present, use concat().
            # We split the string by single quotes and then join them back,
            # inserting the single quote as a separate literal string "'".
            parts: list[str] = text_to_match.split("'")

            # The structure is concat('part1', "'", 'part2', "'", 'part3', ...)
            # Each part is wrapped in single quotes. The single quote itself is a string literal: "'"
            # which in Python needs to be escaped as "'\"'\"'"
            concat_parts: str = ", \"'\", ".join([f"'{part}'" for part in parts])

            return f"concat({concat_parts})"

        def _ensure_search_results_visible() -> None:
            """
            A nested helper to wait for the search result dropdown or click the search button as a fallback.
            It leverages the 'searching_url' and 'movie_name' from the parent scope.
            """
            try:
                self.__browser.wait(method_setting=WaitingCondition(
                    condition=visibility_of_element_located(
                        locator=(By.CSS_SELECTOR, '#film-searcher button.result-item')),
                    timeout=5,
                    error_message=f"Searching '{movie_name}' failed, no movie title drop-down list found."
                ))
            except TimeoutException:
                # Fallback: try clicking the main search button
                self.__logger.debug("Dropdown not visible, attempting to click main search button.")
                search_button_xpath: Final[str] = "//section[@id='search-bar']//button[@type='submit']"
                # This will raise TimeoutException on failure, which is caught by the outer try-except
                self.__browser.click(
                    button_locator=ElementLocator(by=By.XPATH, value=search_button_xpath),
                    post_method=WaitingCondition(
                        condition=visibility_of_element_located(
                            locator=(By.CSS_SELECTOR, '#film-searcher button.result-item')),
                        error_message="Dropdown still not visible after clicking search button.",
                        timeout=5
                    )
                )

        self._check_browser_active()

        movie_name_locator: tuple[str, str] = (By.CSS_SELECTOR, "#film-banner .name")

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
        # Encode the movie name to handle all special URL characters like '/', '?', '='.
        encoded_movie_name: str = quote(movie_name)
        searching_url: str = f"{self.__SEARCHING_URL}/{encoded_movie_name}"
        self.__logger.info(f"Performing search-and-click navigation for '{movie_name}' at '{searching_url}'.")
        try:
            self.__browser.get(url=searching_url)
            _ensure_search_results_visible()
        except TimeoutException as e:
            self.__logger.warning(f"Navigate to search url or find dropdown failed for '{movie_name}': {e}")
            return None

        self.__logger.info(f"Finding all candidate buttons for movie '{movie_name}' in the drop-down list.")
        # Get all candidate buttons using a single, efficient XPath query
        safe_text_query: str = _get_safe_xpath_text_query(movie_name)
        movie_button_xpath: str = f"//button[contains(@class, 'result-item')][.//span[@class='name' and {safe_text_query}]]"
        try:
            initial_candidate_elements: list[WebElement] = \
                self.__browser.find_elements(by=By.XPATH, value=movie_button_xpath)
            num_candidates: int = len(initial_candidate_elements)
        except NoSuchElementException:
            num_candidates = 0

        # Check if any candidates were found
        if num_candidates == 0:
            self.__logger.warning(
                f"Searching '{movie_name}' failed, no matching movie title found in the drop-down list.")
            return None

        self.__logger.info(
            f"Found {num_candidates} potential candidate(s) for '{movie_name}'. Iterating to find a valid page.")

        # Start iterating through candidates
        for i in range(num_candidates):
            self.__logger.info(f"Attempting to process candidate {i + 1}/{num_candidates} for '{movie_name}'.")
            # On subsequent attempts (i > 0), we must navigate back to the search page first.
            if i > 0:
                self.__logger.info("Returning to search page to try the next candidate.")
                try:
                    self.__browser.get(url=searching_url)
                    _ensure_search_results_visible()
                except TimeoutException as e:
                    self.__logger.error(
                        f"Critical failure: Could not navigate back to search page. Aborting for '{movie_name}'. Error: {e}")
                    return None

            try:
                # Re-fetch the list of candidates to get fresh element references.
                # This is the crucial step to prevent StaleElementReferenceException.
                current_candidate_elements: list[WebElement] = self.__browser.find_elements(by=By.XPATH,
                                                                                            value=movie_button_xpath)
                # Sanity check: ensure the element we want to click still exists.
                if i >= len(current_candidate_elements):
                    self.__logger.warning(f"Candidate {i + 1} is no longer present after page reload. Aborting.")
                    break  # Exit the loop if the list of candidates has shrunk.

                # Get the specific candidate for this iteration.
                candidate_to_click: WebElement = current_candidate_elements[i]

                # Click the candidate.
                self.__logger.info(f"Clicking candidate {i + 1}...")

                self.__browser.click(
                    button_locator=candidate_to_click,
                    pre_method=WaitingCondition(
                        condition=element_to_be_clickable(candidate_to_click),
                        timeout=10,
                        error_message=f"Candidate button {i + 1} for '{movie_name}' was not clickable."
                    ),
                    post_method=WaitingCondition(
                        condition=PageChangeCondition(searching_url=searching_url),
                        error_message="Page did not change after clicking candidate.",
                        timeout=self.__page_loading_timeout
                    )
                )

                # Validate the new page.
                self.__logger.info("Validating page for movie name and box office data...")
                self.__browser.wait(method_setting=WaitingCondition(
                    condition=text_to_be_present_in_element(movie_name_locator, movie_name),
                    timeout=5,
                    error_message=f"Page after click does not display the expected movie name '{movie_name}'."
                ))
                self.__browser.wait(method_setting=WaitingCondition(
                    condition=visibility_of_element_located(
                        locator=(By.CSS_SELECTOR, '#weekends-tab-panel > table > tbody > tr')),
                    timeout=5,
                    error_message="Validation failed: Page does not contain any box office data rows (tr)."
                ))

                # If all validations succeed, we're done.
                current_url: str = self.__browser.current_url
                self.__logger.info(f"Successfully navigated to a valid movie page for '{movie_name}' at: {current_url}")
                return current_url

            except (NoSuchElementException, TimeoutException, InvalidSwitchToTargetException,
                    UnexpectedAlertPresentException) as e:
                self.__logger.warning(f"Processing candidate {i + 1} for '{movie_name}' failed: {e}")
                # The loop will naturally proceed to the next iteration.
                # The check for the last attempt is handled by the loop's boundary.

            except Exception as e:
                self.__logger.error(
                    f"An unexpected error occurred while processing candidate {i + 1} for '{movie_name}': {e}",
                    exc_info=True)
                return None

        # If the loop finishes, no valid page was found
        self.__logger.error(
            f"All {num_candidates} candidates for '{movie_name}' failed validation. No valid page found.")
        return None

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
                self.__browser.click(button_locator=ElementLocator(by=By.CSS_SELECTOR, value=week_button_selector))
            except (NoSuchElementException, TimeoutException) as e:
                self.__logger.warning(f"Clicking '本週' button failed: {e}")
                raise

        self.__logger.info(f"Trying to search download button with {self.__scrap_file_extension} format.")
        download_button_selector: str = f"div#export-button-container button[data-ext='{self.__scrap_file_extension}']"
        try:
            self.__browser.click(
                button_locator=ElementLocator(by=By.CSS_SELECTOR, value=download_button_selector),
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

    def __search_and_fetch_box_office(
        self, movie_name: str, movie_id: Optional[int] = None, known_url: Optional[str] = None, trying_times: int = 3
    ) -> Tuple[Optional[list[BoxOffice]], Optional[str]]:
        """
        Handles the complete workflow for fetching a single movie's box office data.

        This method orchestrates the process of navigating to the movie's page
        (optimizing with a known URL if available), clicking the download button,
        and parsing the resulting file. It uses a temporary directory for downloads
        and includes a retry mechanism to handle transient network or browser issues.

        :param movie_name: The name of the movie to fetch.
        :param movie_id: The unique ID of the movie, used for logging.
        :param known_url: An optional, pre-existing URL for the movie's page to attempt direct navigation.
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

    def fetch_single_movie_data(
        self, movie_name: str, movie_id: Optional[int] = None, known_url: Optional[str] = None
    ) -> Tuple[Optional[list[BoxOffice]], Optional[str]]:
        """
        Fetches box office data for a single movie and returns it in memory.

        This is the primary public method for retrieving data for one movie.
        It must be called within the ``with`` context of the collector.

        :param movie_name: The name of the movie to fetch data for.
        :param movie_id: The unique ID of the movie, used for logging purposes.
        :param known_url: An optional, pre-existing URL for the movie's page to
                          attempt direct navigation, potentially speeding up the process.
        :returns: A tuple containing:
                  - A list of ``BoxOffice`` objects representing the movie's data, or ``None`` if fetching fails.
                  - The final URL of the movie's page, or ``None`` if navigation fails.
        """
        self._check_browser_active()

        # The core logic is already in __search_and_fetch_box_office. We just call it.
        box_office_data, movie_url = self.__search_and_fetch_box_office(
            movie_name=movie_name, movie_id=movie_id, known_url=known_url
        )

        if box_office_data is None:
            self.__logger.error(
                f"Failed to download box office data for movie '{movie_name}' (ID: {movie_id}) after multiple attempts."
            )

        return box_office_data, movie_url

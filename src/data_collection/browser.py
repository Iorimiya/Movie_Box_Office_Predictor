import time
from contextlib import contextmanager
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Callable, Final, Iterator, Optional, TypeAlias

from selenium import webdriver
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    NoSuchElementException,
    TimeoutException,
    UnexpectedAlertPresentException
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.ui import WebDriverWait
from seleniumbase import Driver
from seleniumbase import undetected as sel_undef
from typing_extensions import override
from urllib3.exceptions import MaxRetryError

from src.core.logging_manager import LoggingManager

ChromeExperimentalOptions: TypeAlias = dict[str, str]


class Browser(webdriver.Chrome):
    """
    A wrapper class for Selenium Chrome WebDriver with enhanced functionalities.

    This class extends the standard Selenium Chrome WebDriver to include
    custom waiting conditions, simplified navigation, and element interaction methods.
    It also configures common Chrome options for headless browsing and custom download paths.

    :ivar __download_path: The directory path where downloaded files are saved.
    :ivar __page_loading_timeout: The maximum time in seconds to wait for a page to load.
    :ivar __home_url: The URL for the browser's home page.
    :ivar __options: The Selenium Chrome options for the browser instance.
    :ivar __logger: The logger instance for logging messages.
    """

    class DownloadFinishCondition(object):
        """
        A condition class to check if a download is finished by verifying the existence of the target file.

        :ivar __download_path: The path to the downloaded file to check for existence.
        """

        def __init__(self, download_file_path: Path) -> None:
            """
            Initializes the DownloadFinishCondition.

            :param download_file_path: The path to the downloaded file that will be checked for existence.
            """
            self.__download_path: Path = download_file_path

        def __call__(self, driver: webdriver.Chrome) -> bool:
            """
            Checks if the download file exists.

            This method is called by WebDriverWait to determine if the condition is met.

            :param driver: The Chrome WebDriver instance (unused in this specific condition, but required by the WebDriverWait protocol).
            :returns: ``True`` if the file at ``self.__download_path`` exists, ``False`` otherwise.
            """
            return self.__download_path.exists()

    class PageChangeCondition(object):
        """
        A condition class to check if the browser's current URL has changed from an initial URL.

        :ivar __old_url: The initial URL to compare against the current URL.
        """

        def __init__(self, searching_url: str) -> None:
            """
            Initializes the PageChangeCondition.

            :param searching_url: The initial URL to compare against the current URL.
            """
            self.__old_url: str = searching_url

        def __call__(self, driver: webdriver.Chrome) -> bool:
            """
            Checks if the current URL is different from the initial URL.

            This method is called by WebDriverWait to determine if the condition is met.

            :param driver: The Chrome WebDriver instance from which to get the current URL.
            :returns: ``True`` if the current URL of the ``driver`` is different from ``self.__old_url``, ``False`` otherwise.
            """
            return driver.current_url != self.__old_url

    @dataclass(kw_only=True)
    class WaitingCondition:
        """
        A dataclass representing parameters for a waiting condition used with WebDriverWait.

        :ivar condition: A callable (e.g., an expected_condition from Selenium or a custom callable)
                         that WebDriverWait will poll until it returns ``True`` or a non-``None`` WebElement.
        :ivar timeout: The maximum time in seconds to wait for the condition to be met.
                       If ``None``, a default timeout will be used by the ``wait`` method.
        :ivar error_message: A message to log if the waiting condition times out.
        """
        condition: Callable[[any], bool | WebElement]
        timeout: Optional[float]
        error_message: str

    @override
    def __init__(self, download_path: Optional[Path] = None, page_loading_timeout: float = 120,
                 target_url: Optional[str] = None) -> None:
        """
        Initializes the Browser.

        Sets up Chrome options for headless browsing, no-sandbox, disabled GPU,
        and a default window size. If a ``download_path`` is provided,
        it configures Chrome to use this path for downloads.
        Optionally navigates to a ``target_url`` upon initialization.

        :param download_path: The directory path where downloaded files should be saved.
                              If ``None``, the default Chrome download path is used.
        :param page_loading_timeout: The maximum time in seconds to wait for a page to load
                                     during navigation.
        :param target_url: An optional URL to navigate to immediately after the browser is initialized.
        """
        # driver options
        self.__download_path: Path = download_path
        self.__page_loading_timeout: Final[float] = page_loading_timeout
        self.__home_url: Final[str] = "chrome://newtab"

        # options
        self.__options: Options = Options()
        self.__options.add_argument(argument="--headless")
        self.__options.add_argument(argument="--no-sandbox")
        self.__options.add_argument(argument="--disable-dev-shm-usage")
        self.__options.add_argument(argument="--disable-gpu")
        self.__options.add_argument(argument="--window-size=1600,900")
        self.__logger: Logger = LoggingManager().get_logger('root')

        if self.__download_path:
            # options to change defaults download dir
            experimental_option: ChromeExperimentalOptions = {"download.default_directory": str(self.__download_path)}
            self.__options.add_experimental_option(name="prefs", value=experimental_option)
            self.__logger.info(f"Download path switch to \"{download_path}\".")

        # create web driver
        super().__init__(options=self.__options)

        # go to base_url
        if target_url:
            self.get(url=target_url)

    @override
    def __enter__(self) -> any:
        """
        Enters the runtime context related to this object.

        Calls the ``__enter__`` method of the parent ``webdriver.Chrome`` class.

        :returns: The browser instance itself.
        """
        return super().__enter__()

    @override
    def __exit__(self, exc_type, exc_val, exc_tb) -> any:
        """
        Exits the runtime context related to this object, ensuring the browser is properly closed.

        Calls the ``__exit__`` method of the parent ``webdriver.Chrome`` class.

        :param exc_type: The type of the exception that caused the context to be exited, if any.
        :param exc_val: The exception instance that caused the context to be exited, if any.
        :param exc_tb: A traceback object encapsulating the call stack at the point
                       where the exception was raised, if any.
        """
        super().__exit__(exc_type, exc_val, exc_tb)

    def wait(self, method_setting: WaitingCondition, defaults_timeout: float = 120) -> None:
        """
        Waits for a specific condition to be met using WebDriverWait.

        If the condition is not met within the specified timeout (or ``defaults_timeout``
        if ``method_setting.timeout`` is ``None``), a ``TimeoutException`` is caught,
        and the ``error_message`` from ``method_setting`` is logged.

        :param method_setting: An instance of ``WaitingCondition`` defining the condition,
                               timeout, and error message.
        :param defaults_timeout: The default timeout in seconds to use if ``method_setting.timeout``
                                 is not specified.
        """
        try:
            WebDriverWait(self, timeout=method_setting.timeout if method_setting.timeout else defaults_timeout).until(
                method_setting.condition, message='')
        except TimeoutException:
            if method_setting.error_message:
                self.__logger.warning(method_setting.error_message)
        return

    @override
    def get(self, url: str) -> None:
        """
        Navigates to a given URL and waits for the page to change.

        It logs the navigation attempt and success. If the page does not change
        within the ``self.__page_loading_timeout``, an error message is logged.

        :param url: The URL to navigate to.
        """
        old_url = self.current_url
        self.__logger.debug(f"Trying to navigate to \"{url}\".")
        super().get(url)
        if self.wait(self.WaitingCondition(condition=self.PageChangeCondition(searching_url=old_url),
                                           timeout=self.__page_loading_timeout,
                                           error_message=f"Read Timeout Error on {url} caught.")):
            self.__logger.debug(f"Navigate to url \"{self.current_url}\" success.")
        return

    def home(self) -> None:
        """
        Navigates to the browser's configured home URL (``chrome://newtab``).
        """
        self.get(self.__home_url)
        return

    def find_button(self, button_selector_path: str) -> WebElement:
        """
        Finds a button element on the page using a CSS selector.

        Logs the attempt and result of finding the button.

        :param button_selector_path: The CSS selector path of the button element.
        :returns: The found ``WebElement`` representing the button.
        :raises NoSuchElementException: If no element is found matching the ``button_selector_path``.
        """
        try:
            self.__logger.info(f"Trying to find button located on \"{button_selector_path}\".")
            button_element: WebElement = self.find_element(by=By.CSS_SELECTOR, value=button_selector_path)
        except NoSuchElementException:
            self.__logger.warning(f"Cannot find button located on \"{button_selector_path}\".", exc_info=True)
            raise
        else:
            self.__logger.info(f"Found button located on \"{button_selector_path}\".")
            return button_element

    def click(self, button_locator: WebElement | str,
              pre_method: WaitingCondition | None = None,
              post_method: WaitingCondition | None = None) -> None:
        """
        Clicks a button element, with optional waiting conditions before and after the click.

        The button can be specified either as a ``WebElement`` object or by its CSS selector string.

        :param button_locator: The ``WebElement`` to click, or a string representing the CSS selector
                               for the button.
        :param pre_method: An optional ``WaitingCondition`` to satisfy before attempting the click.
        :param post_method: An optional ``WaitingCondition`` to satisfy after the click is performed.
        :raises NoSuchElementException: If ``button_locator`` is a string and the button cannot be found,
                                        or if the located button cannot be clicked (e.g., it's intercepted or gone).
        :raises ValueError: If ``button_locator`` is not a ``WebElement`` or a string.
        """

        if isinstance(button_locator, str):
            self.__logger.debug("Found string parameter, use CSS selector to find button.")
            try:
                button = self.find_button(button_locator)
            except NoSuchElementException:
                self.__logger.debug(f"Cannot find button located on \"{button_locator}\".", exc_info=True)
                raise
        elif isinstance(button_locator, WebElement):
            self.__logger.debug("Found Element parameter, set button variable to it.")
            button = button_locator
        else:
            self.__logger.error("Unknown parameter type.")
            raise ValueError

        if button:
            if pre_method:
                self.wait(pre_method)
            try:
                button.click()
                self.__logger.info(f"Button is clicked.")
            except UnexpectedAlertPresentException:
                # This is the specific error we are now handling.
                self.__logger.warning("An unexpected alert appeared after clicking the button.")
                self._handle_alert()
                # After handling the alert, we raise a known exception to signal the click "failed"
                # in its primary goal (e.g., starting a download).
                raise TimeoutException("Click was intercepted by a website alert.")
            except (ElementClickInterceptedException, AttributeError):
                self.__logger.warning(f"The button cannot be clicked.", exc_info=True)
                raise NoSuchElementException
            if post_method:
                self.wait(post_method)
            return
        else:
            raise NoSuchElementException

    def set_download_path(self, new_path: Path) -> None:
        """
        Dynamically sets the download directory for the current browser session.

        This method uses the Chrome DevTools Protocol (CDP) command 'Page.setDownloadBehavior'
        to change the download path of the running browser instance.

        :param new_path: The new absolute path for the download directory.
        """
        if not new_path.is_absolute():
            new_path: Path = new_path.resolve(strict=True)

        self.execute_cdp_cmd(
            cmd='Page.setDownloadBehavior',
            cmd_args={'behavior': 'allow', 'downloadPath': str(new_path)}
        )
        self.__download_path = new_path
        self.__logger.info(f"Dynamically set browser download path to '{new_path}'.")
        return

    @contextmanager
    def temporary_download_path(self, new_path: Path) -> Iterator[None]:
        """
        Provides a context manager to temporarily change the browser's download path.

        Upon entering the `with` block, the download path is changed to `new_path`.
        Upon exiting the block (either normally or due to an exception), the
        original download path is automatically restored.

        :param new_path: The temporary download path to use within the context.
        """
        original_path: Optional[Path] = self.__download_path
        self.__logger.debug(f"Temporarily setting download path to '{new_path}'. Original was '{original_path}'.")
        self.set_download_path(new_path=new_path)
        try:
            yield
        finally:
            if original_path:
                self.__logger.debug(f"Restoring original download path to '{original_path}'.")
                self.set_download_path(new_path=original_path)
            else:
                # If there was no original path, we can't restore it.
                # The behavior here could be to set it to a default, but for now, we just log it.
                self.__logger.debug("No original download path to restore.")

    def _handle_alert(self) -> None:
        """
        Checks for and dismisses any open JavaScript alert.

        This method attempts to switch to an alert, logs its text,
        and then accepts it to allow the browser to continue.
        """
        try:
            alert: Alert = self.switch_to.alert
            alert_text: str = alert.text
            self.__logger.warning(f"Caught an alert from the website: '{alert_text}'")
            alert.accept()
            self.__logger.info("Alert dismissed.")
        except NoSuchElementException:
            # This is the expected case when no alert is present.
            pass
        except Exception as e:
            # Catch any other potential errors during alert handling.
            self.__logger.error(f"An unexpected error occurred while handling an alert: {e}", exc_info=True)


class CaptchaBrowser:
    """
    A browser class designed to interact with web pages that may present captchas,
    using SeleniumBase's undetected ChromeDriver.

    This class provides a context manager for browser session management and methods
    for navigation, element finding, and script execution, with specific handling
    for captchas using SeleniumBase's UC mode features.
    """

    def __init__(self, no_sandbox: bool = True, incognito: bool = True, size: tuple[int, int] = (1600, 900)) -> None:
        """
        Initializes the CaptchaBrowser configuration.

        Note: The actual SeleniumBase Driver is instantiated in the ``__enter__`` method.

        :param no_sandbox: If ``True``, runs Chrome with the '--no-sandbox' argument.
                           Useful for running in Docker or CI environments.
        :param incognito: If ``True``, runs Chrome in incognito mode.
        :param size: A tuple ``(width, height)`` specifying the desired window size.
        """
        self.__driver: Optional[sel_undef.Chrome] = None
        self.__uc: Final[bool] = True
        self.__headless: Final[bool] = False
        self.__no_sandbox: Final[bool] = no_sandbox
        self.__incognito: Final[bool] = incognito
        self.__size: Final[tuple[int, int]] = size
        self.__home_url: Final[str] = "chrome://newtab"

    def __enter__(self) -> any:
        """
        Initializes and returns the SeleniumBase undetected ChromeDriver instance.

        Configures the driver with the options specified during ``__init__``.
        The window size is set after the driver is created.

        :returns: The ``CaptchaBrowser`` instance itself.
        """
        self.__driver = Driver(uc=self.__uc, headless=self.__headless, no_sandbox=self.__no_sandbox,
                               incognito=self.__incognito)
        self.__driver.set_window_size(self.__size[0], self.__size[1])
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> any:
        """
        Quits the SeleniumBase driver, closing all browser windows and ending the session.

        :param exc_type: The type of the exception that caused the context to be exited, if any.
        :param exc_val: The exception instance that caused the context to be exited, if any.
        :param exc_tb: A traceback object encapsulating the call stack at the point
                       where the exception was raised, if any.
        """
        self.__driver.quit()
        return

    @staticmethod
    def wait(sec: float = 5) -> None:
        """
        Pauses execution for a specified number of seconds.

        A simple wrapper around ``time.sleep()``.

        :param sec: The duration to wait, in seconds.
        """
        time.sleep(sec)

    def get(self, url: str, captcha: bool) -> None:
        """
        Navigates to a URL, with an option to handle potential captchas using SeleniumBase's UC mode.

        If ``captcha`` is ``True``, it activates CDP mode, waits, attempts to click the captcha GUI element,
        and then connects.
        If ``captcha`` is ``False``, it performs a standard ``get`` operation, with a retry mechanism
        in case of ``MaxRetryError``.

        :param url: The URL to navigate to.
        :param captcha: If ``True``, attempts to handle captcha interaction.
        """
        if captcha:
            self.__driver.uc_activate_cdp_mode(url)
            self.wait(5)
            self.__driver.uc_gui_click_captcha()
            self.__driver.connect()
        else:
            try:
                self.__driver.get(url)
            except MaxRetryError:
                self.__driver.reconnect()
        return

    def find_element(self, selector: str) -> WebElement:
        """
        Finds a single web element using a CSS selector.

        This is a direct pass-through to the underlying SeleniumBase driver's ``find_element`` method.

        :param selector: The CSS selector string to locate the element.
        :returns: The first ``WebElement`` found matching the selector.
        :raises NoSuchElementException: If no element is found.
        """
        return self.__driver.find_element(selector)

    def find_elements(self, selector: str) -> list[WebElement]:
        """
        Finds all web elements matching a CSS selector.

        This is a direct pass-through to the underlying SeleniumBase driver's ``find_elements`` method.

        :param selector: The CSS selector string to locate elements.
        :returns: A list of ``WebElement`` objects found. Returns an empty list if no elements are found.
        """
        return self.__driver.find_elements(selector)

    def execute_script(self, script: str) -> None:
        """
        Executes JavaScript in the context of the currently selected frame or window.

        This is a direct pass-through to the underlying SeleniumBase driver's ``execute_script`` method.

        :param script: The JavaScript code to execute.
        """
        self.__driver.execute_script(script=script)

    def home(self) -> None:
        """
        Navigates to the browser's configured home URL (``chrome://newtab``).

        Captcha handling is set to ``False`` for this internal navigation.
        """
        self.get(self.__home_url, captcha=False)
        return

import time
from logging import Logger
from pathlib import Path
from typing import TypeAlias, Callable, Final

from typing_extensions import override
from dataclasses import dataclass

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import *
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.remote.webelement import WebElement

from seleniumbase import Driver
from seleniumbase import undetected as sel_undef
from urllib3.exceptions import MaxRetryError

from tools.logging_manager import LoggingManager

ChromeExperimentalOptions: TypeAlias = dict[str:str]


class Browser(webdriver.Chrome):
    """
    A wrapper class for Selenium Chrome WebDriver with enhanced functionalities.
    """
    class DownloadFinishCondition(object):
        """
        A condition class to check if a download is finished.
        """
        def __init__(self, download_file_path: Path) -> None:
            """
            Initializes the DownloadFinishCondition.

            Args:
                download_file_path (Path): The path to the downloaded file.
            """
            self.__download_path: Path = download_file_path

        def __call__(self, driver: webdriver.Chrome) -> bool:
            """
            Checks if the download file exists.

            Args:
                driver (webdriver.Chrome): The Chrome WebDriver instance.

            Returns:
                bool: True if the file exists, False otherwise.
            """
            return self.__download_path.exists()

    class PageChangeCondition(object):
        """
        A condition class to check if the page has changed.
        """
        def __init__(self, searching_url: str) -> None:
            """
            Initializes the PageChangeCondition.

            Args:
                searching_url (str): The initial URL.
            """
            self.__old_url: str = searching_url

        def __call__(self, driver: webdriver.Chrome) -> bool:
            """
            Checks if the current URL is different from the initial URL.

            Args:
                driver (webdriver.Chrome): The Chrome WebDriver instance.

            Returns:
                bool: True if the URL has changed, False otherwise.
            """
            return driver.current_url != self.__old_url

    @dataclass(kw_only=True)
    class WaitingCondition:
        """
        A dataclass representing waiting conditions.
        """
        condition: Callable[[any], bool | WebElement]
        timeout: Optional[float]
        error_message: str

    @override
    def __init__(self, download_path: Optional[Path] = None, page_loading_timeout: float = 120,
                 target_url: Optional[str] = None) -> None:
        """
        Initializes the Browser.

        Args:
            download_path (Optional[Path]): The download path. Defaults to None.
            page_loading_timeout (float): The page loading timeout in seconds. Defaults to 120.
            target_url (Optional[str]): The target URL to navigate to. Defaults to None.
        """
        # driver options
        self.__download_path: Final[Path] = download_path
        self.__page_loading_timeout: Final[float] = page_loading_timeout
        self.__home_url: Final[str] = "chrome://newtab"

        # options
        self.__options: Options = Options()
        self.__options.add_argument(argument="--headless")
        self.__options.add_argument(argument="--no-sandbox")
        self.__options.add_argument(argument="--disable-dev-shm-usage")
        self.__options.add_argument(argument="--disable-gpu")
        self.__options.add_argument(argument="--window-size=1600,900")
        self.__logger:Logger = LoggingManager().get_logger('root')

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
        return super().__enter__()

    @override
    def __exit__(self, exc_type, exc_val, exc_tb) -> any:
        super().__exit__(exc_type, exc_val, exc_tb)

    def wait(self, method_setting: WaitingCondition, defaults_timeout: float = 120) -> None:
        """
        Waits for a condition to be met.

        Args:
            method_setting (WaitingCondition): The waiting condition.
            defaults_timeout (float): The default timeout in seconds. Defaults to 120.
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
        Navigates to a URL.

        Args:
            url (str): The URL to navigate to.
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
        Navigates to the home URL.
        """
        self.get(self.__home_url)
        return

    def find_button(self, button_selector_path: str) -> WebElement:
        """
        Finds a button element.

        Args:
            button_selector_path (str): The CSS selector path of the button.

        Returns:
            WebElement: The button element.

        Raises:
            NoSuchElementException: If the button is not found.
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
        Clicks a button element.

        Args:
            button_locator (WebElement | str): The button element or its CSS selector path.
            pre_method (WaitingCondition | None): Waiting condition before clicking. Defaults to None.
            post_method (WaitingCondition | None): Waiting condition after clicking. Defaults to None.

        Raises:
            NoSuchElementException: If the button is not found.
            ValueError: If the parameter type is unknown.
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
            except (ElementClickInterceptedException, AttributeError):
                self.__logger.warning(f"The button cannot be clicked.", exc_info=True)
                raise NoSuchElementException
            if post_method:
                self.wait(post_method)
            return
        else:
            raise NoSuchElementException


class CaptchaBrowser:
    """
    A browser class that handles captcha.
    """
    def __init__(self, no_sandbox: bool = True, incognito: bool = True, size: tuple[int, int] = (1600, 900)) -> None:
        """
        Initializes the CaptchaBrowser.

        Args:
            no_sandbox (bool): Whether to run in no-sandbox mode. Defaults to True.
            incognito (bool): Whether to run in incognito mode. Defaults to True.
            size (tuple[int, int]): The window size. Defaults to (1600, 900).
        """
        self.__driver: Optional[sel_undef.Chrome] = None
        self.__uc: Final[bool] = True
        self.__headless: Final[bool] = False
        self.__no_sandbox: Final[bool] = no_sandbox
        self.__incognito: Final[bool] = incognito
        self.__size: Final[tuple[int, int]] = size
        self.__home_url: Final[str] = "chrome://newtab"

    def __enter__(self) -> any:
        self.__driver = Driver(uc=self.__uc, headless=self.__headless, no_sandbox=self.__no_sandbox,
                               incognito=self.__incognito)
        self.__driver.set_window_size(self.__size[0], self.__size[1])
        # self.__driver.minimize_window()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> any:
        self.__driver.quit()
        return

    @staticmethod
    def wait(sec: float = 5) -> None:
        """
        Waits for a specified time.

        Args:
            sec (float): The waiting time in seconds. Defaults to 5.
        """
        time.sleep(sec)

    def get(self, url: str, captcha: bool) -> None:
        """
        Navigates to a URL, handling captcha if necessary.

        Args:
            url (str): The URL to navigate to.
            captcha (bool): Whether to handle captcha.
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
        Finds an element.

        Args:
            selector (str): The CSS selector.

        Returns:
            WebElement: The found element.
        """
        return self.__driver.find_element(selector)

    def find_elements(self, selector: str) -> list[WebElement]:
        """
        Finds multiple elements.

        Args:
            selector (str): The CSS selector.

        Returns:
            list[WebElement]: The found elements.
        """
        return self.__driver.find_elements(selector)

    def execute_script(self, script: str) -> None:
        """
        Executes a JavaScript script.

        Args:
            script (str): The JavaScript script.
        """
        self.__driver.execute_script(script=script)

    def home(self) -> None:
        """
        Navigates to the home URL.
        """
        self.get(self.__home_url, captcha=False)
        return

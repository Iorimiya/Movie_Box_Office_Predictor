import time
import logging
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

ChromeExperimentalOptions: TypeAlias = dict[str:str]


class Browser(webdriver.Chrome):
    class DownloadFinishCondition(object):
        def __init__(self, download_file_path: Path) -> None:
            self.__download_path: Path = download_file_path

        def __call__(self, driver: webdriver.Chrome) -> bool:
            return self.__download_path.exists()

    class PageChangeCondition(object):
        def __init__(self, searching_url: str) -> None:
            self.__old_url: str = searching_url

        def __call__(self, driver: webdriver.Chrome) -> bool:
            return driver.current_url != self.__old_url

    @dataclass(kw_only=True)
    class WaitingCondition:
        condition: Callable[[any], bool | WebElement]
        timeout: Optional[float]
        error_message: str

    @override
    def __init__(self, download_path: Optional[Path] = None, page_loading_timeout: float = 120,
                 target_url: Optional[str] = None) -> None:
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

        if self.__download_path:
            # options to change defaults download dir
            experimental_option: ChromeExperimentalOptions = {"download.default_directory": str(self.__download_path)}
            self.__options.add_experimental_option(name="prefs", value=experimental_option)
            logging.info(f"download path switch to \"{download_path}\".")

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
        try:
            WebDriverWait(self, timeout=method_setting.timeout if method_setting.timeout else defaults_timeout).until(
                method_setting.condition, message='')
        except TimeoutException:
            if method_setting.error_message:
                logging.warning(method_setting.error_message)
        return

    @override
    def get(self, url: str) -> None:
        old_url = self.current_url
        logging.debug(f"trying to navigate to \"{url}\".")
        super().get(url)
        if self.wait(self.WaitingCondition(condition=self.PageChangeCondition(searching_url=old_url),
                                           timeout=self.__page_loading_timeout,
                                           error_message=f"Read Timeout Error on {url} caught.")):
            logging.debug(f"navigate to url \"{self.current_url}\" success..")
        return

    def home(self) -> None:
        self.get(self.__home_url)
        return

    def find_button(self, button_selector_path: str) -> WebElement:
        try:
            logging.info(f"trying to find button located on \"{button_selector_path}\".")
            button_element: WebElement = self.find_element(by=By.CSS_SELECTOR, value=button_selector_path)
        except NoSuchElementException:
            logging.warning(f"cannot find button located on \"{button_selector_path}\".", exc_info=True)
            raise
        else:
            logging.info(f"found button located on \"{button_selector_path}\".")
            return button_element

    def click(self, button_locator: WebElement | str,
              pre_method: WaitingCondition | None = None,
              post_method: WaitingCondition | None = None) -> None:

        if isinstance(button_locator, str):
            logging.debug("found string parameter, use CSS selector to find button.")
            try:
                button = self.find_button(button_locator)
            except NoSuchElementException:
                logging.debug(f"cannot find button located on \"{button_locator}\".", exc_info=True)
                raise
        elif isinstance(button_locator, WebElement):
            logging.debug("found Element parameter, set button variable to it.")
            button = button_locator
        else:
            logging.error("unknown parameter type.")
            raise ValueError

        if button:
            if pre_method:
                self.wait(pre_method)
            try:
                button.click()
                logging.info(f"button is clicked.")
            except (ElementClickInterceptedException, AttributeError):
                logging.warning(f"the button cannot be clicked.", exc_info=True)
                raise NoSuchElementException
            if post_method:
                self.wait(post_method)
            return
        else:
            raise NoSuchElementException


class CaptchaBrowser:
    def __init__(self, uc: bool = True, headless: bool = False, no_sandbox: bool = True, incognito: bool = True,
                 size: tuple[int, int] = (1600, 900)) -> None:
        self.__driver: Optional[sel_undef.Chrome] = None
        self.__uc: Final[bool] = uc
        self.__headless: Final[bool] = headless
        self.__no_sandbox: Final[bool] = no_sandbox
        self.__incognito: Final[bool] = incognito
        self.__size: Final[tuple[int, int]] = size
        self.__home_url: Final[str] = "chrome://newtab"

    def __enter__(self) -> any:
        self.__driver = Driver(uc=self.__uc, headless=self.__headless, no_sandbox=self.__no_sandbox,
                               incognito=self.__incognito)
        self.__driver.set_window_size(self.__size[0], self.__size[1])
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> any:
        self.__driver.quit()
        return

    @staticmethod
    def wait(sec: float = 5) -> None:
        time.sleep(sec)

    def get(self, url: str) -> None:
        self.__driver.uc_activate_cdp_mode(url)
        self.wait(5)
        self.__driver.uc_gui_click_captcha()

    def find_element(self, selector: str) -> WebElement:
        self.__driver.wait_for_element(selector)
        return self.__driver.find_element(selector)

    def find_elements(self,  selector: str) -> list[WebElement]:
        self.__driver.wait_for_element(selector)
        return self.__driver.find_elements(selector)
    def home(self)-> None:
        self.__driver.get(self.__home_url)
        return


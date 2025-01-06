import logging
from enum import Enum
from pathlib import Path
from typing_extensions import override
from typing import TypeAlias, Callable, TypedDict

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common import exceptions as se
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.remote.webelement import WebElement

ChromeExperimentalOptions: TypeAlias = dict[str:str]


class TimeoutType(Enum):
    PAGE_LOADING = 1
    DOWNLOAD = 2


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

    class WaitingMethodSetting(TypedDict):
        method: Callable[[any], bool | WebElement]
        timeout: float
        error_message: str
        timeout_type: TimeoutType

    @override
    def __init__(self, download_path: Path | None = None, page_loading_timeout: float = 0,
                 download_timeout: float = 0, target_url: str | None = None) -> None:
        # driver options
        self.__download_path = download_path
        self.__defaults_timeout = 120
        self.__page_loading_timeout = page_loading_timeout if page_loading_timeout else self.__defaults_timeout
        self.__download_timeout = download_timeout if download_timeout else self.__defaults_timeout

        # options
        self.__options: Options = Options()
        # self.__options.add_argument(argument="--headless")
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

    @override
    def get(self, url: str) -> None:
        old_url = self.current_url
        logging.debug(f"trying to navigate to \"{url}\".")
        super().get(url)
        if self.wait(self.WaitingMethodSetting(method=self.PageChangeCondition(searching_url=old_url),
                                               timeout=self.__page_loading_timeout,
                                               error_message=f"Read Timeout Error on {url} caught.",
                                               timeout_type=TimeoutType.PAGE_LOADING)):
            logging.debug(f"navigate to url \"{self.current_url}\" success..")
        return

    def find_button(self, button_selector_path: str) -> WebElement | bool:
        try:
            logging.info(f"trying to find button located on \"{button_selector_path}\".")
            button_element: WebElement = self.find_element(by=By.CSS_SELECTOR, value=button_selector_path)
        except se.NoSuchElementException:
            logging.warning(f"cannot find button located on \"{button_selector_path}\".")
            logging.debug(msg='', exc_info=True)
            return False
        else:
            logging.info(f"found button located on \"{button_selector_path}\".")
            return button_element

    def wait(self, method_setting: WaitingMethodSetting) -> bool:
        try:
            WebDriverWait(self, timeout=method_setting['timeout']).until(method_setting['method'], message='')
        except se.TimeoutException:
            if method_setting['error_message']:
                logging.warning(method_setting['error_message'])
            return False
        return True

    def click(self, button_locator: WebElement | str,
              pre_method: WaitingMethodSetting | None = None,
              post_method: WaitingMethodSetting | None = None) -> bool:

        if isinstance(button_locator, str):
            logging.debug("found string parameter, use CSS selector to find button.")
            button = self.find_button(button_locator)
        elif isinstance(button_locator, WebElement):
            logging.debug("found Element parameter, set button variable to it.")
            button = button_locator
        else:
            logging.error("unknown parameter type.")
            raise ValueError

        if pre_method:
            if pre_method['timeout_type'] == TimeoutType.PAGE_LOADING:
                pre_method['timeout'] = pre_method['timeout'] if pre_method['timeout'] else self.__page_loading_timeout
            elif pre_method['timeout_type'] == TimeoutType.DOWNLOAD:
                pre_method['timeout'] = pre_method['timeout'] if pre_method['timeout'] else self.__download_timeout
            else:
                raise ValueError

        if post_method:
            if post_method['timeout_type'] == TimeoutType.PAGE_LOADING:
                post_method['timeout'] = post_method['timeout'] if post_method[
                    'timeout'] else self.__page_loading_timeout
            elif post_method['timeout_type'] == TimeoutType.DOWNLOAD:
                post_method['timeout'] = post_method['timeout'] if post_method['timeout'] else self.__download_timeout
            else:
                raise ValueError

        if button:
            if pre_method is not None:
                if not self.wait(pre_method):
                    return False
            try:
                button.click()
                logging.info(f"button is clicked.")
            except (se.ElementClickInterceptedException, AttributeError):
                logging.warning(f"the button cannot be clicked.")
                logging.debug(msg='', exc_info=True)
                return False
            if post_method is not None:
                if not self.wait(post_method):
                    return False
            return True
        return False

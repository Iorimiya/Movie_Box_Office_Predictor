import re
import time
import logging
from typing import TypeAlias
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.remote.webelement import WebElement
from selenium.common import exceptions as selenium_exceptions
from urllib3.exceptions import ReadTimeoutError

ChromeExperimentalOptions: TypeAlias = dict[str:str]


class ColabBrowser(webdriver.Chrome):

    def __init__(self, download_path: Path, target_url: str = None) -> None:
        # options
        self.__options: Options = Options()
        self.__options.add_argument(argument="--headless")
        self.__options.add_argument(argument="--no-sandbox")
        self.__options.add_argument(argument="--disable-dev-shm-usage")
        self.__options.add_argument(argument="--disable-gpu")
        self.__options.add_argument(argument="--window-size=1600,900")

        # options to change defaults download dir
        experimental_option: ChromeExperimentalOptions = {"download.default_directory": str(download_path)}
        self.__options.add_experimental_option(name="prefs", value=experimental_option)
        logging.debug(msg=f"download path switch to {download_path}")

        # create web driver and go to base_url
        super().__init__(options=self.__options)

        if target_url:
            self.get(url=target_url)

    def __enter__(self) -> any:
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb) -> any:
        super().__exit__(exc_type, exc_val, exc_tb)

    def click(self, button_name: str, button_selector_path: str) -> bool:
        try:
            week_box_office_button: WebElement = self.find_element(by=By.CSS_SELECTOR, value=button_selector_path)
        except selenium_exceptions.NoSuchElementException:
            logging.warning(msg=f"cannot find {button_name} button.")
            logging.debug(msg='', exc_info=True)
            return False
        try:
            week_box_office_button.click()
            logging.info(msg=f"{button_name} is clicked.")
        except (selenium_exceptions.ElementClickInterceptedException, AttributeError):
            logging.warning(msg=f"Download failed, the {button_name} cannot be clicked.")
            logging.debug(msg=f'', exc_info=True)
            return False
        return True

    def get(self, url: str, waiting_time: float = 0.05) -> None:
        try:
            logging.debug(msg=f"trying to go to {url}.")
            super().get(url)
        except ReadTimeoutError:
            logging.warning(msg=f"Read Timeout Error on {url} caught.")
            logging.debug(msg='', exc_info=True)
        except selenium_exceptions.UnexpectedAlertPresentException:
            logging.warning(msg="Unexpected Alert Caught,")
            logging.debug(msg='', exc_info=True)
        else:
            logging.debug(msg=f"goto url \"{self.current_url}\".")
        finally:
            time.sleep(waiting_time)

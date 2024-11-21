import time
import logging
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from urllib3.exceptions import ReadTimeoutError


class ColabBrowser(webdriver.Chrome):

    def __init__(self, target_url: str = None) -> None:
        # options
        self.__options = Options()
        self.__options.add_argument("--headless")
        self.__options.add_argument("--no-sandbox")
        self.__options.add_argument("--disable-dev-shm-usage")
        self.__options.add_argument("--disable-gpu")
        self.__options.add_argument("--window-size=1600,900")

        # options to change defaults download dir
        download_path = Path(__file__).resolve(strict=True).parent.joinpath("data")
        experimental_option = {"download.default_directory": str(download_path)}
        self.__options.add_experimental_option(name="prefs", value=experimental_option)
        logging.info(msg=f"download path switch to {download_path}")

        # create web driver and go to base_url
        super().__init__(options=self.__options)

        if target_url:
            self.get(url=target_url)

    def __enter__(self) -> any:
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb) -> any:
        super().__exit__(exc_type, exc_val, exc_tb)

    def get(self, url: str, waiting_time: int = 0.05) -> None:
        try:
            super().get(url)
        except ReadTimeoutError:
            logging.error(msg=f"open URL \"{url}\" failed.")
        else:
            logging.info(f"goto url \"{self.current_url}\"")
        finally:
            time.sleep(waiting_time)

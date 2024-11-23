from colab_browser import ColabBrowser

import csv
import time
import logging
from pathlib import Path
from selenium.webdriver.common.by import By
from selenium.common import exceptions as selenium_exceptions
from urllib3.exceptions import ReadTimeoutError


class MovieWeeklyBoxOfficeCollector:
    def __init__(self, page_changing_waiting_time: int = 1, download_waiting_time: int = 2) -> None:
        # browser setting
        self.__browser = ColabBrowser()
        self.__page_changing_waiting_time = page_changing_waiting_time
        self.__download_waiting_time = download_waiting_time

        # path dependent
        self.__data_path = Path("data")
        self.__downloaded_temp_file = self.__data_path.joinpath("各週週末票房資料匯出.csv")
        self.__weekly_box_office_data_folder = self.__data_path.joinpath("weekly_box_office_data", "by_movie_name")

        # url
        self.__searching_url = "https://boxofficetw.tfai.org.tw/search/0"
        self.__defaults_url = "https://google.com"

        # create path folder
        self.__weekly_box_office_data_folder.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> any:
        self.__browser = self.__browser.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> any:
        return self.__browser.__exit__(exc_type=exc_type, exc_val=exc_val, exc_tb=exc_tb)

    def go_to_movie_page(self, target_movie_name: str) -> None:
        searching_url = f"{self.__searching_url}/{target_movie_name}"
        try:
            self.__browser.get(url=searching_url, waiting_time=self.__page_changing_waiting_time)
        except ReadTimeoutError:
            return
        buttons = self.__browser.find_elements(
            by=By.XPATH,
            value='//div[@id="film-searcher"]/div[@class="body"]/button/span[@class="name"]'
        )
        target_element = next(
            (
                button.find_element(by=By.XPATH, value=f"./..")
                for button in buttons
                if button.text == target_movie_name
            ),
            None,
        )
        try:
            target_element.click()
            logging.info(msg=f"the drop-down list button of {target_movie_name} is clicked.")
        except selenium_exceptions.ElementClickInterceptedException:
            logging.error(msg=f"Searching failed, the drop-down list button of {target_movie_name} cannot be clicked",
                          exc_info=True)
            return
        except AttributeError:
            logging.critical(msg=f"Searching {target_movie_name} failed, none movie title drop-down list found.")
            return
        else:
            time.sleep(self.__page_changing_waiting_time)
            if self.__browser.current_url == self.__searching_url:
                raise AssertionError("No page changing detect.")
            logging.info(msg=f"goto url: {self.__browser.current_url}")
            return

    def get_weekly_box_office_data(self, movie_name: str, trying_times: int = 10) -> None:
        logging.info(f"Searching {movie_name} data")
        for index in range(trying_times):
            try:
                self.go_to_movie_page(target_movie_name=movie_name)
            except (ReadTimeoutError, AssertionError, AttributeError,
                    selenium_exceptions.ElementClickInterceptedException):
                logging.warning(msg=f"Searching {movie_name} failed")
                continue
            csv_button = self.__browser.find_element(
                by=By.XPATH,
                value='//div[@id="export-button-container"]/button[@data-type="CSV"]',
            )
            try:
                csv_button.click()
                logging.info(msg="csv button is clicked.")
            except (selenium_exceptions.ElementClickInterceptedException, AttributeError):
                logging.warning(msg="Download failed, the CSV button cannot be clicked", exc_info=True)
            else:
                time.sleep(self.__download_waiting_time)
                self.__downloaded_temp_file.replace(
                    self.__weekly_box_office_data_folder.joinpath(f"{movie_name}{self.__downloaded_temp_file.suffix}"))
                return
        raise AssertionError(f"Trying {trying_times} times but cannot download box office data.")

    @staticmethod
    def get_movie_list_from_file(csv_file_path: Path) -> list[str]:
        with open(file=csv_file_path, encoding='utf-8') as file:
            reader = csv.DictReader(file)
            return [row['片名'] for row in reader]

    def get_weekly_box_office_data_from_file(self, csv_file_path: Path):
        movie_list = self.get_movie_list_from_file(csv_file_path=csv_file_path)
        for movie in movie_list:
            if self.__weekly_box_office_data_folder.joinpath(f"{movie}{self.__downloaded_temp_file.suffix}").exists():
                logging.info(msg=f"{movie} is searched.")
                continue
            try:
                self.get_weekly_box_office_data(movie_name=movie)
            except AssertionError:
                download_fail_movie_list_path = self.__data_path.joinpath("Error_movies.txt")
                with open(file=download_fail_movie_list_path, mode='a') as file:
                    print(f"{movie}", file=file)
            finally:
                self.__browser.get(self.__defaults_url)

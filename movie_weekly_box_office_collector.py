from colab_browser import ColabBrowser

import csv
import time
import logging
from pathlib import Path
from selenium.webdriver.common.by import By
from selenium.common import exceptions as selenium_exceptions
from selenium.webdriver.remote.webelement import WebElement
from urllib3.exceptions import ReadTimeoutError


class MovieWeeklyBoxOfficeCollector:
    def __init__(self, page_changing_waiting_time: int = 2, download_waiting_time: int = 1) -> None:
        # browser setting
        self.__browser: ColabBrowser = ColabBrowser()
        self.__page_changing_waiting_time: int = page_changing_waiting_time
        self.__download_waiting_time: int = download_waiting_time

        # path dependent
        self.__data_path: Path = Path("data")
        self.__download_type: str = 'json'
        self.__downloaded_temp_file: Path = self.__data_path.joinpath(f"各週週末票房資料匯出.{self.__download_type}")
        self.__weekly_box_office_data_folder: Path = self.__data_path.joinpath("weekly_box_office_data",
                                                                               "by_movie_name")
        self.__file_of_searching_failed_movies: Path = self.__data_path.joinpath("searching_failed_movies.csv")

        # url
        self.__searching_url: str = "https://boxofficetw.tfai.org.tw/search/0"
        self.__defaults_url: str = "https://google.com"

        # csv dependent
        self.__csv_header: str = '片名'

    def __enter__(self) -> any:
        self.__browser = self.__browser.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> any:
        return self.__browser.__exit__(exc_type=exc_type, exc_val=exc_val, exc_tb=exc_tb)

    def initialize_failed_movie_file(self):
        self.__file_of_searching_failed_movies.touch(exist_ok=True)
        with open(self.__file_of_searching_failed_movies, mode='w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, [self.__csv_header])
            writer.writeheader()
        return

    def initialize_path_and_temp_files(self):
        # create path folder
        self.__weekly_box_office_data_folder.mkdir(parents=True, exist_ok=True)
        self.__downloaded_temp_file.unlink(missing_ok=True)
        self.__file_of_searching_failed_movies.unlink(missing_ok=True)

    def navigate_to_movie_page(self, target_movie_name: str) -> None:
        searching_url: str = f"{self.__searching_url}/{target_movie_name}"
        try:
            self.__browser.get(url=searching_url, waiting_time=self.__page_changing_waiting_time)
        except ReadTimeoutError:
            return
        buttons: list = self.__browser.find_elements(
            by=By.XPATH,
            value='//div[@id="film-searcher"]/div[@class="body"]/button/span[@class="name"]'
        )
        target_element: WebElement | None = next(
            (button.find_element(by=By.XPATH, value=f"./..") for button in buttons if button.text == target_movie_name),
            None,
        )
        try:
            target_element.click()
            logging.info(msg=f"the drop-down list button of {target_movie_name} is clicked.")
        except selenium_exceptions.ElementClickInterceptedException:
            logging.error(msg=f"Searching failed, the drop-down list button of {target_movie_name} cannot be clicked.",
                          exc_info=True)
            return
        except AttributeError:
            logging.critical(msg=f"Searching {target_movie_name} failed, none movie title drop-down list found.")
            return
        else:
            time.sleep(self.__page_changing_waiting_time)
            if self.__browser.current_url == self.__searching_url:
                logging.warning("No page changing detect.")
                raise AssertionError
            logging.debug(msg=f"goto url: {self.__browser.current_url}")
            return

    def download_weekly_box_office_data_from_csv_file(self, movie_name: str) -> None:
        try:
            self.navigate_to_movie_page(target_movie_name=movie_name)
        except (ReadTimeoutError, AssertionError, AttributeError,
                selenium_exceptions.ElementClickInterceptedException):
            logging.warning(msg=f"cannot enter to movie page.")
            raise AssertionError
        csv_button: WebElement = self.__browser.find_element(
            by=By.XPATH,
            value=f'//div[@id="export-button-container"]/button[@data-type="{self.__download_type.upper()}"]',
        )
        try:
            csv_button.click()
            logging.info(msg=f"{self.__download_type.upper()} button is clicked.")
        except (selenium_exceptions.ElementClickInterceptedException, AttributeError):
            logging.warning(msg=f"Download failed, the {self.__download_type.upper()} button cannot be clicked.",
                            exc_info=True)
            raise AssertionError
        else:
            time.sleep(self.__download_waiting_time)
            try:
                self.__downloaded_temp_file.replace(self.__weekly_box_office_data_folder.joinpath(
                    f"{movie_name}.{self.__download_type}"))
            except FileNotFoundError:
                while not self.__downloaded_temp_file.exists():
                    time.sleep(1)
                self.__downloaded_temp_file.unlink()
                self.__download_waiting_time = min(self.__download_waiting_time * 2, 120)
                logging.warning(f"Download time not enough.")
                raise AssertionError
            except OSError:
                self.__downloaded_temp_file.unlink()
                logging.warning(f"The filename \"{movie_name}.{self.__download_type}\" is incorrect.")
                raise
            else:
                if self.__download_waiting_time != 2:
                    self.__download_waiting_time = 2
                return

    @staticmethod
    def get_movie_list_from_file(csv_file_path: Path) -> list[str]:
        with open(file=csv_file_path, encoding='utf-8') as file:
            reader = csv.DictReader(file)
            return [row['片名'] for row in reader]

    def search_weekly_box_office_data(self, movie_name: str, trying_times: int = 10) -> None:
        logging.info(f"Searching box office of {movie_name}.")
        if self.__weekly_box_office_data_folder.joinpath(f"{movie_name}.{self.__download_type}").exists():
            logging.info(msg=f"found box office data from file. Searching finish.")
            return
        logging.info("Searching box office data from browser.")
        successful_flag: bool = False

        for index in range(trying_times):
            try:
                self.download_weekly_box_office_data_from_csv_file(movie_name=movie_name)
            except AssertionError:
                logging.warning(f"The {index} times of searching box office data failed.")
            except OSError:
                break
            else:
                successful_flag = True
                break
        if not successful_flag:
            if not self.__file_of_searching_failed_movies.exists():
                self.initialize_failed_movie_file()
            logging.info(f"Problem occurred when searching data, append movie {movie_name} into failed files.")
            with open(file=self.__file_of_searching_failed_movies, mode='a', newline='') as file:
                print(f"{movie_name}", file=file)
        self.__browser.get(self.__defaults_url)
        return

    def get_weekly_box_office_data(self, csv_file_path: Path) -> None:
        self.initialize_path_and_temp_files()
        movie_list: list[str] = self.get_movie_list_from_file(csv_file_path=csv_file_path)
        [self.search_weekly_box_office_data(movie_name=movie) for movie in movie_list]
        return

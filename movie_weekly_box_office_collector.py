from colab_browser import ColabBrowser

import time
import logging
from pathlib import Path
from selenium.webdriver.common.by import By
from selenium.common import exceptions as selenium_exceptions


class MovieWeeklyBoxOfficeCollector:
    def __init__(self, page_changing_waiting_time: int = 1, download_waiting_time: int = 2) -> None:
        self.__browser = ColabBrowser()
        self.__page_changing_waiting_time = page_changing_waiting_time
        self.__download_waiting_time = download_waiting_time
        self.__defaults_download_path = Path("data")
        self.__downloaded_temp_file = self.__defaults_download_path.joinpath("各週週末票房資料匯出.csv")
        self.__weekly_box_office_data_folder = self.__defaults_download_path.joinpath("weekly_box_office_data",
                                                                                      "by_movie_name")
        self.__weekly_box_office_data_folder.mkdir(parents=True, exist_ok=True)

        self.__searching_url = "https://boxofficetw.tfai.org.tw/search/0"

    def __enter__(self) -> any:
        self.__browser = self.__browser.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> any:
        return self.__browser.__exit__(
            exc_type=exc_type, exc_val=exc_val, exc_tb=exc_tb
        )

    def go_to_movie_page(self, target_movie_name: str) -> None:

        searching_url = f"{self.__searching_url}/{target_movie_name}"
        self.__browser.get(url=searching_url, waiting_time=self.__page_changing_waiting_time)
        buttons = self.__browser.find_elements(
            by=By.XPATH,
            value=f'//div[@id="film-searcher"]/'
                  f'div[@class="body"]/button/span[@class="name"]',
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
            logging.info(f"{target_movie_name} button clicked.")
            time.sleep(self.__page_changing_waiting_time)
        except selenium_exceptions.ElementClickInterceptedException:
            logging.error("Element cannot to click", exc_info=True)
            exit(2)
        except AttributeError:
            logging.warning(f"Searching {target_movie_name} failed, none button find.")
            exit(1)
        except Exception:
            logging.critical("Unknown Exception Catched.", exc_info=True)
            exit(-1)
        else:
            if self.__browser.current_url == self.__searching_url:
                raise AssertionError('No page switching detect.')
            return

    def get_weekly_box_office_data(self, movie_name: str, trying_times: int = 10) -> None:
        for index in range(trying_times):
            self.go_to_movie_page(target_movie_name=movie_name)

            csv_button = self.__browser.find_element(
                by=By.XPATH,
                value=f'//div[@id="export-button-container"]/button[@data-type="CSV"]',
            )

            try:
                csv_button.click()
                logging.info('csv button clicked.')
                time.sleep(self.__page_changing_waiting_time)
            except (selenium_exceptions.ElementClickInterceptedException, AttributeError):
                logging.warning("Element cannot to click", exc_info=True)
            except Exception:
                logging.critical("Unknown Exception Caught.", exc_info=True)
                exit(-1)
            else:
                time.sleep(self.__download_waiting_time)
                self.__downloaded_temp_file.replace(
                    self.__weekly_box_office_data_folder.joinpath(f"{movie_name}{self.__downloaded_temp_file.suffix}"))
                return
        raise AssertionError(f'Trying {trying_times} times but cannot click the button.')

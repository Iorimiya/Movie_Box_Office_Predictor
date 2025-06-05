from datetime import datetime
from enum import Enum
from logging import Logger
from pathlib import Path
import re
from typing import Final, Optional, TypeAlias

from bs4 import BeautifulSoup
from bs4.element import NavigableString
import requests
from requests import Response
from selenium.common.exceptions import StaleElementReferenceException
from tqdm import tqdm

from src.core.constants import Constants
from src.core.logging_manager import LoggingManager
from src.data_collection.browser import CaptchaBrowser
from src.data_handling.movie_data import load_index_file, MovieData, PublicReview
from src.utilities.util import delete_duplicate

Url: TypeAlias = str
RegularExpressionPattern: TypeAlias = str
Selector: TypeAlias = str


class ReviewCollector:
    """
    A class to collect reviews from various websites.
    """

    class TargetWebsite(Enum):
        """
        Enum representing the target website for review collection.
        """
        PTT = 0
        DCARD = 1
        IMDB = 2
        ROTTEN_TOMATO = 3

    def __init__(self, target_website: TargetWebsite):
        """
        Initializes the ReviewCollector.

        Args:
            target_website (TargetWebsite): The target website for review collection.
        """
        self.__search_target: TargetWebsite = target_website
        self.__base_url: tuple[str, str, str, str] = ('https://www.ptt.cc/bbs/movie/', 'https://www.dcard.tw/',
                                                      'https://www.imdb.com/', 'https://www.rottentomatoes.com/')
        self.__search_url_part: tuple[str, str, str, str] = \
            ('search?q=', 'search?forum=movie&query=', 'find/?q=', 'search?search=')
        self.__browser: Optional[CaptchaBrowser] = None
        self.__logger: Logger = LoggingManager().get_logger('root')
        self.__logger.info(f"Download {self.__search_target.name} data.")

    @staticmethod
    def get_movie_search_keys(movie_name: str) -> list[str]:
        """
        Generates search keys for a movie.

        Args:
            movie_name (str): The name of the movie.

        Returns:
            list[str]: A list of search keys.
        """
        LoggingManager().get_logger('root').info("Creating search keys.")
        output = list()

        space: Final[RegularExpressionPattern] = " "
        empty: Final[RegularExpressionPattern] = ""
        dash: Final[RegularExpressionPattern] = "-"
        double_quotation: Final[RegularExpressionPattern] = "\""
        # these elements need to be deleted first
        delete_pattern: Final[RegularExpressionPattern] = \
            "[\(（]((數位)?\s?(修復)?\s?(IMAX|A)?\s?((日|英|國)(文|語))?\s?((2|3)D)?\s?版?)+[\)）]$"

        # original version, space version,pattern-deleted version, dash version and subtitle version
        separator_pattern: Final[RegularExpressionPattern] = "\s*[：:\-－]\s*"
        # original version and pattern-deleted version
        dual_double_quotation_pattern: Final[RegularExpressionPattern] = "\"{2}"
        start_with_double_quotation_pattern: Final[RegularExpressionPattern] = "^\""
        end_with_double_quotation_pattern: Final[RegularExpressionPattern] = "\"$"
        # space number (original version and space-deleted version)
        space_with_number_pattern: Final[RegularExpressionPattern] = " \d"

        movie_name: str = re.sub(pattern=delete_pattern, repl=empty, string=movie_name)
        output.append(movie_name)
        if re.search(pattern=separator_pattern, string=movie_name) is not None:
            output.append(re.sub(pattern=separator_pattern, repl=empty, string=movie_name))
            output.append(re.sub(pattern=separator_pattern, repl=space, string=movie_name))
            output.append(re.sub(pattern=separator_pattern, repl=dash, string=movie_name))
            output.append(re.split(pattern=separator_pattern, string=movie_name)[1])
        if re.search(pattern=double_quotation, string=movie_name) is not None:
            if re.search(pattern=dual_double_quotation_pattern, string=movie_name) is not None:
                temp2 = re.sub(pattern=end_with_double_quotation_pattern, repl=empty,
                               string=re.sub(pattern=start_with_double_quotation_pattern, repl=empty,
                                             string=movie_name))
                output.append(re.sub(pattern=dual_double_quotation_pattern, repl=double_quotation,
                                     string=temp2))
                output.append(re.sub(pattern=dual_double_quotation_pattern, repl=space,
                                     string=temp2))
            output.append(re.sub(pattern=double_quotation, repl=empty, string=movie_name))
        if re.search(pattern=space_with_number_pattern, string=movie_name) is not None:
            output.append(re.sub(pattern=space, repl=empty, string=movie_name))
        LoggingManager().get_logger('root').info("Creation of search keys finish.")
        return output

    __get_search_page_url = lambda self, search_key: \
        f"{self.__base_url[self.__search_target.value]}{self.__search_url_part[self.__search_target.value]}{search_key}"
    """
    Constructs the search page URL.
    
    Args:
        self: The ReviewCollector instance.
        search_key: The search key.
    
    Returns:
        The search page URL.
    """

    def __get_bs_element(self, url: str) -> BeautifulSoup:
        """
        Gets a BeautifulSoup element from a URL.

        Args:
            url (str): The URL.

        Returns:
            BeautifulSoup: The BeautifulSoup element.
        """
        # PTT在特定的板中需要over18=1這個cookies
        match self.__search_target:
            case TargetWebsite.PTT:
                response: Response = requests.get(url=url, cookies={'over18': '1'})
            case _:
                response: Response = requests.get(url=url)
        response.encoding = response.apparent_encoding
        return BeautifulSoup(response.text, features='html.parser')

    def __get_largest_result_page_number(self, bs_root_element: BeautifulSoup) -> int:
        """
        Gets the largest result page number from a BeautifulSoup element.

        Args:
            bs_root_element (BeautifulSoup): The BeautifulSoup element.

        Returns:
            int: The largest result page number.

        Raises:
            ValueError: If the target website is not PTT.
        """
        match self.__search_target:
            case TargetWebsite.PTT:
                re_pattern: Final[RegularExpressionPattern] = "page=(\d+)"
                key_word: Final[str] = '最舊'
                selector: Final[Selector] = "#action-bar-container a"
                last_page_url = \
                    [element['href'] for element in bs_root_element.select(selector) if element.text == key_word][0]
                return int(re.split(pattern='=', string=re.search(pattern=re_pattern, string=last_page_url)[0])[1])
            case _:
                raise ValueError

    def __get_review_urls(self, search_key: str) -> list[str]:
        """
        Gets review URLs for a search key.

        Args:
            search_key (str): The search key.

        Returns:
            list[str]: A list of review URLs.

        Raises:
            ValueError: If the target website is not PTT or DCARD.
        """
        search_url: Url = self.__get_search_page_url(search_key)
        match self.__search_target:
            case TargetWebsite.PTT:
                try:
                    max_page_number: int = self.__get_largest_result_page_number(self.__get_bs_element(search_url))
                except IndexError:
                    return list()
                self.__logger.info(f"Find {max_page_number} pages of result.")
                domain_pattern: Final[RegularExpressionPattern] = '^[^:\/]+:\/\/[^\/]+'
                base_url = re.search(pattern=domain_pattern, string=self.__base_url[self.__search_target.value]).group(
                    0)
                selector: Selector = "#main-container div.r-list-container div.title a"
                urls: list[str] = [base_url + review['href']
                                   for current_page_number in range(1, max_page_number + 1)
                                   for review in
                                   self.__get_bs_element(f"{search_url}&page={current_page_number}").select(selector)]

            case TargetWebsite.DCARD:
                self.__browser.get(search_url, captcha=True)
                selector: Selector = "div#__next div[role='main'] div[data-key] article[role='article'] h2 a[href]"
                scroll_height: int = int(self.__browser.find_element(selector="body").get_attribute("scrollHeight"))
                urls = list()
                for current_height in range(0, scroll_height, 150):
                    self.__browser.execute_script(f"window.scrollTo(0,{current_height})")
                    new_urls = list()
                    for element in self.__browser.find_elements(selector=selector):
                        try:
                            # selenium.common.exceptions.StaleElementReferenceException
                            href = element.get_attribute("href")
                        except StaleElementReferenceException:
                            self.__logger.error(f"Cannot locale element on {search_key}, height {current_height}.",
                                                exc_info=True)
                        else:
                            new_urls.append(href)
                    urls.extend(url for url in new_urls)
                urls = delete_duplicate(urls)

            case _:
                raise ValueError
        self.__logger.info(f"{len(urls)} urls found.")
        return urls

    def __get_review_information(self, url: str) -> PublicReview | None:
        """
        Gets review information from a URL.

        Args:
            url (str): The review URL.

        Returns:
            PublicReview | None: The review information or None if an error occurs.
        """
        self.__logger.info(f"Search review information for \"{url}\".")
        title: str | None = None
        content: str | None = None
        replies: list[str] | None = None
        posted_time: datetime | None = None
        match self.__search_target:
            case TargetWebsite.PTT:
                meta_element_selector: Final[Selector] = '.article-metaline'
                meta_tag_selector: Final[Selector] = '.article-meta-tag'
                meta_value_selector: Final[Selector] = '.article-meta-value'
                time_format: Final[str] = '%a %b %d %H:%M:%S %Y'
                key_words: Final[tuple[str, str]] = ('標題', '時間')
                try:
                    content_base_element: BeautifulSoup | None = self.__get_bs_element(url=url).select_one(
                        selector='#main-content')
                    article_meta_elements = content_base_element.select(selector=meta_element_selector)
                    for article_meta_element in article_meta_elements:
                        if article_meta_element.select_one(selector=meta_tag_selector).text == key_words[0]:
                            title = article_meta_element.select_one(selector=meta_value_selector).text
                        elif article_meta_element.select_one(selector=meta_tag_selector).text == key_words[1]:
                            posted_time = datetime.strptime(
                                article_meta_element.select_one(selector=meta_value_selector).text,
                                time_format)
                    content = ''.join(
                        [element for element in content_base_element if
                         isinstance(element, NavigableString)]).strip()
                    replies = [reply.text for reply in content_base_element.select(selector='.push .push-content')]
                    if not (title and posted_time and content and replies):
                        return None
                except Exception as e:
                    self.__logger.warning(f"Cannot search element in url:\"{url}\".")
                    self.__logger.error(f"Error message: {e}")
                    return None
            case TargetWebsite.DCARD:
                selector_base: Final[Selector] = "div#__next div[role='main']"
                selector_title: Final[Selector] = selector_base + " article h1"
                selector_time: Final[Selector] = selector_base + " article time"
                selector_content: Final[Selector] = selector_base + " article span"
                selector_reply: Final[Selector] = selector_base + " section div[data-key^='comment'] span:not([class])"
                time_format: Final[str] = '%Y 年 %m 月 %d 日 %H:%M'

                self.__browser.home()
                self.__browser.get(url=url, captcha=True)

                try:

                    title = self.__browser.find_element(selector=selector_title).text
                    posted_time = datetime.strptime(
                        self.__browser.find_element(selector=selector_time).text,
                        time_format)
                    content = self.__browser.find_element(selector=selector_content).text
                    # selenium.common.exceptions.InvalidSelectorException
                    scroll_height: int = int(self.__browser.find_element(selector="body").get_attribute("scrollHeight"))
                    replies_list = list()
                    for current_height in range(0, scroll_height, 150):
                        self.__browser.execute_script(f"window.scrollTo(0,{current_height})")
                        # 　selenium.common.exceptions.StaleElementReferenceException
                        replies = [reply_element.text for reply_element in
                                   self.__browser.find_elements(selector=selector_reply)]
                        replies_list.extend(replies)
                except Exception as e:
                    self.__logger.warning(f"Cannot search element in url:\"{url}\".")
                    self.__logger.error(f"Error message: {e}")
                    return None

        return PublicReview(url=url, title=title, content=content, date=posted_time.date(), reply_count=len(replies),
                            sentiment_score=False)

    def __get_reviews_by_keyword(self, search_key: str) -> list[PublicReview]:
        """
        Gets reviews by keyword.

        Args:
            search_key (str): The search key.

        Returns:
            list[PublicReview]: A list of reviews.

        Raises:
            ValueError: If the target website is not PTT or DCARD.
        """
        match self.__search_target:
            case TargetWebsite.PTT | TargetWebsite.DCARD:
                self.__logger.info(f"Start search reviews with search key \"{search_key}.")
                urls: list[str] = [url for url in self.__get_review_urls(search_key=search_key)]
                return list(filter(lambda x: x, [self.__get_review_information(url=url) for url in
                                                 tqdm(urls, desc='review urls',
                                                      bar_format=Constants.STATUS_BAR_FORMAT)]))
            case _:
                raise ValueError

    def __get_reviews_by_name(self, movie_name: str) -> list[PublicReview] | None:
        """
        Gets reviews by movie name.

        Args:
            movie_name (str): The movie name.

        Returns:
            list[PublicReview] | None: A list of reviews or None if an error occurs.
        """
        search_keys: list[str] = self.get_movie_search_keys(movie_name=movie_name)
        reviews: list[PublicReview] = [review for search_key in search_keys for review in
                                       self.__get_reviews_by_keyword(search_key=search_key)]
        self.__logger.info("Trying to delete duplicate reviews.")
        reviews = delete_duplicate(reviews)
        self.__logger.info("Deletion of duplicate reviews finished.")
        return reviews

    def __search_review_and_save(self, movie_list: list[MovieData], save_folder_path: Path) -> None:
        """
        Searches reviews for a list of movies and saves them.

        Args:
            movie_list (list[MovieData]): The list of movies.
            save_folder_path (Path): The path to save the reviews.
        """
        for movie in tqdm(movie_list, desc='movies', bar_format=Constants.STATUS_BAR_FORMAT):
            reviews: list[PublicReview] = self.__get_reviews_by_name(movie_name=movie.movie_name)
            if save_folder_path.joinpath(f"{movie.movie_id}.{Constants.DEFAULT_SAVE_FILE_EXTENSION}").exists():
                movie.load_public_review()
            movie.update_data(public_reviews=reviews)
            movie.save_public_review(save_folder_path=save_folder_path)

    def search_review_with_single_movie(self, movie_data: str | MovieData) -> list[PublicReview] | None:
        """
        Searches reviews for a single movie.

        Args:
            movie_data (str | MovieData): The movie data.

        Returns:
            list[PublicReview] | None: A list of reviews or None if an error occurs.

        Raises:
            ValueError: If movie_data is not a string or MovieData.
        """
        if isinstance(movie_data, MovieData):
            movie_name = movie_data.movie_name
        elif isinstance(movie_data, str):
            movie_name = movie_data
        else:
            raise ValueError
        match self.__search_target:
            case TargetWebsite.PTT:
                reviews: list[PublicReview] = self.__get_reviews_by_name(movie_name=movie_name)
            case TargetWebsite.DCARD:
                with CaptchaBrowser() as self.__browser:
                    reviews: list[PublicReview] = self.__get_reviews_by_name(movie_name=movie_name)
            case _:
                raise ValueError
        if isinstance(movie_data, MovieData):
            movie_data.update_data(public_reviews=reviews)
            return None
        elif isinstance(movie_data, str):
            return reviews
        else:
            raise ValueError

    def search_review_with_multiple_movie(self, index_path: Path = Constants.INDEX_PATH,
                                          save_folder_path: Path = None):
        """
        Searches reviews for multiple movies.

        Args:
            index_path (Path): The path to the index file. Defaults to Constants.INDEX_PATH.
            save_folder_path (Path): The path to save the reviews. Defaults to None.
        """
        # with CaptchaBrowser() as self.__browser:
        if save_folder_path is None:
            save_folder_path = Constants.PUBLIC_REVIEW_FOLDER
        if not save_folder_path.exists():
            save_folder_path.mkdir(parents=True)
        movie_data: list[MovieData] = load_index_file(file_path=index_path)
        match self.__search_target:
            case TargetWebsite.PTT:
                self.__search_review_and_save(movie_list=movie_data, save_folder_path=save_folder_path)
            case TargetWebsite.DCARD:
                with CaptchaBrowser() as self.__browser:
                    self.__search_review_and_save(movie_list=movie_data, save_folder_path=save_folder_path)


TargetWebsite: TypeAlias = ReviewCollector.TargetWebsite

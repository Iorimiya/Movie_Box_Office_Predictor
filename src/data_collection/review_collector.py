import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from logging import Logger
from pathlib import Path
from typing import Final, Optional, TypeAlias, Iterator

import requests
from bs4 import BeautifulSoup
from bs4.element import NavigableString
from requests import Response
from selenium.common.exceptions import StaleElementReferenceException
from tqdm import tqdm
from yaml import YAMLError

from src.core.constants import Constants
from src.core.logging_manager import LoggingManager
from src.data_collection.browser import CaptchaBrowser
from src.data_handling.file_io import YamlFile
from src.data_handling.movie_collections import MovieData
from src.data_handling.reviews import PublicReview
from src.utilities.collection_utils import delete_duplicate

Url: TypeAlias = str
RegularExpressionPattern: TypeAlias = str
Selector: TypeAlias = str


@dataclass(kw_only=True, frozen=True)
class _WebsiteConfig:
    """
    A data class to store configuration for a target website.

    :ivar base_url: The base URL of the website.
    :ivar search_url_part: The part of the URL used for searching.
    """
    base_url: str
    search_url_part: str


class TargetWebsite(Enum):
    """
    Enum representing the target website for review collection.

    Each member holds a _WebsiteConfig object containing site-specific URLs.
    """
    PTT = _WebsiteConfig(
        base_url='https://www.ptt.cc/bbs/movie/',
        search_url_part='search?q='
    )
    DCARD = _WebsiteConfig(
        base_url='https://www.dcard.tw/',
        search_url_part='search?forum=movie&query='
    )
    IMDB = _WebsiteConfig(
        base_url='https://www.imdb.com/',
        search_url_part='find/?q='
    )
    ROTTEN_TOMATO = _WebsiteConfig(
        base_url='https://www.rottentomatoes.com/',
        search_url_part='search?search='
    )


class ReviewCollector:
    """
    A class to collect reviews from various websites.

    It supports fetching reviews from PTT and Dcard, handling search key generation,
    URL retrieval, and parsing review content.

    :ivar __logger: The logger instance for logging messages.
    :ivar __search_target: The target website for review collection.
    """

    def __init__(self, target_website: TargetWebsite) -> None:
        """
        Initializes the ReviewCollector.

        :param target_website: The target website for review collection.
        """
        self.__logger: Logger = LoggingManager().get_logger('root')
        self.__search_target: TargetWebsite = target_website
        self.__logger.info(f"ReviewCollector initialized for target: {self.__search_target.name}")

    @staticmethod
    def get_movie_search_keys(movie_name: str) -> list[str]:
        """
        Generates a list of potential search keys for a movie name.

        This method applies various transformations to the input movie name,
        such as removing specific patterns (e.g., "數位修復版"),
        handling separators (e.g., ':', '-'), and dealing with quotation marks
        and spaces around numbers, to create a comprehensive list of search terms.

        :param movie_name: The original name of the movie.
        :returns: A list of generated search key strings.
        """
        LoggingManager().get_logger('root').info(f"Creating search keys for '{movie_name}'.")
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
        LoggingManager().get_logger('root').info(f"Finished creating {len(output)} search keys.")
        return output

    def __get_search_page_url(self, search_key: str) -> str:
        """
        Constructs the search page URL for the configured target website and a given search key.

        :param search_key: The search key (e.g., movie name) to be used in the URL.
        :returns: The fully constructed search page URL.
        """
        return f"{self.__search_target.value.base_url}{self.__search_target.value.search_url_part}{search_key}"

    def __get_bs_element(self, url: str) -> BeautifulSoup:
        """
        Fetches the content from a given URL and parses it into a ``BeautifulSoup`` object.

        Handles specific cookie requirements for certain sites (e.g., 'over18' for PTT).
        The response encoding is set based on ``response.apparent_encoding``.

        :param url: The URL from which to fetch the content.
        :returns: A ``BeautifulSoup`` object representing the parsed HTML content.
        :raises requests.exceptions.RequestException: For issues during the HTTP request (e.g., network problems, invalid URL).
        """
        # Some boards on PTT require the over18=1 cookie.
        match self.__search_target:
            case TargetWebsite.PTT:
                response: Response = requests.get(url=url, cookies={'over18': '1'})
            case _:
                response: Response = requests.get(url=url)
        response.encoding = response.apparent_encoding
        return BeautifulSoup(response.text, features='html.parser')

    def __get_largest_result_page_number(self, bs_root_element: BeautifulSoup) -> int:
        """
        Extracts the largest result page number from a PTT search results page.

        This method is specific to PTT's HTML structure for search pagination.
        It looks for a link with the text '最舊' to find the URL of the last page
        and then extracts the page number from that URL.

        :param bs_root_element: The ``BeautifulSoup`` object of a PTT search results page.
        :returns: The largest page number found in the pagination.
        :raises ValueError: If the ``self.__search_target`` is not ``TargetWebsite.PTT``.
        :raises IndexError: If the '最舊' link or the page number pattern is not found,
                            indicating an unexpected page structure or no pagination.
        :raises AttributeError: If the page number pattern is found but cannot be correctly parsed.
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

    def __get_review_urls(self, search_key: str, browser: Optional[CaptchaBrowser]) -> list[str]:
        """
        Gets a list of review URLs for a given search key from the target website.

        For PTT, it fetches URLs from all pages of search results.
        For Dcard, it uses a Selenium browser to scroll through search results and extract URLs.

        :param search_key: The search key (e.g., movie name) to find reviews for.
        :param browser: The browser instance to use for Dcard, or None for PTT.
        :returns: A list of unique review URLs. Returns an empty list if no URLs are found
                  or if an error occurs during PTT page number retrieval.
        :raises ValueError: If the ``self.__search_target`` is not ``TargetWebsite.PTT`` or ``TargetWebsite.DCARD``.
        :raises RuntimeError: If Dcard collection is attempted without a browser instance.
        """
        search_url: Url = self.__get_search_page_url(search_key=search_key)
        match self.__search_target:
            case TargetWebsite.PTT:
                try:
                    max_page_number: int = self.__get_largest_result_page_number(self.__get_bs_element(url=search_url))
                except IndexError:
                    return list()
                self.__logger.info(f"Found {max_page_number} pages of results for '{search_key}'.")
                domain_pattern: Final[RegularExpressionPattern] = '^[^:\/]+:\/\/[^\/]+'
                base_url: str = re.search(pattern=domain_pattern, string=self.__search_target.value.base_url).group(0)
                selector: Selector = "#main-container div.r-list-container div.title a"
                urls: list[str] = [base_url + review['href']
                                   for current_page_number in range(1, max_page_number + 1)
                                   for review in
                                   self.__get_bs_element(url=f"{search_url}&page={current_page_number}").select(
                                       selector)]

            case TargetWebsite.DCARD:
                if not browser:
                    raise RuntimeError("A CaptchaBrowser instance is required for Dcard collection.")
                browser.get(url=search_url, captcha=True)
                selector: Selector = "div#__next div[role='main'] div[data-key] article[role='article'] h2 a[href]"
                scroll_height: int = int(browser.find_element(selector="body").get_attribute("scrollHeight"))
                urls = list()
                for current_height in range(0, scroll_height, 150):
                    browser.execute_script(f"window.scrollTo(0,{current_height})")
                    new_urls = list()
                    for element in browser.find_elements(selector=selector):
                        try:
                            href: str = element.get_attribute("href")
                        except StaleElementReferenceException:
                            self.__logger.error(
                                f"StaleElementReferenceException for '{search_key}' at height {current_height}.",
                                exc_info=True)
                        else:
                            new_urls.append(href)
                    urls.extend(url for url in new_urls)
                urls = delete_duplicate(items=urls)

            case _:
                raise ValueError(f"Unsupported target website: {self.__search_target.name}")
        self.__logger.info(f"Found {len(urls)} review URLs for '{search_key}'.")
        return urls

    def __get_review_information(self, url: str, browser: Optional[CaptchaBrowser]) -> Optional[PublicReview]:
        """
        Extracts review information (title, content, post time, replies) from a given review URL.

        Supports PTT and Dcard. For Dcard, it uses a Selenium browser.
        The extracted information is used to create a ``PublicReview`` object.

        :param url: The URL of the review page.
        :param browser: The browser instance to use for Dcard, or None for PTT.
        :returns: A ``PublicReview`` object containing the extracted information,
                  or ``None`` if essential information cannot be found or an error occurs during parsing.
        :raises RuntimeError: If Dcard collection is attempted without a browser instance.
        """
        self.__logger.info(f"Fetching review information from: \"{url}\".")
        title: Optional[str] = None
        content: Optional[str] = None
        replies: Optional[list[str]] = None
        posted_time: Optional[datetime] = None
        match self.__search_target:
            case TargetWebsite.PTT:
                meta_element_selector: Final[Selector] = '.article-metaline'
                meta_tag_selector: Final[Selector] = '.article-meta-tag'
                meta_value_selector: Final[Selector] = '.article-meta-value'
                time_format: Final[str] = '%a %b %d %H:%M:%S %Y'
                key_words: Final[tuple[str, str]] = ('標題', '時間')
                try:
                    content_base_element: Optional[BeautifulSoup] = self.__get_bs_element(url=url).select_one(
                        selector='#main-content')
                    if not content_base_element:
                        self.__logger.warning(f"Could not find main content element in URL: \"{url}\".")
                        return None
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
                    if not (title and posted_time and content):
                        return None
                except Exception as e:
                    self.__logger.warning(f"Cannot parse element in URL: \"{url}\".")
                    self.__logger.error(f"Error message: {e}", exc_info=True)
                    return None
            case TargetWebsite.DCARD:
                if not browser:
                    raise RuntimeError("A CaptchaBrowser instance is required for Dcard collection.")
                selector_base: Final[Selector] = "div#__next div[role='main']"
                selector_title: Final[Selector] = selector_base + " article h1"
                selector_time: Final[Selector] = selector_base + " article time"
                selector_content: Final[Selector] = selector_base + " article span"
                selector_reply: Final[Selector] = selector_base + " section div[data-key^='comment'] span:not([class])"
                time_format: Final[str] = '%Y 年 %m 月 %d 日 %H:%M'

                browser.home()
                browser.get(url=url, captcha=True)

                try:
                    title = browser.find_element(selector=selector_title).text
                    posted_time = datetime.strptime(
                        browser.find_element(selector=selector_time).text,
                        time_format)
                    content = browser.find_element(selector=selector_content).text
                    scroll_height: int = int(browser.find_element(selector="body").get_attribute("scrollHeight"))
                    replies_list = list()
                    for current_height in range(0, scroll_height, 150):
                        browser.execute_script(f"window.scrollTo(0,{current_height})")
                        replies = [reply_element.text for reply_element in
                                   browser.find_elements(selector=selector_reply)]
                        replies_list.extend(replies)
                    replies = replies_list
                except Exception as e:
                    self.__logger.warning(f"Cannot find element in URL: \"{url}\".")
                    self.__logger.error(f"Error message: {e}", exc_info=True)
                    return None

        if title and content and posted_time:
            return PublicReview(url=url, title=title, content=content, date=posted_time.date(),
                                reply_count=len(replies or []),
                                sentiment_score=None)
        return None

    def __get_reviews_by_keyword(self, search_key: str, browser: Optional[CaptchaBrowser]) -> list[PublicReview]:
        """
        Retrieves and processes all public reviews found using a specific search keyword.

        This method first gets all review URLs for the given ``search_key`` using ``__get_review_urls``,
        then iterates through these URLs to fetch detailed review information using ``__get_review_information``.
        Only successfully parsed reviews are returned.

        :param search_key: The keyword to search for reviews.
        :param browser: The browser instance to use for Dcard, or None for PTT.
        :returns: A list of ``PublicReview`` objects.
        :raises ValueError: If the ``self.__search_target`` is not ``TargetWebsite.PTT`` or ``TargetWebsite.DCARD``
                            (propagated from ``__get_review_urls``).
        """
        self.__logger.info(f"Searching reviews with keyword: \"{search_key}\".")
        urls: list[str] = self.__get_review_urls(search_key=search_key, browser=browser)
        return list(filter(None, [self.__get_review_information(url=url, browser=browser) for url in
                                  tqdm(urls, desc=f'Fetching reviews for "{search_key}"',
                                       bar_format=Constants.STATUS_BAR_FORMAT)]))

    def __get_reviews_by_name(self, movie_name: str, browser: Optional[CaptchaBrowser]) -> list[PublicReview]:
        """
        Gets all unique public reviews for a given movie name.

        It first generates multiple search keys from the ``movie_name`` using ``get_movie_search_keys``.
        Then, for each search key, it fetches reviews using ``__get_reviews_by_keyword``.
        Finally, it consolidates and de-duplicates all found reviews.

        :param movie_name: The name of the movie.
        :param browser: The browser instance to use for Dcard, or None for PTT.
        :returns: A list of unique ``PublicReview`` objects. Returns an empty list if no reviews are found.
        """
        search_keys: list[str] = self.get_movie_search_keys(movie_name=movie_name)
        reviews: list[PublicReview] = [review for search_key in search_keys for review in
                                       self.__get_reviews_by_keyword(search_key=search_key, browser=browser)]
        self.__logger.info(f"Found {len(reviews)} raw reviews for '{movie_name}'. De-duplicating...")
        reviews = delete_duplicate(items=reviews)
        self.__logger.info(f"Finished with {len(reviews)} unique reviews for '{movie_name}'.")
        return reviews

    @contextmanager
    def _managed_browser_session(self) -> Iterator[Optional[CaptchaBrowser]]:
        """
        A context manager that provides a CaptchaBrowser session only if the target is DCARD.

        This centralizes the logic for browser instantiation and cleanup. For non-DCARD
        targets, it yields None and does nothing.

        :yields: A ``CaptchaBrowser`` instance if the target is DCARD, otherwise ``None``.
        """
        if self.__search_target == TargetWebsite.DCARD:
            browser: Optional[CaptchaBrowser] = CaptchaBrowser()
            try:
                browser.__enter__()
                yield browser
            finally:
                browser.__exit__(None, None, None)
        else:
            yield None

    def collect_reviews_for_movie(self, movie_name: str) -> list[PublicReview]:
        """
        Searches for public reviews for a single movie by name.

        This method fetches reviews for the specified movie. If no reviews are found,
        an empty list is returned. Any errors during the collection process will be raised.

        :param movie_name: The name of the movie.
        :returns: A list of PublicReview objects. Returns an empty list if no reviews are found.
        :raises Exception: Propagates any exceptions encountered during the review collection process.
        """
        self.__logger.info(f"Starting single collection for movie: '{movie_name}'.")
        with self._managed_browser_session() as browser:
            reviews: list[PublicReview] = self.__get_reviews_by_name(movie_name=movie_name, browser=browser)
        self.__logger.info(f"Finished collecting {len(reviews)} reviews for '{movie_name}'.")
        return reviews

    def collect_reviews_for_movies(self, movie_list: list[MovieData], data_folder: Path) -> None:
        """
        Collects public reviews for a list of movies and saves them to files.

        For each movie, it fetches reviews. If no reviews are found, an empty YAML file
        is created. If an error occurs during collection for a movie, it is logged,
        and the process continues to the next movie. CaptchaBrowser is instantiated
        only once for Dcard collection across all movies.

        :param movie_list: A list of MovieData objects for which to collect reviews.
        :param data_folder: The directory where the final data will be saved.
        """
        self.__logger.info(f"Starting batch review collection for {len(movie_list)} movies.")
        data_folder.mkdir(parents=True, exist_ok=True)

        with self._managed_browser_session() as browser:
            for movie in tqdm(movie_list, desc='Collecting Reviews', bar_format=Constants.STATUS_BAR_FORMAT):
                self.__logger.debug(f"Processing reviews for movie ID {movie.id} ('{movie.name}').")
                try:
                    newly_fetched_reviews: list[PublicReview] = self.__get_reviews_by_name(
                        movie_name=movie.name, browser=browser
                    )

                    movie.update_public_reviews(update_method='EXTEND', data=newly_fetched_reviews)
                    saved_path: Path = movie.save_public_reviews(target_directory=data_folder)

                    if not newly_fetched_reviews:
                        self.__logger.info(
                            f"No new reviews found for movie ID {movie.id}. Empty file created at '{saved_path}'.")
                    else:
                        self.__logger.info(
                            f"Successfully collected and saved {len(newly_fetched_reviews)} reviews for movie ID {movie.id} to '{saved_path}'.")

                except Exception as e:
                    self.__logger.error(
                        f"Failed to collect reviews for movie ID {movie.id} ('{movie.name}'): {e}",
                        exc_info=True
                    )
                    empty_file_path: Path = data_folder / f"{movie.id}.yaml"
                    try:
                        YamlFile(path=empty_file_path).save(data=[])
                        self.__logger.info(
                            f"Created empty review file for failed movie ID {movie.id} at '{empty_file_path}'.")
                    except (OSError, YAMLError) as file_e:
                        self.__logger.error(
                            f"Failed to create empty review file for movie ID {movie.id}: {file_e}",
                            exc_info=True)

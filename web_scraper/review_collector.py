import logging
import re
from tqdm import tqdm
from datetime import datetime
from enum import Enum
from typing import TypeAlias, Final

import requests
from bs4 import BeautifulSoup
from bs4.element import NavigableString
from requests import Response

from selenium.common.exceptions import *

from movie_data import PublicReview
from tools.util import *
from web_scraper.browser import CaptchaBrowser

Url: TypeAlias = str
RegularExpressionPattern: TypeAlias = str
Selector: TypeAlias = str


class ReviewCollector:
    class TargetWebsite(Enum):
        PPT = 0
        DCARD = 1
        IMDB = 2
        ROTTEN_TOMATO = 3

    def __init__(self, target_website: TargetWebsite):
        self.__search_target: TargetWebsite = target_website
        self.__base_url: tuple[str, str, str, str] = ('https://www.ptt.cc/bbs/movie/', 'https://www.dcard.tw/',
                                                      'https://www.imdb.com/', 'https://www.rottentomatoes.com/')
        self.__search_url_part: tuple[str, str, str, str] = ('search?q=', 'search?query=', 'find/?q=', 'search?search=')
        self.__browser: Optional[CaptchaBrowser] = None
        logging.info(f"download {self.__search_target.name} data.")

    @staticmethod
    def get_movie_search_keys(movie_name: str) -> list[str]:
        logging.info("creating search keys.")
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
        logging.info("creation of search keys finish.")
        return output

    __get_search_page_url = lambda self, search_key: \
        f"{self.__base_url[self.__search_target.value]}{self.__search_url_part[self.__search_target.value]}{search_key}"

    def __get_bs_element(self, url: str) -> BeautifulSoup:
        # PTT在特定的板中需要over18=1這個cookies
        match self.__search_target:
            case TargetWebsite.PPT:
                response: Response = requests.get(url=url, cookies={'over18': '1'})
            case _:
                response: Response = requests.get(url=url)
        response.encoding = response.apparent_encoding
        return BeautifulSoup(response.text, features='html.parser')

    def __get_largest_result_page_number(self, bs_root_element: BeautifulSoup) -> int:
        match self.__search_target:
            case TargetWebsite.PPT:
                re_pattern: Final[RegularExpressionPattern] = "page=(\d+)"
                key_word: Final[str] = '最舊'
                selector: Final[Selector] = "#action-bar-container a"
                last_page_url = \
                    [element['href'] for element in bs_root_element.select(selector) if element.text == key_word][0]
                return int(re.split(pattern='=', string=re.search(pattern=re_pattern, string=last_page_url)[0])[1])
            case _:
                raise ValueError

    def __get_review_urls(self, search_key: str) -> list[str]:
        search_url: Url = self.__get_search_page_url(search_key)
        match self.__search_target:
            case TargetWebsite.PPT:
                max_page_number: int = self.__get_largest_result_page_number(self.__get_bs_element(search_url))
                logging.info(f"find {max_page_number} pages of result.")
                domain_pattern: Final[RegularExpressionPattern] = '^[^:\/]+:\/\/[^\/]+'
                base_url = re.search(pattern=domain_pattern, string=self.__base_url[self.__search_target.value]).group(
                    0)
                selector: Selector = "#main-container div.r-list-container div.title a"
                urls:list[str] = [base_url + review['href']
                        for current_page_number in range(1, max_page_number + 1)
                        for review in
                        self.__get_bs_element(f"{search_url}&page={current_page_number}").select(selector)]
                
            case TargetWebsite.DCARD:
                self.__browser.get(search_url)
                selector: Selector = "div#__next div[role='main'] div[data-key] article[role='article'] h2 a[href]"
                scroll_height: int = int(self.__browser.find_element(selector="body").get_attribute("scrollHeight"))
                urls = list()
                for current_height in range(0, scroll_height, 150):
                    self.__browser.execute_script(f"window.scrollTo(0,{current_height})")
                    #　selenium.common.exceptions.StaleElementReferenceException
                    new_urls = list()
                    try:

                        for element in self.__browser.find_elements(selector=selector):
                            href = element.get_attribute("href")
                            new_urls.append(href)
                    except StaleElementReferenceException:
                        a=1
                        raise
                    urls.extend(url for url in new_urls)
                urls = self.__delete_duplicate(urls)

            case _:
                raise ValueError
        logging.info(f"{len(urls)} urls found.")
        return urls

    def __get_review_information(self, url: str) -> PublicReview | None:
        logging.info(f"search review information for \"{url}\".")
        title: str | None = None
        content: str | None = None
        replies: list[str] | None = None
        posted_time: datetime | None = None
        match self.__search_target:
            case TargetWebsite.PPT:
                meta_element_selector:Final[Selector] = '.article-metaline'
                meta_tag_selector:Final[Selector] = '.article-meta-tag'
                meta_value_selector:Final[Selector] = '.article-meta-value'
                time_format:Final[str] = '%a %b %d %H:%M:%S %Y'
                key_words: Final[list[str]] = ['標題', '時間']
                content_base_element: BeautifulSoup | None = self.__get_bs_element(url=url).select_one(
                    selector='#main-content')
                article_meta_elements = content_base_element.select(selector=meta_element_selector)
                if not article_meta_elements:
                    return None
                else:
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
            case TargetWebsite.DCARD:
                selector_base:Final[Selector] = "div#__next div[role='main']"
                selector_title:Final[Selector] = selector_base + " article h1"
                selector_time:Final[Selector] = selector_base + " article time"
                selector_content:Final[Selector] = selector_base + " article span"
                selector_reply:Final[Selector] = selector_base + " section div[data-key^='comment'] span:not([class])"
                time_format: Final[str] ='%Y 年 %m 月 %d 日 %H:%M'

                self.__browser.home()
                self.__browser.get(url=url)

                title = self.__browser.find_element(selector=selector_title).text
                posted_time = datetime.strptime(
                    self.__browser.find_element(selector=selector_time).text,
                    time_format)
                content = self.__browser.find_element(selector=selector_content).text
                # selenium.common.exceptions.InvalidSelectorException
                scroll_height: int = int(self.__browser.find_element(selector="body").get_attribute("scrollHeight"))
                replies = list()
                for current_height in range(0, scroll_height, 150):
                    self.__browser.execute_script(f"window.scrollTo(0,{current_height})")
                    #　selenium.common.exceptions.StaleElementReferenceException
                    try:
                        repliess = [reply_element.text for reply_element in
                                   self.__browser.find_elements(selector=selector_reply)]
                    except (StaleElementReferenceException,InvalidSelectorException):
                        a=1
                        raise
                    replies.extend(repliess)

        return PublicReview(url=url, title=title, content=content, date=posted_time.date(), reply_count=len(replies))

    def __get_reviews(self, search_key: str) -> list[PublicReview]:
        match self.__search_target:
            case TargetWebsite.PPT:
                logging.info(f"start search reviews with search key \"{search_key}.")
                urls: list[str] = [url for url in self.__get_review_urls(search_key=search_key)]
                return list(filter(lambda x:x,[self.__get_review_information(url=url) for url in tqdm(urls)]))
            case TargetWebsite.DCARD:
                with CaptchaBrowser() as self.__browser:
                    urls: list[str] = [url for url in self.__get_review_urls(search_key=search_key)]
                    return [self.__get_review_information(url=url) for url in tqdm(urls)]
            case _:
                raise ValueError

    __delete_duplicate = lambda self, input_reviews: list(set(input_reviews))

    def search_review(self, movie_name: str) -> list[PublicReview] | None:
        search_keys: list[str] = self.get_movie_search_keys(movie_name=movie_name)
        match self.__search_target:
            case TargetWebsite.PPT | TargetWebsite.DCARD:
                reviews: list[PublicReview] = [review for search_key in search_keys for review in
                                               self.__get_reviews(search_key=search_key)]
                logging.info("trying to delete duplicate reviews.")
                reviews = self.__delete_duplicate(reviews)
                logging.info("deletion of duplicate reviews finished.")
                return reviews
            case _:
                raise ValueError

    def scrap_train_review_data(self, index_path: Path = Constants.INDEX_PATH,
                                save_folder_path: Path = Constants.PUBLIC_REVIEW_FOLDER):
        movie_data: list[MovieData] = read_index_file(file_path=index_path)
        for movie in tqdm(movie_data,bar_format = Constants.STATUS_BAR_FORMAT):
            movie.update_data(public_reviews=self.search_review(movie_name=movie.movie_name))
            movie.save_public_review(save_folder_path=save_folder_path)


TargetWebsite: TypeAlias = ReviewCollector.TargetWebsite

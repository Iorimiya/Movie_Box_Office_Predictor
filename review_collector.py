from browser import Browser
from movie_review import MovieReview, ReviewInformation

import re
import logging
import requests
from enum import Enum
from requests import Response
from typing import TypeAlias
from datetime import datetime
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup, NavigableString

Url: TypeAlias = str
RegularExpressionPattern: TypeAlias = str
Selector: TypeAlias = str


class ReviewCollector:
    class TargetWebsite(Enum):
        PPT = 1
        DCARD = 2
        IMDB = 3
        ROTTEN_TOMATO = 4

    def __init__(self, target_website: TargetWebsite):
        self.__search_target: TargetWebsite = target_website
        self.__base_url: list[Url | None] = [None, 'https://www.ptt.cc/bbs/movie/', 'https://www.dcard.tw/',
                                             'https://www.imdb.com/', 'https://www.rottentomatoes.com/']
        self.__browser: Browser | None = None
        logging.info(f"download {self.__search_target.name} data.")

    @staticmethod
    def get_movie_search_keys(movie_name: str) -> list[str]:
        output = list()

        space: RegularExpressionPattern = " "
        empty: RegularExpressionPattern = ""
        dash: RegularExpressionPattern = "-"
        double_quotation: RegularExpressionPattern = "\""
        # these elements need to be deleted first
        delete_pattern: RegularExpressionPattern = "[\(（]((數位)?\s?(修復)?\s?(IMAX|A)?\s?((日|英|國)(文|語))?\s?((2|3)D)?\s?版?)+[\)）]$"

        # original version, space version,pattern-deleted version, dash version and subtitle version
        separator_pattern: RegularExpressionPattern = "\s*[：:\-－]\s*"
        # original version and pattern-deleted version
        dual_double_quotation_pattern: RegularExpressionPattern = "\"{2}"
        start_with_double_quotation_pattern: RegularExpressionPattern = "^\""
        end_with_double_quotation_pattern: RegularExpressionPattern = "\"$"
        # space number (original version and space-deleted version)
        space_with_number_pattern: RegularExpressionPattern = " \d"

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
        return output

    def __get_search_page_url(self, search_key: str) -> str:
        match self.__search_target:
            case TargetWebsite.PPT:
                search_url_part: str = "search?q="
            case TargetWebsite.DCARD:
                search_url_part: str = "search?query="
            case TargetWebsite.IMDB:
                search_url_part: str = "find/?q="
            case TargetWebsite.ROTTEN_TOMATO:
                search_url_part: str = "search?search="
            case _:
                raise ValueError
        return f"{self.__base_url[self.__search_target.value]}{search_url_part}{search_key}"

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
                re_pattern: RegularExpressionPattern = "page=(\d+)"
                key_word: str = '最舊'
                selector: Selector = "#action-bar-container a"
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
                domain_pattern: RegularExpressionPattern = '^[^:\/]+:\/\/[^\/]+'
                base_url = re.search(pattern=domain_pattern, string=self.__base_url[self.__search_target.value]).group(
                    0)
                selector: Selector = "#main-container div.r-list-container div.title a"
                return [base_url + review['href']
                        for current_page_number in range(1, max_page_number + 1)
                        for review in
                        self.__get_bs_element(f"{search_url}&page={current_page_number}").select(selector)]
            case TargetWebsite.DCARD:
                self.__browser.get(search_url)
                scroll_height: int = int(
                    self.__browser.find_element(by=By.CSS_SELECTOR, value="body").get_attribute("scrollHeight"))
                selector: Selector = "div#__next div[role='main'] div[data-key] article[role='article'] h2 a[href]"
                urls = list()
                for current_height in range(0, scroll_height, 150):
                    self.__browser.execute_script(f"window.scrollTo(0,{current_height})")
                    new_urls = [element.get_attribute("href") for element
                                in self.__browser.find_elements(by=By.CSS_SELECTOR, value=selector)]
                    urls.extend(url for url in new_urls)
                urls = self.__delete_duplicate(urls)
                return urls

            case _:
                raise ValueError

    def __get_review_information(self, url: str) -> ReviewInformation:
        title: str | None = None
        content: str | None = None
        replies: list[str] | None = None
        posted_time: datetime | None = None
        match self.__search_target:
            case TargetWebsite.PPT:
                key_words: list[str] = ['標題', '時間']
                content_base_element: BeautifulSoup | None = self.__get_bs_element(url=url).select_one(
                    selector='#main-content')
                for article_meta_element in content_base_element.select(selector='.article-metaline'):
                    if article_meta_element.select_one(selector='.article-meta-tag').text == key_words[0]:
                        title = article_meta_element.select_one(selector='.article-meta-value').text
                    elif article_meta_element.select_one(selector='.article-meta-tag').text == key_words[1]:
                        posted_time = datetime.strptime(
                            article_meta_element.select_one(selector='.article-meta-value').text,
                            '%a %b %d %H:%M:%S %Y')

                content = ''.join(
                    [element for element in content_base_element if
                     isinstance(element, NavigableString)]).strip()
                replies = [reply.text for reply in content_base_element.select(selector='.push .push-content')]
            case TargetWebsite.DCARD:
                selector_base = "div#__next div[role='main']"
                selector_title = selector_base + " article h1"
                selector_time = selector_base + " article time"
                selector_content = selector_base + " article span"
                selector_reply = selector_base + " section div[data-key^='comment'] span:not[class]"
                self.__browser.home()
                self.__browser.get(url=url)
                title = self.__browser.find_element(by=By.CSS_SELECTOR, value=selector_title).text
                posted_time = datetime.strptime(
                    self.__browser.find_element(by=By.CSS_SELECTOR, value=selector_time).text,
                    '%Y 年 %m 月 %d 日 %H:%M')
                content = self.__browser.find_element(by=By.CSS_SELECTOR, value=selector_content).text
                replies = [reply_element.text for reply_element in
                           self.__browser.find_elements(by=By.CSS_SELECTOR, value=selector_reply)]

        return ReviewInformation(title=title, content=content, time=posted_time, replies=replies)

    def __get_reviews(self, search_key: str):
        match self.__search_target:
            case TargetWebsite.PPT:
                urls: list[str] = [url for url in self.__get_review_urls(search_key=search_key)]
                return [MovieReview.from_information(url, self.__get_review_information(url=url)) for url in urls]
            case TargetWebsite.DCARD:
                with Browser() as self.__browser:
                    urls: list[str] = [url for url in self.__get_review_urls(search_key=search_key)]
                    return [MovieReview.from_information(url, self.__get_review_information(url=url)) for url in urls]
            case _:
                raise ValueError

    __delete_duplicate = lambda self, input_reviews: list(set(input_reviews))

    def search_review(self, movie_name: str) -> list[MovieReview]:
        search_keys: list[str] = self.get_movie_search_keys(movie_name=movie_name)
        match self.__search_target:
            case TargetWebsite.PPT | TargetWebsite.DCARD:
                reviews: list[MovieReview] = [review for search_key in search_keys for review in
                                              self.__get_reviews(search_key=search_key)]
                reviews = self.__delete_duplicate(reviews)
                return reviews
            case TargetWebsite.IMDB:
                pass
            case TargetWebsite.ROTTEN_TOMATO:
                pass
            case _:
                raise ValueError

    def get_num_of_review(self,movie_name: str) -> int:
        search_keys: list[str] = self.get_movie_search_keys(movie_name=movie_name)
        with Browser() as self.__browser:
            review_urls:list[Url] = [url for search_key in search_keys for url in self.__get_review_urls(search_key=search_key)]
        review_urls = self.__delete_duplicate(review_urls)
        return len(review_urls)

TargetWebsite: TypeAlias = ReviewCollector.TargetWebsite

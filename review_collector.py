from movie_review import MovieReview

import re
import logging
import requests
from enum import Enum
from lxml import etree
from requests import Response
from typing import TypeAlias, TypedDict
from datetime import datetime
from bs4 import BeautifulSoup, NavigableString

Url: TypeAlias = str
RegularExpressionFormat: TypeAlias = str
XpathSelector: TypeAlias = str


class ReviewInformation(TypedDict):
    title: str | None
    content: str | None
    time: datetime | None
    replies: list[str] | None


class ReviewCollector:
    class Mode(Enum):
        PPT = 1
        DCARD = 2
        IMDB = 3
        ROTTEN_TOMATO = 4

    class SearchingMethod(Enum):
        CSS_SELECTOR = 1
        XPATH_SELECTOR = 2

    def __init__(self, search_mode: Mode):
        self.__download_mode: ReviewCollector.Mode = search_mode
        self.__base_url: list[str | None] = [None, 'https://www.ptt.cc/bbs/movie/', 'https://www.imdb.com/',
                                             'https://www.rottentomatoes.com/']

    def __get_search_page_url(self, search_key: str) -> str:
        if self.__download_mode == ReviewCollector.Mode.PPT:
            search_url_part: str = "search?q="
        elif self.__download_mode == ReviewCollector.Mode.IMDB:
            search_url_part: str = "find/?q="
        elif self.__download_mode == ReviewCollector.Mode.ROTTEN_TOMATO:
            search_url_part: str = "search?search="
        else:
            raise ValueError
        return f"{self.__base_url[self.__download_mode.value]}{search_url_part}{search_key}"

    def __get_bs_element(self, url: str) -> BeautifulSoup:
        # PTT在特定的板中需要over18=1這個cookies
        response: Response = requests.get(url=url, cookies={'over18': '1'}) \
            if self.__download_mode == ReviewCollector.Mode.PPT \
            else requests.get(url=url)
        response.encoding = response.apparent_encoding
        return BeautifulSoup(response.text, features='html.parser')

    def __get_xpath_element(self, url: str) -> etree.Element:
        return etree.HTML(str(self.__get_bs_element(url)))

    def __get_largest_result_page_number(self, html_root_element: etree.Element) -> int:
        if self.__download_mode == ReviewCollector.Mode.PPT:
            xpath: XpathSelector = "//*[@id='action-bar-container']//a[contains(text(),'最舊')]/@href"
            last_page_url: Url = html_root_element.xpath(xpath)[0]
            return int(re.split(pattern='=', string=re.search(pattern='page=(\d+)', string=last_page_url)[0])[1])
        else:
            raise ValueError

    def __get_review_urls(self, search_key: str) -> list[str]:
        if self.__download_mode == ReviewCollector.Mode.PPT:
            search_url: Url = self.__get_search_page_url(search_key)
            max_page_number: int = self.__get_largest_result_page_number(self.__get_xpath_element(search_url))
            url_search_xpath: XpathSelector = "//*[@id='main-container']//*[contains(@class, 'r-list-container')]//div[@class='title']/a/@href"
            domain_pattern: RegularExpressionFormat = '^[^:\/]+:\/\/[^\/]+'
            base_url = re.search(pattern=domain_pattern, string=self.__base_url[self.__download_mode.value]).group(0)

            return [base_url + review
                    for current_page_number in range(1, max_page_number + 1)
                    for review in
                    self.__get_xpath_element(f"{search_url}&page={current_page_number}").xpath(url_search_xpath)]
        else:
            raise ValueError

    @staticmethod
    def get_movie_search_keys(movie_name: str) -> list[str]:
        output = list()
        # these elements need to change to space, comma, etc.
        # separator_format = ["([\"＂／\/。]|[^\d]?[:：][^\d])+"]
        # these elements need to be deleted first
        delete_format: RegularExpressionFormat = "[\(（]((數位)?\s?(修復)?\s?(IMAX|A)?\s?((日|英|國)(文|語))?\s?((2|3)D)?\s?版?)+[\)）]$"
        if re.search(pattern=f"", string=movie_name):
            movie_name: str = re.sub(pattern=delete_format, repl='', string=movie_name)
        output.append(movie_name)

        return output

    def __get_review_information(self, url: str) -> ReviewInformation:
        title: str | None = None
        content: str | None = None
        replies: list[str] | None = None
        post_time: datetime | None = None
        if self.__download_mode == ReviewCollector.Mode.PPT:
            content_base_element: BeautifulSoup | None = self.__get_bs_element(url=url).select_one(
                selector='#main-content')
            for article_meta in content_base_element.select(selector='.article-metaline'):
                if article_meta.select_one(selector='.article-meta-tag').text == '標題':
                    title = article_meta.select_one(selector='.article-meta-value').text
                elif article_meta.select_one(selector='.article-meta-tag').text == '時間':
                    post_time = datetime.strptime(article_meta.select_one(selector='.article-meta-value').text,
                                                  '%a %b %d %H:%M:%S %Y')

            content = ''.join(
                [element for element in content_base_element if isinstance(element, NavigableString)]).strip()
            replies = [reply.text for reply in content_base_element.select(selector='.push .push-content')]

        return ReviewInformation(title=title, content=content, time=post_time, replies=replies)

    def __get_reviews(self, search_key: str):
        urls: list[str] = [url for url in self.__get_review_urls(search_key=search_key)]
        reviews = [MovieReview(url) for url in urls]
        for review in reviews:
            review_information: ReviewInformation = self.__get_review_information(url=review.url)
            review.update_information(title=review_information['title'], content=review_information['content'],
                                      time=review_information['time'], replies=review_information['replies'])
        return reviews

    __delete_duplicate_review = lambda self, input_reviews: list(set(input_reviews))

    def search_ptt_review(self, search_keys: list[str]):
        return self.__delete_duplicate_review(
            review for search_key in search_keys for review in self.__get_reviews(search_key=search_key))

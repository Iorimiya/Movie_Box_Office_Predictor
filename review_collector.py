
from movie_review import MovieReview

import re
import logging
import requests
from enum import Enum
from lxml import etree
from typing import TypeAlias, TypedDict
from bs4 import BeautifulSoup
from datetime import datetime, date, time


class ReviewInformation(TypedDict):
    title: str | None
    content: str | None
    time: datetime | None
    replies: list[str] | None


class ReviewCollector:
    class Mode(Enum):
        PPT = 1
        IMDB = 2
        ROTTEN_TOMATO = 3

    def __init__(self, search_mode: Mode):
        self.__download_mode: ReviewCollector.Mode = search_mode
        self.__base_url = [None, 'https://www.ptt.cc/bbs/movie/', 'https://www.imdb.com/',
                           'https://www.rottentomatoes.com/']

    def __get_search_page_url(self, search_key: str) -> str:
        if self.__download_mode == ReviewCollector.Mode.PPT:
            search_url_part = "search?q="
        elif self.__download_mode == ReviewCollector.Mode.IMDB:
            search_url_part = "find/?q="
        elif self.__download_mode == ReviewCollector.Mode.ROTTEN_TOMATO:
            search_url_part = "search?search="
        else:
            raise ValueError
        return f"{self.__base_url[self.__download_mode.value]}{search_url_part}{search_key}"

    @staticmethod
    def __get_xpath_element(url) -> etree.Element:
        html_file = requests.get(url=url)
        html_file.encoding = html_file.apparent_encoding
        return etree.HTML(str(BeautifulSoup(html_file.text, features='html.parser')))

    def __get_largest_result_page_number(self, html_root_element: etree.Element) -> int:
        if self.__download_mode == ReviewCollector.Mode.PPT:
            xpath = "//*[@id='action-bar-container']//a[contains(text(),'最舊')]/@href"
            last_page_url = html_root_element.xpath(xpath)[0]
            return int(re.split(pattern='=', string=re.search(pattern='page=(\d+)', string=last_page_url)[0])[1])
        else:
            raise ValueError

    def __get_review_urls(self, search_key: str) -> list[str]:
        if self.__download_mode == ReviewCollector.Mode.PPT:
            search_url: str = self.__get_search_page_url(search_key)
            max_page_number: int = self.__get_largest_result_page_number(self.__get_xpath_element(search_url))
            url_search_xpath = "//*[@id='main-container']//*[contains(@class, 'r-list-container')]//div[@class='title']/a/@href"
            domain_pattern = '^[^:\/]+:\/\/[^\/]+'
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
        # these elements need to change to space, comma, etc
        # separator_format = ["([\"＂／\/。]|[^\d]?[:：][^\d])+"]
        # these elements need to be deleted first
        delete_format = "[\(（]((數位)?\s?(修復)?\s?(IMAX|A)?\s?((日|英|國)(文|語))?\s?((2|3)D)?\s?版?)+[\)）]$"
        if re.search(pattern=f"", string=movie_name):
            movie_name = re.sub(pattern=delete_format, repl='', string=movie_name)
        output.append(movie_name)

        return output

    def __get_review_informaion(self, url: str) -> ReviewInformation:
        pass
        return ReviewInformation(title=None, content=None, time=None, replies=None)

    def __get_reviews(self, search_key: str):
        urls: list[str] = [url for url in self.__get_review_urls(search_key=search_key)]
        return [MovieReview(url=url,
                            title=self.__get_review_informaion(url=url)['title'],
                            content=self.__get_review_informaion(url=url)['content'],
                            time=self.__get_review_informaion(url=url)['time'],
                            replies=self.__get_review_informaion(url=url)['replies']) for url in urls]

    @staticmethod
    def __delete_duplicate_review(input_reviews: list[MovieReview]) -> list[MovieReview]:
        return list(set(input_reviews))

    def search_ptt_review(self, search_keys: list[str]):
        movie_reviews: list[MovieReview] = [review for search_key in search_keys
                                            for review in self.__get_reviews(search_key=search_key)]
        return self.__delete_duplicate_review(movie_reviews)

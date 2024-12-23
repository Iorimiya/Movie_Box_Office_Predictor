from enum import Enum
from movie_review import MovieReview

import re
import csv
import yaml
import logging
import requests
from lxml import etree
from pathlib import Path
from itertools import chain
from typing import TypeAlias, TypedDict
from bs4 import BeautifulSoup
from datetime import datetime, date, time


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

    def get_review_urls(self, search_key: str) -> list[str]:
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
    def __get_movie_search_keys(movie_name: str) -> list[str]:
        output = list()
        # these elements need to change to space, comma, etc
        separator_format = ["([\"＂／\/。]|[^\d]?[:：][^\d])+"]
        # these elements need to be deleted first
        delete_format = "[\(（]((數位)?\s?(修復)?\s?(IMAX|A)?\s?((日|英|國)(文|語))?\s?((2|3)D)?\s?版?)+[\)）]$"
        if re.search(pattern=f"", string=movie_name):
            movie_name = re.sub(pattern=delete_format, repl='', string=movie_name)
        output.append(movie_name)

        return output

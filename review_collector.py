from enum import Enum

class ReviewCollector:
    class Mode(Enum):
        PPT = 1
        IMDB = 2
        ROTTEN_TOMATO = 3
    def __init__(self, search_mode: Mode):
        self.__download_mode: ReviewCollector.Mode = search_mode
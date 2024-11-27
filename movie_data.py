from typing import TypeAlias
from datetime import date
BOX_OFFICE: TypeAlias = list[int]


class MovieData:
    def __init__(self, movie_name: str, box_office:BOX_OFFICE, init_date: date):
        self.movie_name = movie_name
        self.box_office = box_office
        self.init_date = init_date

from typing import TypeAlias, TypedDict
from datetime import date


class WeeklyBoxOfficeData(TypedDict):
    start_date: date
    end_date: date
    box_office: int


BoxOfficeData: TypeAlias = list[WeeklyBoxOfficeData]


class MovieData:
    def __init__(self, movie_name: str, box_office: BoxOfficeData):
        self.movie_name: str = movie_name
        self.box_offices: BoxOfficeData = box_office
        self.init_date: date = self.box_offices[0]['start_date']
        self.week_lens: int = len(self.box_offices)

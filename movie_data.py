from typing import TypeAlias, TypedDict
from datetime import date


class WeeklyBoxOfficeData(TypedDict):
    start_date: date
    end_date: date
    box_office: int


BoxOfficeData: TypeAlias = list[WeeklyBoxOfficeData]


class MovieData:
    def __init__(self, movie_id: int, movie_name: str, box_office: BoxOfficeData | None = None) -> None:
        self.movie_name: str = movie_name
        self.movie_id: int = movie_id
        self.box_offices: BoxOfficeData | None = None
        self.init_date: date | None = None
        self.week_lens: int = 0
        if box_office:
            self.update_box_office_data(box_office)

    def update_box_office_data(self, box_offices: BoxOfficeData) -> None:
        self.box_offices: BoxOfficeData | None = box_offices
        self.init_date: date | None = self.box_offices[0]['start_date']
        self.week_lens: int = len(self.box_offices)
        return

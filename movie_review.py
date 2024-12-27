from typing import TypeAlias, TypedDict
from datetime import datetime

Replies: TypeAlias = list[str]


class ReviewInformation(TypedDict):
    title: str | None
    content: str | None
    time: datetime | None
    replies: list[str] | None


class MovieReview:
    def __init__(self,
                 url: str,
                 title: str | None = None,
                 content: str | None = None,
                 time: datetime | None = None,
                 replies: Replies | None = None) -> None:
        self.url: str = url
        self.title: str | None = None
        self.content: str | None = None
        self.time: datetime | None = None
        self.replies: Replies | None = None
        if url or title or content or time or replies:
            self.update_information(title=title, content=content, time=time, replies=replies)

    @classmethod
    def from_information(cls, url: str, movie_information: ReviewInformation):
        return MovieReview(url=url, title=movie_information['title'], content=movie_information['content'],
                           time=movie_information['time'], replies=movie_information['replies'])

    def __key(self):
        return self.url

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, MovieReview):
            return self.__key() == other.__key()
        return NotImplemented

    def __str__(self):
        return self.url.__str__()

    def __repr__(self):
        return self.url.__repr__()

    def update_information(self,
                           title: str | None = None,
                           content: str | None = None,
                           time: datetime | None = None,
                           replies: Replies | None = None) -> None:
        self.title = title if title else None
        self.content = content if content else None
        self.time = time if time else None
        self.replies = replies if replies else None

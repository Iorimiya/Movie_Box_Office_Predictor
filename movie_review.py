from typing import TypeAlias
from datetime import datetime

Replies: TypeAlias = list[str]


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

    def __eq__(self, other):
        return self.url == other.url

    def update_information(self,
                           title: str | None = None,
                           content: str | None = None,
                           time: datetime | None = None,
                           replies: Replies | None = None) -> None:
        self.title = title if title else None
        self.content = content if content else None
        self.time = time if time else None
        self.replies = replies if replies else None

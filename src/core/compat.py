import sys

if sys.version_info >= (3, 11):
    # noinspection PyUnresolvedReferences
    from typing import TypedDict
else:
    # noinspection PyUnresolvedReferences
    from typing_extensions import TypedDict

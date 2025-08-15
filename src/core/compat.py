import sys

if sys.version_info >= (3, 11):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

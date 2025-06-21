from typing import Hashable, Iterable, TypeVar

T = TypeVar('T', bound=Hashable)


def delete_duplicate(items: Iterable[T]) -> list[T]:
    """
    Removes duplicate items from an iterable, preserving the order of first appearance.

    :param items: The input iterable (e.g., a list) from which to remove duplicates.
                  Items must be hashable.
    :returns: A new list containing unique items from the input iterable,
              with the order of first appearance preserved.
    """
    seen: set[T] = set()
    ordered_unique_items: list[T] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered_unique_items.append(item)
    return ordered_unique_items

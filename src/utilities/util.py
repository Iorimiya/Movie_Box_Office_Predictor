from pathlib import Path
from typing import Hashable, Iterable, TypeVar

T = TypeVar('T', bound=Hashable)


def recreate_folder(path: Path) -> None:
    """Deletes a folder (and its contents) or a file if it exists, then recreates the folder.

    If the path points to a file, the file is deleted, and a folder with the same
    name (at the same location) is created. If the path points to a directory, it is recursively deleted
    and then recreated as an empty folder.

    :param path: The path to the folder to be recreated.
    :raises Exception: For potential I/O or permission errors during deletion or creation.
    :returns: None
    """
    if path.exists():
        rmtree(path=path)  # Calls the rmtree function defined below
    path.mkdir(parents=True, exist_ok=True)
    return


def rmtree(path: Path) -> None:
    """Recursively removes a file or a directory and its contents.

    If the path points to a file or a symbolic link, it is unlinked.
    If it points to a directory, all its contents are removed recursively,
    and then the directory itself is removed.
    If the path does not exist, this function does nothing and does not raise an error.

    :param path: The path to the file or directory to be removed.
    :raises OSError: For permission errors or other OS-level issues during deletion if the path exists
                     and is a file/directory that cannot be removed.
    :returns: None
    """
    if not path.exists():
        return

    if path.is_file() or path.is_symlink():
        path.unlink()
    elif path.is_dir():
        for child in path.iterdir():
            rmtree(child)
        path.rmdir()
    # else: path exists but is not a regular file, symlink, or directory.
    # This case is currently not explicitly handled (e.g. it won't be deleted).
    return


def delete_duplicate(items: Iterable[T]) -> list[T]:
    """
    Removes duplicate items from an iterable, preserving the order of first appearance.

    :param items: The input iterable (e.g., a list) from which to remove duplicates.
                  Items must be hashable.
    :type items: Iterable[T]
    :returns: A new list containing unique items from the input iterable,
              with the order of first appearance preserved.
    :rtype: list[T]
    """
    seen: set[T] = set()
    ordered_unique_items: list[T] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered_unique_items.append(item)
    return ordered_unique_items


def check_path_exists(path_obj: Path | None) -> bool:
    """Checks if a given object is a ``pathlib.Path`` instance and if the path exists on the filesystem.

    :param path_obj: The object to check. Can be None.
    :returns: ``True`` if ``path_obj`` is a ``pathlib.Path`` object and it exists, ``False`` otherwise (including if path_obj is None).
    """
    return isinstance(path_obj, Path) and path_obj.exists()

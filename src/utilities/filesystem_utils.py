from pathlib import Path
from shutil import rmtree
from typing import Optional


def recreate_folder(path: Path) -> None:
    """
    Ensures a path exists as an empty directory.

    If the path does not exist, it is created. If it exists and is a directory,
    it is recursively deleted and then recreated.

    :param path: The path to the folder to be recreated.
    :raises NotADirectoryError: If the path exists but points to a file.
    :raises OSError: For potential I/O errors (e.g., permission denied)
                     during deletion or creation.
    """
    try:
        if path.exists():
            if not path.is_dir():
                raise NotADirectoryError(f"Path exists but is not a directory: {path}")
            _remove_path(path=path)
        path.mkdir(parents=True, exist_ok=True)
    except (NotADirectoryError, OSError) as e:
        raise type(e)(f"Failed to recreate folder at {path}: {e}") from e



def _remove_path(path: Path) -> None:
    """
    Recursively removes a file or a directory if it exists.

    If the path points to a file or a symbolic link, it is unlinked.
    If it points to a directory, it is recursively removed.
    If the path does not exist, this function does nothing.

    :param path: The path to the file or directory to be removed.
    :raises OSError: For permission errors or other OS-level issues during deletion.
    """
    if not path.exists():
        return
    try:
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            rmtree(path=path)
    except OSError as e:
        raise OSError(f"Failed to remove path {path}: {e}") from e


def is_existing_path(path_obj: Optional[Path]) -> bool:
    """
    Safely checks if an object is a Path instance and if the path exists.

    This is a convenience method that combines a type check with an existence
    check, preventing `AttributeError` on `None` or other invalid types.
    The `is_` prefix is a common convention for functions returning a boolean.

    :param path_obj: The object to check. Can be None.
    :returns: ``True`` if ``path_obj`` is a ``Path`` object and it exists,
              ``False`` otherwise.
    """
    return isinstance(path_obj, Path) and path_obj.exists()

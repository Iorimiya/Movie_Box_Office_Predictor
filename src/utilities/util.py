from pathlib import Path
from shutil import rmtree
from typing import Optional


class FilesystemUtils:
    """
    Utility class for common filesystem operations.
    """

    @staticmethod
    def recreate_folder(path: Path) -> None:
        """
        Deletes a path (file or folder) if it exists, then recreates it as an empty folder.

        If the path points to a file, the file is deleted, and a folder with the same name (at the same location) is
        created. If the path points to a directory, it is recursively deleted and then recreated as an empty folder.

        :param path: The path to the folder to be recreated.
        :raises OSError: For potential I/O or permission errors during deletion or creation.
        """
        if path.exists():
            FilesystemUtils.remove_path(path=path)
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def remove_path(path: Path) -> None:
        """
        Recursively removes a file or a directory and its contents.

        If the path points to a file or a symbolic link, it is unlinked.
        If it points to a directory, all its contents are removed recursively,
        and then the directory itself is removed.
        If the path does not exist, this function does nothing and does not raise an error.

        This method uses `shutil.rmtree` for directories and `Path.unlink` for files,
        providing a robust and standard way to remove paths.

        :param path: The path to the file or directory to be removed.
        :raises OSError: For permission errors or other OS-level issues during deletion if the path exists
                         and is a file/directory that cannot be removed.
        """
        if not path.exists():
            return

        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            rmtree(path)
        # else: path exists but is not a regular file, symlink, or directory.
        # This case is currently not explicitly handled (e.g. it won't be deleted).

    @staticmethod
    def check_path_exists(path_obj: Optional[Path]) -> bool:
        """
        Checks if a given object is a ``pathlib.Path`` instance and if the path exists on the filesystem.

        :param path_obj: The object to check. Can be None.
        :returns: ``True`` if ``path_obj`` is a ``pathlib.Path`` object, and it exists, ``False`` otherwise (including if path_obj is None).
        """
        return isinstance(path_obj, Path) and path_obj.exists()

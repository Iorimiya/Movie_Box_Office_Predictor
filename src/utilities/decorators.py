from functools import wraps
from typing import Any, Type, TypeVar

T = TypeVar('T')


def frozen_after_init(cls: Type[T]) -> Type[T]:
    """
    Decorates a class to enforce immutability after initialization.

    Wraps the ``__init__`` method to automatically call ``_lock()`` upon completion.
    The decorated class must implement ``_lock()`` and ``__setattr__`` to handle
    the locking logic.

    :param cls: The class to be decorated.
    :return: The decorated class with the modified ``__init__``.
    """
    original_init = cls.__init__

    @wraps(original_init)
    def new_init(self, *args: Any, **kwargs: Any):
        """
        Initializes the instance and locks it.

        Calls the original ``__init__`` and then triggers the ``_lock`` method.

        :param self: The instance being initialized.
        :param args: Positional arguments passed to the original ``__init__``.
        :param kwargs: Keyword arguments passed to the original ``__init__``.
        :raises TypeError: If the instance does not have a callable ``_lock`` method.
        """
        original_init(self, *args, **kwargs)

        if hasattr(self, '_lock') and callable(getattr(self, '_lock')):
            getattr(self, '_lock')()
        else:
            raise TypeError(f"Class {cls.__name__} decorated with @frozen_after_init must implement a _lock() method.")

    cls.__init__ = new_init

    return cls

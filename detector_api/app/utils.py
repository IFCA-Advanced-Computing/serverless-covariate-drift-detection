"""Utility functions for the app."""


class SingletonMeta(type):
    """Singleton metaclass."""

    _instances = {}

    def __call__(cls, *args, **kwargs):  # noqa: ANN204, ANN101, ANN002, ANN003
        """Return existing instance if exists, otherwise create a new one."""
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

import logging
import os
import pathlib

import yaml

cache_directory = pathlib.Path.home() / ".core_cache"


def set_global_loggers_to_warning():
    """
    Sets the level of all global loggers to Warning.
    """
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(logging.WARNING)

    FORMAT = "[(%(asctime)s):%(filename)s:%(lineno)s:%(funcName)s()] %(message)s"
    logging.basicConfig(
        filename="example.log",
        level=logging.DEBUG,
        format=FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def log_decorator(func, *args, **kwargs):
    def wrapper_with_logs(*args, **kwargs):
        logger = logging.getLogger(func.__name__)
        logger.debug(f"Starting {func.__name__}")
        return_val = func(*args, **kwargs)
        logger.debug(f"Finished {func.__name__}")
        return return_val

    return wrapper_with_logs


class CoreCacheManager:
    cache_directory = pathlib.Path.home() / ".core_cache"

    def __init__(self):
        if not self.cache_directory.exists():
            self.create_core_cache_directory()

    def create_core_cache_directory(self):
        """
        Create a cache file in the user's .core_cache directory.

        This function retrieves the current user's username, constructs the path to
        the .core_cache directory, creates the directory if it doesn't exist,
        and then creates a cache file inside the directory with some content.

        Returns:
            str: Path to the created cache file.
        """
        self.cache_directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_value_from_config(key):
        return ConfigCacheReader().get_value_from_config(key)

    @staticmethod
    def add_key_value_to_config(key, value):
        ConfigCacheWriter().add_key_value_to_config(key, value)


class ConfigCacheWriter:
    config_path = cache_directory / "local_config.yml"

    def add_key_value_to_config(self, key, value):
        existing_data = self._read_yaml()

        data = {key: value}
        existing_data.update(data)

        self._dump_yaml(existing_data)

    def _dump_yaml(self, existing_data):
        with open(self.config_path, "w") as yaml_file:
            yaml.dump(existing_data, yaml_file, default_flow_style=False)

    def _read_yaml(self):
        with open(self.config_path, "r") as yaml_file:
            existing_data = yaml.safe_load(yaml_file)
        return existing_data


class ConfigCacheReader:
    config_path = cache_directory / "local_config.yml"

    def _read_yaml(self):
        with open(self.config_path, "r") as yaml_file:
            existing_data = yaml.safe_load(yaml_file)
        return existing_data

    def get_value_from_config(self, key):
        existing_data = self._read_yaml()
        return existing_data.get(key)

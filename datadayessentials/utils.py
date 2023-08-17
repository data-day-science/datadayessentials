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

    def clean_core_cache(self):
        """
        Removes any files in the core cache folder that are older than 60 days.
        """

        cached_files = list(filter(self._get_cache_file_names, os.listdir(self.core_cache_path)))
        files_flagged_old = list(filter(self._is_old_file, cached_files))
        list(map(self._remove_file, files_flagged_old))

    def _get_cache_file_names(self, path: str) -> list:
        """
        Returns True if the specified path is a file in the cache directory, False otherwise.

        Args:
            path (str): The path to check.

        Returns:
            bool: True if the specified path is a file in the cache directory, False otherwise.
        """
        file_path = os.path.join(self.core_cache_path, path)
        return os.path.isfile(file_path)

    def _is_old_file(self, file_name: str) -> bool:
        """
        Returns True if the specified file is older than 60 days, False otherwise.

        Args:
            file_name (str): The name of the file to check.

        Returns:
            bool: True if the specified file is older than 60 days, False otherwise.
        """
        file_path = os.path.join(self.core_cache_path, file_name)
        return (time.time() - os.stat(file_path).st_mtime) // (24 * 3600) >= 60

    def _remove_file(self, file_name: str):
        """
        Removes the specified file.

        Args:
            file_name (str): The name of the file to remove.

        Raises:
            OSError: If the specified file does not exist or is not accessible.
        """
        file_path = os.path.join(self.core_cache_path, file_name)
        os.remove(file_path)

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


class ConfigCacheWriter:
    config_path = cache_directory / "config.yaml"

    def add_key_value_to_config(self, key, value):
        existing_data = self._read_yaml()

        data = {key: value}
        existing_data.update(data)

        self._dump_yaml(existing_data)

    def _dump_yaml(self, existing_data):
        with open(self.config_path, "r") as yaml_file:
            yaml.dump(existing_data, yaml_file, default_flow_style=False)

    def _read_yaml(self):
        with open(self.config_path, "r") as yaml_file:
            existing_data = yaml.safe_load(yaml_file)
        return existing_data


class ConfigCacheReader:
    config_path = cache_directory / "config.yaml"

    def _read_yaml(self):
        with open(self.config_path, "r") as yaml_file:
            existing_data = yaml.safe_load(yaml_file)
        return existing_data

    def get_value_from_config(self, key):
        existing_data = self._read_yaml()
        return existing_data[key]

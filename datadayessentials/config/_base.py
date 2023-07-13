from abc import ABC, abstractmethod
import yaml
from typing import Optional, List


class IConfig(ABC):
    def load_from_path(self, path) -> dict:
        with open(path, "r") as ymlfile:
            return yaml.safe_load(ymlfile)

    @classmethod
    def get_value_from_config(cls, keys: List[str]) -> Optional[str]:
        """Get a specific value from the config file.
        Example:
            >>> config = Config()
            >>> config.get_config_value(["storage_account"])
        """
        instance = cls()
        config = instance.read()
        for key in keys:
            config = config[key]
        return config

    @abstractmethod
    def read(self) -> dict:
        pass


class IGlobalConfig(IConfig):
    pass


class ILocalConfig(IConfig):
    @abstractmethod
    def create_local_config(self):
        pass

    @abstractmethod
    def write(self, config: dict):
        pass


class IConfigManager:
    @abstractmethod
    def add_database(self, server, database, credentials_name):
        pass

    @abstractmethod
    def add_database_login(
        self, credentials_name: str, username_key: str, password_key: str
    ):
        pass


class IConfigManager:
    @abstractmethod
    def __init__(self):
        pass

    def pull_config(self, version):
        pass

    def replace_local_config(self):
        pass

    def register_new_config(self):
        pass

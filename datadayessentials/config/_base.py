from abc import ABC, abstractmethod
import yaml
from typing import Optional, List

from azure.identity import (
    EnvironmentCredential,
    InteractiveBrowserCredential,
    ChainedTokenCredential,
)
import datadayessentials.utils


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

class IAuthentication(ABC):
    """Abstract base class for all authentication classes"""

    @staticmethod
    def get_azure_credentials():
        """

        Returns:
            ChainedTokenCredential: Credential chain for authenticating with azure that looks for environment credentials first and then tries to use the browser to authenticate.
        """
        environment_credentials = EnvironmentCredential()

        try:
            tenant_id = datadayessentials.utils.ConfigCacheReader().get_value_from_config("tenant_id")
        except KeyError:
            print(f"'tenant_id' does not exist in the core_cache config. Please set tenant_id using"
                    " datadayessentials.utils.ConfigCacheWriter().add_key_value_to_config(key = 'tenant_id',"
                    " value = 'your_tenant_id')")

        interactive_credentials = InteractiveBrowserCredential(
            tenant_id=tenant_id
        )
        credential_chain = ChainedTokenCredential(
            environment_credentials, interactive_credentials
        )
        return credential_chain



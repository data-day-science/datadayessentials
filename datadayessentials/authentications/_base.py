from abc import ABC, abstractmethod
import logging
from azure.identity import (
    EnvironmentCredential,
    InteractiveBrowserCredential,
    ChainedTokenCredential,
)
import datadayessentials.utils
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class AzureAuthenticationSingleton(object):
    """Creates a single azure credential chain so that authentication only needs to happen once. Singleton design
    pattern."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print("Creating authenticator")
            cls._instance = super(AzureAuthenticationSingleton, cls).__new__(cls)
            environment_credentials = EnvironmentCredential()

            try:
                tenant_id = datadayessentials.utils.ConfigCacheReader().get_value_from_config("tenant_id")
            except KeyError:
                print(f"'tenant_id' does not exist in the core_cache config. Please set tenant_id using"
                      " datadayessentials.utils.ConfigCacheWriter().add_key_value_to_config(key = 'tenant_id',"
                      " value = '736f9f09-0fa9-4930-86b0-bc4e9631f407')")

            interactive_credentials = InteractiveBrowserCredential(
                tenant_id=tenant_id
            )
            cls._instance.credential_chain = ChainedTokenCredential(
                environment_credentials, interactive_credentials
            )

        return cls._instance


class IAuthentication(ABC):
    """Abstract base class for all authentication classes"""

    @staticmethod
    def get_azure_credentials():
        """Retrieves the single instance of AzureAuthenticationSingleton

        Returns:
            ChainedTokenCredential: Credential chain for authenticating with azure that looks for environment credentials first and then tries to use the browser to authenticate.
        """
        return AzureAuthenticationSingleton().credential_chain

    @abstractmethod
    def get_credentials(self) -> dict:
        pass


class ISQLServerConnection:
    @abstractmethod
    def connect(self):
        """
        Connect to database/cloud provider
        """
        pass

    @abstractmethod
    def run_sql(self):
        """
        Run SQL with the connection above
        """
        pass

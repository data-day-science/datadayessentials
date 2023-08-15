from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, List
import os
import yaml
from pathlib import Path
from azure.identity import DefaultAzureCredential, EnvironmentCredential
import logging
from azure.identity._internal import interactive
from azure.keyvault.secrets import SecretClient
from azure.identity import (
    EnvironmentCredential,
    InteractiveBrowserCredential,
    ChainedTokenCredential,
)
from azure.core.exceptions import ClientAuthenticationError

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class AzureAuthenticationSingleton(object):
    """Creates a single azure credential chain so that authentication only needs to happen once. Singleton design pattern."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print("Creating authenticator")
            cls._instance = super(AzureAuthenticationSingleton, cls).__new__(cls)
            environment_credentials = EnvironmentCredential()
            interactive_credentials = InteractiveBrowserCredential(
                tenant_id=LocalConfig.get_environment()["tenant_id"]
            )
            cls._instance.credential_chain = ChainedTokenCredential(
                environment_credentials, interactive_credentials
            )

        return cls._instance


class IAuthentication(ABC):
    """Abstract base class for all authentication classes"""

    def get_azure_credentials(self):
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

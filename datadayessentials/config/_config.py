import dataclasses
from typing import Union
from azure.appconfiguration import AzureAppConfigurationClient
import os
from ._execution_environment_manager import ExecutionEnvironmentManager, ExecutionEnvironment
from ._base import IAuthentication
from ..utils import CoreCacheManager


class AzureConfigAuthentication(IAuthentication):
    def get_credentials(self):
        """Retrieves azure credentials, for using cloud resources. This object is needed by many other parts of core that rely on cloud services.

        Returns:
            __type__: Azure credential chain (ethods for authenticating login)
        """
        credentials = super().get_azure_credentials()
        return credentials


class AzureConfigManager:

    def __init__(self):
        pass

    def get_config_variable(self, key: str):
        return self.get_config_variable_from_cloud(key)

    def get_config_variable_from_local(self, key: str) -> Union[str, None]:
        raise NotImplementedError("Local config not implemented yet")

    def get_config_variable_from_cloud(self, key: str):
        execution_env = ExecutionEnvironmentManager.get_execution_environment()

        if execution_env == ExecutionEnvironment.PROD:
            label = execution_env.value
            client = self.get_client_from_connection_string()
        elif execution_env == ExecutionEnvironment.DEV:
            label = execution_env.value
            client = self.get_client_from_connection_string()
        elif execution_env == ExecutionEnvironment.LOCAL:
            label = 'dev'  # limitation of enum class, means that ExecutionEnvironment.LOCAL.value need to be set to dev
            if not CoreCacheManager.get_value_from_config("base_url") \
                    or not CoreCacheManager.get_value_from_config("tenant_id"):
                msg = """
                    To configure the core settings, use the 'initialise_core_config' function.
                    Example usage:
                      from config._config_setup import ConfigSetup
                      tenant_id = 'your_tenant_id'
                      base_url = 'your_base_url'
                      ConfigSetup.initialise_core_config(tenant_id, base_url)"
                """
                raise ValueError(msg)

            label = 'dev'
            client = self.get_client_via_authenticator()
        else:
            raise ValueError(f"Environment {execution_env} not recognised")

        variable_value = client.get_configuration_setting(key=key, label=label)

        return variable_value.value

    @staticmethod
    def get_client_via_authenticator():
        client = AzureAppConfigurationClient(
            base_url=CoreCacheManager.get_value_from_config("tenant_id"),
            credential=AzureConfigAuthentication().get_credentials())
        return client

    @staticmethod
    def get_client_from_connection_string():
        client = AzureAppConfigurationClient.from_connection_string(
            connection_string=os.getenv("AZURE_APP_CONFIG_CONNECTION_STRING"))
        return client


@dataclasses.dataclass
class AzureAppConfigValues:
    __dataclass_fields__ = None
    client_id: str = ""
    client_secret: str = ""
    data_lake: str = ""
    key_vault: str = ""
    machine_learning_workspace: str = ""
    project_dataset_container: str = ""
    resource_group: str = ""
    subscription_id: str = ""
    tenant_id: str = ""


class Config:
    """
    This class facilitates access to environment variables essential for machine learning productionization. If an attempt is made
    to access an environment variable that has not been configured in the local environment, this class retrieves the value
    from the provided cloud provider.

    """

    def __init__(self):
        """
        Initializes the CloudProviderConfig instance.
        """
        self.azure_config_manager = AzureConfigManager()

    def get_environment_variable(self, variable_name: str) -> str:
        """
        Retrieves the value of an environment variable.

        Args:
            variable_name (str): The name of the environment variable to retrieve.
   
        Returns:
            str: The value of the requested environment variable.

        Raises:
            ValueError:If the specified environment variable is not found in local or cloud configuration after re-getting
             variables.
        """

        if os.getenv(variable_name):
            return os.getenv(variable_name)
        else:
            variable_value = self.azure_config_manager.get_config_variable(variable_name)
            os.environ[variable_name] = variable_value
            return variable_value

    def set_default_variables(self, list_of_variables: list = AzureAppConfigValues.__dataclass_fields__.keys()):
        list(map(self.get_environment_variable, list_of_variables))

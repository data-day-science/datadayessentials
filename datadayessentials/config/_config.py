import dataclasses
import enum
import platform
from typing import Union

from azure.core.exceptions import ServiceRequestError
from azure.identity import DefaultAzureCredential
from datascience_core.authentications import DataLakeAuthentication
from azure.appconfiguration import AzureAppConfigurationClient
import os


def do_something():
    """
    # TODO: remove once replacement implemented
    Placeholder function for demonstration purposes.
    """
    pass


class ExecutionEnvironment(enum.Enum):
    DEV = "development"
    PROD = "production"
    LOCAL = "local"


@dataclasses.dataclass
class AzureAppConfigValues:
    client_id: str = ""
    client_secret: str = ""
    data_lake: str = ""
    key_vault: str = ""
    machine_learning_workspace: str = ""
    project_dataset_container: str = ""
    resource_group: str = ""
    subscription_id: str = ""
    tenant_id: str = ""


class TEMP_AZURE_CONFIG:
    def __init__(self):
        self.azure_app_config_connection_client = None

    def create_azure_app_configuration_client_connection(self, environment: str, connection_string: str):
        """
        Creates a connection to Azure App Configuration.

        Args:
            connection_string (str): The connection string for the Azure App Configuration.

        Raises:
            ConnectionError: If an error occurs while connecting to Azure.
            :param connection_string:
            :param environment:
        """
        if environment == "production" or environment == "development":
            try:
                credential = DefaultAzureCredential()
                credential = DataLakeAuthentication()
                azure_app_config_connection_client = (
                    AzureAppConfigurationClient.from_connection_string(connection_string)
                )

                # Verify connection by getting a configuration settings.Will throw error if the connection was not
                # successful.
                self.azure_app_config_connection_client.list_configuration_settings().next()
                return azure_app_config_connection_client

            except Exception as e:
                raise ServiceRequestError("An error occurred while connecting to Azure.") from e

        #manual authentication



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
        self.azure_app_values = None
        self.azure_app_config_connection_client = None
        
        self.config_values = AzureAppConfigValues()
        
        self.environment = self.set_environment()
        if self.environment.value == "production" or self.environment.value == "development":
            self.set_environment_variables()
    
    @staticmethod
    def set_environment() -> ExecutionEnvironment:
        """
        Checks whether the environment is set to production.

        Returns:
            bool: True if the environment is production, False otherwise.
        """

        if platform.system() == 'Windows':
            return ExecutionEnvironment.LOCAL
        return ExecutionEnvironment.PROD if os.getenv("AZURE_ENVIRONMENT_NAME") == "prod" else ExecutionEnvironment.DEV

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
            self.set_environment_variables()
            if os.getenv(variable_name):
                return os.getenv(variable_name)
            else:
                raise ValueError(
                    f"Environment variable '{variable_name}' not found.\n"
                    f"Make sure the variable is properly set in your cloud configuration."
                )

    def get_client_connection_from_authentication_object (self):
       pass

    def populate_config_dataclass_from_app_client_connection(self):
        """
        Retrieves values from Azure App Configuration and sets them as environment variables.
        """
        items = self.azure_app_config_connection_client.list_configuration_settings()
        [self.parse_connection_item(item.key, item.label, item.value) for item in items]

    def parse_connection_item(self, key, label, value):
        """
        Populates config variable based on Azure configuration.
        Will only set environment variables if the environment type matches the configuration label.

        Args:
            key (str): The configuration key.
            label (str): The label associated with the configuration.
            value (str): The value of the configuration.

        Note:
            This method populates config dataclass based on the configuration label,considering the environment type.
        """
        if (label == "prod" and self.environment.value == "production") or (
                label == "dev" and self.environment.value != "production"):

            self.config_values.__setattr__(key, value)

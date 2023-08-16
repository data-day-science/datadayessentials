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


class Config:
    """
    This class facilitates access to environment variables essential for machine learning productionization. If an attempt is made
    to access an environment variable that has not been configured in the local environment, this class retrieves the value
    from the provided cloud provider.

    """

    def __init__(self, use_local_config: bool = False):
        """
        Initializes the CloudProviderConfig instance.
        """
        self.azure_client_app_client = None
        self.base_url = None
        self.environment = self.get_environment()
        self.use_local_config = use_local_config

    def set_base_url(self, base_url: str):
        self.base_url = base_url

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
        if self.use_local_config:
            # search for yaml file in base directory
            pass

        if os.getenv(variable_name):
            return os.getenv(variable_name)
        else:
            # authenticate to azure

            # set values to dataclass
            # set environment variables
            # return value.
            azure_client = self.create_azure_app_configuration_client_connection(self.base_url)
            configuration_settings = self.parse_and_save_client_items_to_azure_dataclass(azure_client)
            self.set_environment_variables(configuration_settings)

            if os.getenv(variable_name):
                return os.getenv(variable_name)
            else:
                raise ValueError(
                    f"Environment variable '{variable_name}' not found.\n"
                    f"Make sure the variable is properly set in your cloud configuration."
                )

    @staticmethod
    def create_azure_app_configuration_client_connection(base_url: str = None):

        azure_app_config_client_connection = AzureAppConfigurationClient(
            base_url=base_url,
            credential=DataLakeAuthentication().get_credentials())

        try:
            azure_app_config_client_connection.list_configuration_settings().next()
            return azure_app_config_client_connection

        except Exception as e:
            raise ServiceRequestError("Unable to create connection to AzureAppConfigurationClient with current "
                                      "base_url attribute") from e

    @staticmethod
    def get_environment() -> ExecutionEnvironment:
        """
        Checks whether the environment is set to production.

        Returns:
            bool: True if the environment is production, False otherwise.
        """

        if platform.system() == 'Windows':
            return ExecutionEnvironment.LOCAL
        return ExecutionEnvironment.PROD if os.getenv("AZURE_ENVIRONMENT_NAME") == "prod" else ExecutionEnvironment.DEV

    @staticmethod
    def parse_and_save_client_items_to_azure_dataclass(client):
        azure_values = AzureAppConfigValues()
        local_environment = Config.get_environment()

        for item in client.list_configuration_settings():
            config_key = item.key
            environment_flag = item.label
            config_value = item.value

            if environment_flag == local_environment:
                setattr(azure_values, config_key, config_value)

        return azure_values

    @staticmethod
    def set_environment_variables(azure_dataclass):
        for key, value in vars(azure_dataclass).items():
            os.environ[key.upper()] = value

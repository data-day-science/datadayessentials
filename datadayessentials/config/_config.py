import platform
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


class CloudProviderConfig:
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

    @staticmethod
    def in_local_environment():
        """
        Determines if the code is running in a local environment.

        Returns:
            bool: True if running locally, False otherwise.
        """
        return platform.system() == 'Windows'

    def in_production_environment(self):
        """
        Checks whether the environment is set to production.

        Returns:
            bool: True if the environment is production, False otherwise.
        """

        if self.in_local_environment():
            return False
        # TODO: Implement checking of local environment variable in GitHub actions
        env_variable_value = os.getenv("ENV_VARIABLE_NAME")
        return env_variable_value == "production"

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

    def set_environment_variables(self):
        """
        Sets environment variables based on the cloud configuration.
        This method fetches values from Azure App Configuration and sets environment variables accordingly.
        If in production, It will fetch values from Azure App Configuration and set environment variables accordingly.
        If in development, it will fetch values from Azure App Configuration using manual authentication
            and set environment variables accordingly.
        """
        if self.in_production_environment():
            # TODO: Implement getting connection string in GitHub actions
            self.create_azure_app_configuration_client_connection("placeholder")
            self.set_environment_variables_from_cloud_config()

        if not self.in_production_environment():
            self.client_azure_connection()
            self.set_environment_variables_from_cloud_config()

    def create_azure_app_configuration_client_connection(self, connection_string: str):
        """
        Creates a connection to Azure App Configuration.

        Args:
            connection_string (str): The connection string for the Azure App Configuration.

        Raises:
            ConnectionError: If an error occurs while connecting to Azure.
        """
        try:
            credential = DefaultAzureCredential()
            credential = DataLakeAuthentication()
            self.azure_app_config_connection_client = (
                AzureAppConfigurationClient.from_connection_string(connection_string)
            )
        except Exception as e:
            raise ConnectionError("An error occurred while connecting to Azure.") from e

    def client_azure_connection(self):
        """
        Manually connects to Azure.

        This method demonstrates a manual connection to Azure services.
        It is a placeholder and should be replaced with actual implementation.
        """
        try:
            # TODO: Implement function of manual authentication to azure
            do_something()
        except Exception as e:
            raise ConnectionError("An error occurred while connecting to Azure via manual authentication.") from e

    def set_environment_variables_from_cloud_config(self):
        """
        Retrieves values from Azure App Configuration and sets them as environment variables.
        """
        items = self.azure_app_config_connection_client.list_configuration_settings()
        [self.set_environment_variable_as_per_environment(item.key, item.label, item.value) for item in items]

    def set_environment_variable_as_per_environment(self, key, label, value):
        """
        Sets environment variables based on Azure configuration.
        Will only set environment variables if the environment type matches the configuration label.

        Args:
            key (str): The configuration key.
            label (str): The label associated with the configuration.
            value (str): The value of the configuration.

        Note:
            This method sets environment variables based on the configuration label,considering the environment type.

        Example:
        set_environment_variable_as_per_environment("data_lake","prod","lake247")

        This would be saved it as AZURE_APP_DATA_LAKE_PROD=lake247 if the environment is production.
        This would not be saved if the environment is development.


        """
        if (label == "prod" and self.in_production_environment) or (
                label == "dev" and not self.in_production_environment):
            env_variable_name = f"AZURE_APP_{key.upper()}_{label.upper()}"
            os.environ[env_variable_name] = value

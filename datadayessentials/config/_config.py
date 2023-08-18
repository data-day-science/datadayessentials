import dataclasses
from datadayessentials.authentications import DataLakeAuthentication
from azure.appconfiguration import AzureAppConfigurationClient
import os
from datadayessentials.config._execution_environment_manager import ExecutionEnvironmentManager, ExecutionEnvironment


class AzureConfigManager:
    """
    Manages configuration variables retrieval from local or cloud sources.
    """

    def __init__(self, use_local_config: bool = False, base_url: str = None):
        """
        Initializes an instance of AzureConfigManager.

        Args:
            use_local_config (bool, optional): If True; use local_core_cache to find variables.Default to False.
            base_url (str, optional): Base URL for cloud configuration.Defaults to None.
        """
        self.use_local_config = use_local_config
        self.base_url = base_url

    def get_base_url(self) -> str:
        """
        Returns the base URL for cloud configuration required for getting a config client while in local environment.

        Returns:
            str: Base URL.
        """
        return self.base_url

    def get_config_variable(self, key: str) -> str:
        """
        Retrieves a configuration variable based on the key.
        If use_local_config is True, retrieve the variable from the local source.
        If use_local_config is False, retrieve the variable from the cloud source.

        Args:
            key (str): Key of the configuration variable.

        Returns: Value of the configuration variable or None if not found.
        """
        if self.use_local_config:
            return self._get_config_variable_from_local(key)
        else:
            return self._get_config_variable_from_cloud(key)

    def _get_config_variable_from_local(self, key: str) -> str:
        """
        Retrieves a configuration variable from the local source.

        Args:
            key (str): Key of the configuration variable.

        Returns: Value of the configuration variable or None if not found.
        """
        raise NotImplementedError("Local config not implemented yet")

    def _get_config_variable_from_cloud(self, key: str) -> str:
        """
        Retrieves a configuration variable from the cloud source.

        Args:
            key (str): Key of the configuration variable.

        Returns:
            str: Value of the configuration variable.
        """
        execution_env = ExecutionEnvironmentManager.get_execution_environment()

        if execution_env == ExecutionEnvironment.PROD:
            client = self._get_client_from_connection_string()
        elif execution_env == ExecutionEnvironment.DEV:
            client = self._get_client_from_connection_string()
        elif execution_env == ExecutionEnvironment.LOCAL:
            client = self._get_client_via_authenticator()
        else:
            raise ValueError(f"Environment {execution_env} not recognized")

        variable_value = client.get_configuration_setting(key=key, label=execution_env.value)

        return variable_value.next().value

    def _get_client_via_authenticator(self) -> AzureAppConfigurationClient:
        """
        Retrieves a client for AzureAppConfiguration based on the authenticator.

        Returns:
            AzureAppConfigurationClient: Azure App Configuration client.
        """
        client = AzureAppConfigurationClient(
            base_url=self.get_base_url(),
            credential=DataLakeAuthentication().get_credentials())
        return client

    @staticmethod
    def _get_client_from_connection_string() -> AzureAppConfigurationClient:
        """
        Retrieves a client for AzureAppConfiguration from a connection string.

        Returns:
            AzureAppConfigurationClient: Azure App Configuration client.
        """
        client = AzureAppConfigurationClient.from_connection_string(
            connection_string=os.getenv("AZURE_APP_CONFIG_CONNECTION_STRING"))
        return client


@dataclasses.dataclass
class AzureAppConfigValues:
    """
    Dataclass representing values for Azure App Configuration.
    """
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
    Facilitates access to environment variables essential for machine learning productionisation.
    If an attempt is made to access an environment variable that has not been configured in the local environment,
    this class retrieves the value from the provided cloud provider.
    """

    def __init__(self, use_local_config: bool = False, base_url: str = None):
        """
        Initializes a Config instance.

        Args:
            use_local_config (bool, optional): Use local configuration.Defaults to False.
            base_url (str, optional): Base URL for cloud configuration.Defaults to None.
        """
        self.azure_config_manager = AzureConfigManager(use_local_config=use_local_config, base_url=base_url)

    def get_environment_variable(self, variable_name: str) -> str:
        """
        Retrieves the value of an environment variable.

        Args:
            variable_name (str): Name of the environment variable to retrieve.

        Returns:
            str: Value of the requested environment variable.

        Raises: ValueError: If the specified environment variable is not found in local or cloud configuration after
        re-getting variables.
        """
        if os.getenv(variable_name):
            return os.getenv(variable_name)
        else:
            variable_value = self.azure_config_manager.get_config_variable(variable_name)
            os.environ[variable_name] = variable_value
            return variable_value

    def set_default_variables(self, list_of_variables: list = AzureAppConfigValues.__dataclass_fields__.keys()):
        """
        Sets default values for a list of variables.

        Args:
            list_of_variables (list, optional): List of variable names.Defaults to keys of AzureAppConfigValues
             dataclass.
        """
        list(map(self.get_environment_variable, list_of_variables))

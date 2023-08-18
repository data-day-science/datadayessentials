import dataclasses
from typing import Union
from datadayessentials.authentications import DataLakeAuthentication
from azure.appconfiguration import AzureAppConfigurationClient
import os
from datadayessentials.config._execution_environment_manager import ExecutionEnvironmentManager, ExecutionEnvironment


class AzureConfigManager:

    def __init__(self, use_local_config: bool = False, base_url: str = ""):
        self.use_local_config = use_local_config
        self.base_url = base_url

    def get_base_url(self):
        return self.base_url

    def get_config_variable(self, key: str):
        if self.use_local_config:
            return self.get_config_variable_from_local(key)
        else:
            return self.get_config_variable_from_cloud(key)

    def get_config_variable_from_local(self, key: str) -> Union[str, None]:
        raise NotImplementedError("Local config not implemented yet")

    def get_config_variable_from_cloud(self, key: str):
        execution_env = ExecutionEnvironmentManager.get_execution_environment()

        if execution_env == ExecutionEnvironment.PROD or execution_env == ExecutionEnvironment.DEV:
            client = AzureAppConfigurationClient.from_connection_string(
                connection_string=os.getenv("AZURE_APP_CONFIG_CONNECTION_STRING"))

        elif execution_env == ExecutionEnvironment.LOCAL:
            client = AzureAppConfigurationClient(
                base_url=self.get_base_url(),
                credential=DataLakeAuthentication().get_credentials())
        else:
            raise ValueError(f"Environment {execution_env} not recognised")

        variable_value = client.get_configuration_setting(key=key, label=execution_env.value)

        return variable_value.next().value



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

    def __init__(self, use_local_config: bool = False, base_url: str = ""):
        """
        Initializes the CloudProviderConfig instance.
        """
        self.azure_config_manager = AzureConfigManager(use_local_config=use_local_config, base_url=base_url)

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

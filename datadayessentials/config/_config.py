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
        self.execution_environment = ExecutionEnvironmentManager.get_execution_environment()
        if self.execution_environment == ExecutionEnvironment.PROD:
            self.label = self.execution_environment.value
            self.client = self.get_client_from_connection_string()
        elif self.execution_environment == ExecutionEnvironment.DEV:
            self.label = self.execution_environment.value
            self.client = self.get_client_from_connection_string()
        elif self.execution_environment == ExecutionEnvironment.LOCAL:
            self.label = 'dev'
            self.client = self.get_client_via_authenticator()
        else:
            raise ValueError(f"Environment {self.execution_environment} not recognised")

    def get_config_variable(self, key: str):
        return self.get_config_variable_from_cloud(key)

    def get_config_variable_from_local(self, key: str) -> Union[str, None]:
        raise NotImplementedError("Local config not implemented yet")

    def get_config_variable_from_cloud(self, key: str):
        variable_value = self.client.get_configuration_setting(key=key, label=self.label)
        return variable_value.value

    def get_client_via_authenticator(self):
        client = AzureAppConfigurationClient(base_url=CoreCacheManager.get_value_from_config("base_url"),
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
    client_id: str = "",
    client_secret: str = "",
    data_lake: str = "",
    key_vault: str = "",
    machine_learning_workspace: str = "",
    project_dataset_container: str = "",
    resource_group: str = "",
    subscription_id: str = "",
    tenant_id: str = "",


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
        if self.check_environent_available():
            return
        self.execution_env = ExecutionEnvironmentManager.get_execution_environment()
        self.validate_local_config()
        if self.execution_env == ExecutionEnvironment.LOCAL:
            os.environ["AZURE_TENANT_ID"] = CoreCacheManager.get_value_from_config("tenant_id")  
            os.environ["BASE_URL"] = CoreCacheManager.get_value_from_config("base_url")  
        self.azure_config_manager = AzureConfigManager()
        self.set_default_variables()

    def validate_local_config(self):
        msg = """
            No means of downloading the config from Azure App Configuration found. Please include AZURE_APP_CONFIG_CONNECTION_STRING for use in a remote server or initialise datadayessentials using the initialise_core_config fuction as below:

                from datadayessentials import initialise_core_config
                tenant_id = 'your_tenant_id'
                base_url = 'your_base_url for an Azure App Configuration service'
                initialise_core_config(tenant_id, base_url)
        """
        available_environment_variables = os.environ.keys()
        if self.execution_env == ExecutionEnvironment.LOCAL:
            tenant_id = CoreCacheManager.get_value_from_config("tenant_id")  
            base_url = CoreCacheManager.get_value_from_config("base_url")
            if (tenant_id is None) or (base_url is None):
                raise EnvironmentError(msg)
        elif self.execution_env in [ExecutionEnvironment.DEV, ExecutionEnvironment.PROD, ExecutionEnvironment.STAGING]:
            if "AZURE_APP_CONFIG_CONNECTION_STRING" not in os.environ.keys():
                raise EnvironmentError(
                    "'AZURE_APP_CONFIG_CONNECTION_STRING' environment variable not set for remote access to Azure Application Configuration"
                )

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
        env_variable_name = "AZURE_" + variable_name.upper()

        if os.getenv(env_variable_name):
            return os.getenv(env_variable_name)
        else:

            variable_value = self.azure_config_manager.get_config_variable(variable_name)
            os.environ[env_variable_name] = variable_value
            return variable_value
        
    def set_default_variables(self, list_of_variables: list = AzureAppConfigValues.__dataclass_fields__.keys()):
        list(map(self.get_environment_variable, list_of_variables))
        

    # def set_default_variables(self, variables : AzureAppConfigValues)-> AzureAppConfigValues:
    #     if not self.check_environent_available():


    #         fields = variables.__annotations__
    #         updated_fields = {field_name: self.get_environment_variable(getattr(variables, field_name)) for field_name in fields.keys()}
    #         return AzureAppConfigValues(**updated_fields)

    def check_environent_available(self):
        for variable in AzureAppConfigValues.__dataclass_fields__.keys():
            if variable not in os.environ.keys():
                return False
        return True

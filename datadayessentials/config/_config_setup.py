from dataclasses import dataclass

from datadayessentials.config._config import GlobalConfig, LocalConfig
from pathlib import Path
from typing import Dict, List, Union, Optional, Any
import os
import logging

from dataclasses import dataclass

class CloudConfig:
    def __init__(self):
        self.required_attributes = None

    def __post_init__(self):
        for attr_name in self.required_attributes:
            attr_value = getattr(self, attr_name)
            if not attr_value:
                raise ValueError(f"{attr_name.replace('_', ' ').capitalize()} is required, please update"
                                 f" your input dictionary")

@dataclass
class AzureConfig(CloudConfig):
    client_id: str
    subscription_id: str
    tenant_id: str
    resource_group_name: str
    app_configuration: str
    required_attributes = ['client_id', 'subscription_id', 'tenant_id', 'resource_group_name']

@dataclass
class GCPConfig(CloudConfig):
    project_id: str
    account_key: str
    required_attributes = ['project_id', 'account_key']

@dataclass
class AWSConfig(CloudConfig):
    access_key_id: str
    secret_access_key: str
    region: str
    required_attributes = ['access_key_id', 'secret_access_key', 'region']


class ConfigAlreadyExistsError(Exception):
    pass


class ConfigSetup:
    @staticmethod
    def set_up_global_config() -> None:
        """
        Creates a global config file if one does not already exist.

        Returns:
            None
        """
        global_cfg = GlobalConfig()
        global_cfg.ensure_local_cache_exists()

    @staticmethod
    def create_local_config_if_not_exists() -> None:
        """
        Creates a global config file if one does not already exist.

        Returns:
            None
        """
        LocalConfig()
        
    @staticmethod
    def initialise_core_config(
        environment_name: str,
        subscription_id: str,
        tenant_id: str,
        resource_group: str,
        machine_learning_workspace: str,
        data_lake: str,
        create_new_config: Optional[bool] = False,
        project_dataset_container: str = "projects",
    ):
        """
        Initialise the core local config. This provides the minimum config required to either load an existing config file that your team has registered in a data lake or create a new config file for your team to use.
        """
        
        LocalConfig().create_local_config()

        team_env_settings = {
            "environment_name": environment_name,
            "subscription_id": subscription_id,
            "tenant_id": tenant_id,
            "resource_group": resource_group,
            "machine_learning_workspace": machine_learning_workspace,
            "data_lake": data_lake,
            "project_dataset_container": project_dataset_container
        }

    @staticmethod
    def load_config(cloud_provider, config_data):
        """
        Load a config object based on the cloud provider
        Pass in the cloud provider and the config data to loa. The config data should be a dictionary of the required attributes for the cloud provider

        Params:
            cloud_provider: str
                The cloud provider to load the config for
            config_data: dict
                The config data to load

        Returns:
            config: CloudConfig
                The config object

        Raises:
            ValueError: If the cloud provider is not recognised
            ValueError: If the config data does not contain the required attributes

        Example:
            >>> config_data = {
                    "keyvault_name": "my_keyvault",
                    "resource_group_name": "my_resource_group",
                    "subscription_id": "my_subscription_id"
                }
            >>> config = ConfigSetup.load_config("azure", config_data)

        """
        if cloud_provider == "azure":
            return AzureConfig(**config_data)
        elif cloud_provider == "gcp":
            return GCPConfig(**config_data)
        elif cloud_provider == "aws":
            return AWSConfig(**config_data)
        else:
            raise ValueError("Invalid cloud provider, Please provide either 'azure', 'gcp' or 'aws'")




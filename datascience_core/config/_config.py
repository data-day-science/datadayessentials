import pandas as pd
import os
import yaml
from pathlib import Path
from ._base import ILocalConfig, IGlobalConfig


class GlobalConfig(IGlobalConfig):
    """Class to manage the global configuration file. Which cannot be edited by the user, and is for storing any settings that any user should be able to access.
    Example:
        >>> global_config = GlobalConfig()
        >>> global_config.read()
    """

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "global_config.yml")

    def read(self):
        global_config = self.load_from_path(self.path)
        return global_config


class LocalConfig(ILocalConfig):
    """Class to manage the local configuration file. THe local configuration file contains any user specific azure resources, including resource groups and storage accounts.
    Example:
        >>> local_config = LocalConfig()
        >>> local_config.read()
    """

    ENVIRONMENT = os.environ.get("ENVIRONMENT", "dev")
    DEFAULT_CONFIG = {"sync_with_remote": False}

    def __init__(self):
        """
        Initialise the local config, creating the cache directory and config file
        if they do not exist. Using the default config.
        """
        self.global_config = GlobalConfig().read()
        print(self.global_config)
        self.folder = os.path.join(
            str(Path.home()), self.global_config["local_cache_dir"]
        )
        self.filename = "local_config.yml"
        self.path = os.path.join(self.folder, self.filename)

        path_exists = os.path.exists(self.path)
        if not path_exists:
            self.create_local_config()

    def catch_key_error(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except KeyError as e:
                raise KeyError(
                    f"Keys {args} not found in local config. See full error:\n{e}"
                )

        return wrapper

    def set_environment(self, environment: str):
        valid_environments = self.get_value_from_config(["env"]).keys()
        if environment not in valid_environments:
            raise ValueError(f"Environment {environment} not found in local config")
        LocalConfig.ENVIRONMENT = environment

    def read(self) -> dict:
        local_config = self.load_from_path(self.path)
        return local_config

    def create_local_config(self):
        os.makedirs(self.folder, exist_ok=True)
        self.write(self.DEFAULT_CONFIG)

    def write(self, config: dict):
        with open(self.path, "w") as ymlfile:
            yaml.dump(config, ymlfile, default_flow_style=False)

    @staticmethod
    @catch_key_error
    def get_key_vault():
        # checked
        key_vault_ref = LocalConfig.get_value_from_config(
            ["azure", "environments", LocalConfig.ENVIRONMENT, "key_vault"]
        )
        return LocalConfig.get_value_from_config(["azure", "key_vaults", key_vault_ref])

    @staticmethod
    @catch_key_error
    def get_storage_account():
        storage_account_ref = LocalConfig.get_value_from_config(
            ["azure", "environments", LocalConfig.ENVIRONMENT, "storage_account"]
        )
        return LocalConfig.get_value_from_config(
            ["azure", "storage_accounts", storage_account_ref]
        )

    @staticmethod
    @catch_key_error
    def get_data_lake():
        data_lake_ref = LocalConfig.get_value_from_config(
            ["azure", "environments", LocalConfig.ENVIRONMENT, "data_lake"]
        )
        return LocalConfig.get_value_from_config(
            ["azure", "data_lake_named_folders", LocalConfig.ENVIRONMENT, data_lake_ref]
        )

    @staticmethod
    @catch_key_error
    def get_environment():
        return LocalConfig.get_value_from_config(
            ["azure", "environments", LocalConfig.ENVIRONMENT]
        )

    @staticmethod
    @catch_key_error
    def get_environment_from_name(environment_name: str):
        return LocalConfig.get_value_from_config(
            ["azure", "environments", environment_name]
        )

    @staticmethod
    @catch_key_error
    def get_machine_learning_workspace():
        machine_learning_workspace_ref = LocalConfig.get_value_from_config(
            [
                "azure",
                "environments",
                LocalConfig.ENVIRONMENT,
                "machine_learning_workspace",
            ]
        )
        return LocalConfig.get_value_from_config(
            ["azure", "machine_learning_workspaces", machine_learning_workspace_ref]
        )

    @staticmethod
    @catch_key_error
    def get_database_credentials(credentials_reference: str) -> dict:
        key_vault = LocalConfig.get_environment()["key_vault"]
        return LocalConfig.get_value_from_config(
            [
                "azure",
                "key_vaults",
                key_vault,
                "database_credentials",
                credentials_reference,
            ]
        )

    @staticmethod
    @catch_key_error
    def get_database(database_reference: str) -> dict:
        return LocalConfig.get_value_from_config(["databases", database_reference])

    @staticmethod
    @catch_key_error
    def list_available_databases() -> list:
        return list(LocalConfig.get_value_from_config(["databases"]).keys())

    @staticmethod
    @catch_key_error
    def get_local_cache_dir() -> str:
        return os.path.join(Path.home(), LocalConfig().global_config["local_cache_dir"])

    @staticmethod
    @catch_key_error
    def get_data_lake_folder(
        named_folder: str, data_lake: str = None, use_current_environment: bool = True
    ) -> str:
        """Retrieve a folder inside the data lake for the current environment, if use_current_environment is False then search across all envorinments for this named folder. The reason for this additional argument is because there are two use cases:

        1. A folder that is environment specific (dev and prod have different locations for data storage)
        2. A folder that is consistent across all environments

        Args:
            named_folder (str): Folder reference in the LocalConfig
            data_lake (str, optional): Uses the environment default data lake unless this argument is passed
        Returns:
            str: Container, path and datalake for the named folder inside the local config
        """
        if use_current_environment:
            return LocalConfig.get_current_environment_data_lake_folder(
                named_folder, data_lake
            )
        else:
            return LocalConfig.get_any_environment_data_lake_folder(named_folder)

    @staticmethod
    def get_any_environment_data_lake_folder(named_folder: str):
        # If we dont care about environment, search across all environments and datalakes for the folder details
        for env in LocalConfig.get_value_from_config(
            ["azure", "data_lake_named_folders"]
        ):
            for data_lake in LocalConfig.get_value_from_config(
                ["azure", "data_lake_named_folders", env]
            ):
                for key, value in LocalConfig.get_value_from_config(
                    ["azure", "data_lake_named_folders", env, data_lake]
                ).items():
                    if named_folder == str(key):
                        named_folder_details = value
                        return {**named_folder_details, **{"data_lake": data_lake}}
        raise KeyError(f"Named folder {named_folder} not found in local config")

    @staticmethod
    def get_current_environment_data_lake_folder(
        named_folder: str, data_lake: str = None
    ):
        env = LocalConfig.get_environment()
        data_lake = data_lake if data_lake else env["data_lake"]
        named_folder_details = LocalConfig.get_value_from_config(
            [
                "azure",
                "data_lake_named_folders",
                LocalConfig.ENVIRONMENT,
                data_lake,
                named_folder,
            ]
        )
        return {**named_folder_details, **{"data_lake": data_lake}}

    @staticmethod
    @catch_key_error
    def get_batch_endpoint(endpoint_reference: str) -> str:
        return LocalConfig.get_value_from_config(
            ["azure", "batch_endpoints", endpoint_reference]
        )

    @staticmethod
    def get_dataset_manager_environment():
        settings = LocalConfig.get_value_from_config(
            ["azure", "project_dataset_manager"]
        )
        return LocalConfig.get_value_from_config(
            ["azure", "environments", settings["environment"]]
        )

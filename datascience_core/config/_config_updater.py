from ._base import IConfigManager
from ._config import LocalConfig, GlobalConfig
from ._config_manager import ConfigManager
from typing import Optional


class ConfigContentUpdater(IConfigManager):
    """Class to add and delete entities in the local configuration file.

    Example:
        >>> config_manager = ConfigManager()
        >>> config_manager.add_environment(
                "dev",
                subscription_id = "1234567890",
                resource_group = "my_resource_group",
                client_id = "1234567890",
                tenant_id = "1234567890"
            )
    """

    def __init__(self):
        self.local_config = LocalConfig()
        if LocalConfig.get_value_from_config(["sync_with_remote"]) is True:
            self.config_manager = ConfigManager()

    def config_update(func):
        """Decorater to pre-load the latest version of the config before making changes, then to push the updated version afterwards."""
        def wrapper(self, *args, **kwargs):
            self.pre_update_pull()
            new_config = func(self, *args, **kwargs)
            self.post_update_push(new_config)
        return wrapper
    
    def pre_update_pull(self):
        """Syncs the local config with the remote config before making any changes."""
        local_config = LocalConfig()
        if LocalConfig.get_value_from_config(["sync_with_remote"]) is True:
            # This ensures tha the config manager is initialised
            if not hasattr(self, "config_manager"):
                self.config_manager = ConfigManager()
            self.config_manager.pull_config()
    
    def post_update_push(self, new_config: dict):
        """Pushes the updated config to the remote config after making changes."""
        local_config = LocalConfig()
        local_config.write(new_config)
        if LocalConfig.get_value_from_config(["sync_with_remote"]) is True:
            if not hasattr(self, "config_manager"):
                self.config_manager = ConfigManager()
            self.config_manager.register_new_config()


    @config_update
    def add_project_dataset_manager(self, environment_name: str):
        """
        Adds a new project dataset manager to the local configuration file.

        Args:
            env (str): The environment of the project dataset manager. The environment then contains the machine learning workspace and the data lake required by the ProjectDatasetManager
        Raises:
            ValueError: If the environment does not exist in the global configuration file.
        """

        if environment_name not in LocalConfig.get_value_from_config(
            ["azure", "environments"]
        ):
            raise ValueError("Environment does not exist in global config")

        local_config = self.local_config.read()
        entry = {"environment": environment_name}
        local_config["azure"]["project_dataset_manager"] = entry
        return local_config

    def set_sync_with_remote(self, sync_with_remote: bool):
        """
        Sets the sync_with_remote flag in the local configuration file.

        Args:
            sync_with_remote (bool): The value to set the sync_with_remote flag to.
        """
        new_config = self.local_config.read()
        new_config["sync_with_remote"] = sync_with_remote
        local_config = LocalConfig()
        local_config.write(new_config)

    @config_update
    def remove_project_dataset_manager(self):
        """Removes the project dataset manager from the local configuration file."""
        local_config = self.local_config.read()
        local_config = self.remove_config_key(
            local_config, ["azure", "project_dataset_manager"]
        )
        return local_config

    @config_update
    def add_data_lake_named_folder(
        self, environment: str, data_lake: str, folder_reference, container, path
    ):
        """
        Adds a new storage location (data lake folder) to the local configuration file.

        Args:
            self: The object instance of the class.
            environment (str): The environment of the data lake folder. The environment then contains the service principal which should be used to access the datalake
            data_lake (str): The name of the data lake.
            folder_reference (str): A reference name for the folder.
            folder_name (str): The name of the folder.
            container (str): The name of the container.
            path (str): The path to the folder in the container.
        """

        local_config = self.local_config.read()
        local_config = self.create_nested_keys(local_config, ['azure', 'data_lake_named_folders', environment, data_lake])

        entry = {"data_lake": data_lake, "container": container, "path": path}
        local_config["azure"]["data_lake_named_folders"][environment][data_lake][
            folder_reference
        ] = entry
        return local_config
    
    @staticmethod
    def create_nested_keys(config, keys):
        """Creates nested keys in a dictionary.

        Args:
            config (dict): The dictionary to create the nested keys in.
            keys (list): The list of keys to create.
        """
        if len(keys) == 1:
            if keys[0] not in config:
                config[keys[0]] = {}
        elif keys[0] not in config:
            config[keys[0]] = ConfigContentUpdater.create_nested_keys({}, keys[1:])
        else:
            config[keys[0]] = ConfigContentUpdater.create_nested_keys(config[keys[0]], keys[1:])
        return config

    @config_update
    def remove_data_lake_named_folder(
        self, environment: str, data_lake: str, folder_reference: str
    ):
        """Removes a named folder from the local configuration file.
        Args:
            self: The object instance of the class.
            folder_reference (str): A reference name for the folder.
        """
        local_config = self.local_config.read()
        local_config = self.remove_config_key(
            local_config,
            [
                "azure",
                "data_lake_named_folders",
                environment,
                data_lake,
                folder_reference,
            ],
        )
        return local_config

    @config_update
    def add_database(
        self,
        database_reference: str,
        server_address: str,
        database_name: str,
        credentials_name: str,
    ):
        """
        Adds a new database to the local configuration file.

        Args:
            self: The object instance of the class.
            database_reference (str): A reference name for the database.
            server_address (str): The server address of the database.
            database_name (str): The name of the database.
            credentials_name (str): The name of the credentials to be associated with the database.
        Raises:
            ValueError: If the credentials name does not exist in the local configuration file.
        """

        credentials_found = False
        for key_vault in self.local_config.read()["azure"]["key_vaults"].keys():
            if (
                credentials_name
                in self.local_config.read()["azure"]["key_vaults"][key_vault][
                    "database_credentials"
                ].keys()
            ):
                credentials_found = True

        if not credentials_found:
            raise ValueError("Credentials name does not exist in local config")

        local_config = self.local_config.read()
        entry = {
            "server": server_address,
            "database": database_name,
            "credentials": credentials_name,
        }
        local_config = self.create_nested_keys(local_config, ['databases'])
        local_config["databases"][database_reference] = entry
        return local_config

    @config_update
    def remove_database(self, database_reference: str):
        """
        Removes a database from the local configuration file.

        Args:
            self: The object instance of the class.
            database_reference (str): The reference name of the database to be removed.
        """
        local_config = self.local_config.read()
        local_config = self.remove_config_key(
            local_config, ["databases", database_reference]
        )
        return local_config

    @config_update
    def add_database_credentials(
        self,
        credentials_name: str,
        username_key: str,
        password_key: str,
        key_vault: str,
    ):
        """
        Adds the given database connection credentials to the local configuration yaml file.

        Args:
            self: The object instance of the class.
            credentials_name (str): The name of the credentials to be added.
            username_key (str): The key for the username associated with the credentials.
            password_key (str): The key for the password associated with the credentials.
        """

        local_config = self.local_config.read()
        entry = {
            "username_key": username_key,
            "password_key": password_key,
        }

        local_config = self.create_nested_keys(local_config, ['azure', 'key_vaults', key_vault, 'database_credentials'])

        local_config["azure"]["key_vaults"][key_vault]["database_credentials"][
            credentials_name
        ] = entry
        return local_config

    @config_update
    def remove_database_credentials(self, credentials_name: str, key_vault: str):
        """
        Removes the given database connection credentials from the local configuration yaml file

        Args:
            credentials_name (str): The name of the credentials to be removed.
        """
        for database in self.local_config.read()["databases"].keys():
            if (
                credentials_name
                in self.local_config.read()["databases"][database]["credentials"]
            ):
                raise ValueError(
                    "Cannot remove credentials as they are used by a database"
                )
        # Use the remove_config_key function to remove the credentials
        local_config = self.local_config.read()
        local_config = self.remove_config_key(
            local_config,
            [
                "azure",
                "key_vaults",
                key_vault,
                "database_credentials",
                credentials_name,
            ],
        )
        return local_config

    @config_update
    def add_key_vault(
        self, key_vault_name: str, environment_name: str, dl_storage_account_key: str
    ) -> dict:
        """
        Sets the key vault name in the local configuration file.

        Args:
            self: The object instance of the class.
            key_vault_name (str): The name of the key vault.
        """
        # Check the environment exists in the local config
        if environment_name not in self.local_config.read()["azure"]["environments"]:
            raise ValueError(
                "Environment does not exist in local config, please add it first"
            )

        local_config = self.local_config.read()
        local_config = self.create_nested_keys(local_config, ['azure', 'key_vaults'])

        local_config["azure"]["key_vaults"][key_vault_name] = {
            "environment": environment_name,
            "dl_storage_account_key": dl_storage_account_key,
            "key_vault_name": key_vault_name,
            "credentials": {},
        }

        return local_config

    @config_update
    def remove_key_vault(self, key_vault_name: str) -> dict:
        """
        Removes the key vault name from the local configuration file.

        Args:
            self: The object instance of the class.
            key_vault_name (str): The name of the key vault.
        """
        for env in self.local_config.read()["azure"]["environments"]:
            if (
                key_vault_name
                in self.local_config.read()["azure"]["environments"][env]["key_vault"]
            ):
                raise ValueError(
                    "Cannot remove this key vault as it is in use by an environment, please remove the environment first"
                )

        for credential_name in self.local_config.read()["azure"]["key_vaults"][
            key_vault_name
        ]["credentials"]:
            for database in self.local_config.read()["databases"]:
                if (
                    credential_name
                    in self.local_config.read()["databases"][database]["credentials"]
                ):
                    raise ValueError(
                        "Cannot remove this key vault as it contains a credential that is in use by a database, please remove the database first"
                    )
        local_config = self.local_config.read()
        local_config = self.remove_config_key(
            local_config, ["azure", "key_vaults", key_vault_name]
        )
        return local_config

    @config_update
    def add_environment(
        self,
        environment_name: str,
        subscription_id: str,
        resource_group: str,
        tenant_id: str,
        project_dataset_container: str,
        client_id: Optional[str] = None,
        machine_learning_workspace: Optional[str] = None,
        data_lake: Optional[str] = None,
    ) -> dict:
        """
        Adds a new environment to the local configuration file.

        Args:
            self: The object instance of the class.
            environment_name (str): The name of the environment.
            subscription_id (str): The subscription id of the environment.
            resource_group (str): The resource group of the environment.
            client_id (str): The client id of the environment.
            tenant_id (str): The tenant id of the environment.
            project_dataset_container (str): The container name that will hold all project datasets
            machine_learning_workspace (str): The machine learning workspace of the environment.
            data_lake (str): The data lake of the environment.

        Returns:
            dict: The modified local configuration file.
        """

        local_config = self.local_config.read()
        entry = {
            "subscription_id": subscription_id,
            "resource_group": resource_group,
            "tenant_id": tenant_id,
            "project_dataset_container": project_dataset_container
        }
        if client_id is not None:
            entry["client_id"] = (client_id,)
        if machine_learning_workspace is not None:
            entry["machine_learning_workspace"] = machine_learning_workspace
        if data_lake is not None:
            entry["data_lake"] = data_lake

        local_config = self.create_nested_keys(local_config, ['azure', 'environments'])

        local_config["azure"]["environments"][environment_name] = entry
        return local_config

    @config_update
    def remove_environment(self, environment_name: str) -> dict:
        """
        Removes an environment from the local configuration file.

        Args:
            self: The object instance of the class.
            environment_name (str): The name of the environment.

        Returns:
            dict: The modified local configuration file.
        """

        local_config = self.local_config.read()
        local_config = self.remove_config_key(
            local_config, ["azure", "environments", environment_name]
        )
        return local_config
    
    @config_update
    def add_batch_endpoint(self, endpoint_name: str, deployment_name: str, payload_storage: str, local_save_location: str) -> dict:
        """
        Adds a new batch endpoint to the local configuration file.

        Args:
            self: The object instance of the class.
            endpoint_name (str): The name of the endpoint.
            deployment_name (str): The name of the deployment.
            payload_storage (str): The name of the payload storage.
            local_save_location (str): The local save location.

        Returns:
            dict: The modified local configuration file.
        """

        local_config = self.local_config.read()
        local_config = self.create_nested_keys(local_config, ['azure', 'batch_endpoints'])
        
        storage_found = self.check_named_data_lake_folder_exists(payload_storage)
        if storage_found == False:
            raise ValueError("The payload storage is not a named data lake folder")

        local_config["batch_endpoints"][endpoint_name] = {
            "endpoint_name": endpoint_name,
            "deployment_name": deployment_name,
            "payload_storage": payload_storage,
            "inference_results_local_save": {
                'full_path': local_save_location
            }
        }
        return local_config
    
    def check_named_data_lake_folder_exists(self, named_folder: str) -> bool:
        # Check if the payload storage is a named data lake folder
        storage_found = False
        for env in self.local_config.read()["azure"]["data_lake_named_folders"]:
            for data_lake in self.local_config.read()["azure"]["data_lake_named_folders"][env]:
                if named_folder in self.local_config.read()["azure"]["data_lake_named_folders"][env][data_lake]:
                    storage_found = True 
        return storage_found

    
    @config_update
    def remove_batch_endpoint(self, endpoint_name: str) -> dict:
        """
        Removes a batch endpoint from the local configuration file.

        Args:
            self: The object instance of the class.
            endpoint_name (str): The name of the endpoint.

        Returns:
            dict: The modified local configuration file.
        """

        local_config = self.local_config.read()
        local_config = self.remove_config_key(
            local_config, ["batch", "endpoints", endpoint_name]
        )
        return local_config

    def remove_config_key(self, nested_dict, keys_to_delete):
        """
        Delete a part of a nested dictionary.

        Args:
            keys_to_delete (list): A list of keys that represent the path to the nested part.
            keys_to_delete (list): A list of keys that represent the path to the nested part.

        Returns:
            dict: The modified dictionary with the nested part deleted.

        Example:
            >>> nested_dict = {'a': {'b': {'c': 1}}}
            >>> remove_config_key(nested_dict, ['a', 'b', 'c'])
            {'a': {'b': {}}}
        """
        current_dict = nested_dict
        for key in keys_to_delete[:-1]:
            current_dict = current_dict[key]

        del current_dict[keys_to_delete[-1]]
        return nested_dict

"""
Tests for setting and removing configuration settings stored in the LocalConfig class and file. The tests are functional rather thatn unit tests, so they add and remove config settings and check that they have been added or removed correctly. 
"""
import unittest
from .._config_updater import ConfigContentUpdater, LocalConfig
from unittest.mock import patch

# This dictionary is used to mock the local_config file so nothing is overwritten during testing.
LOCAL_CONFIG = {"sync_with_remote": False}

# Mock Functions to replace the LocalConfig class functions
def write(self, new_config):
    global LOCAL_CONFIG
    LOCAL_CONFIG = new_config

def read(self):
    return LOCAL_CONFIG

@patch.object(LocalConfig, "write", write)
@patch.object(LocalConfig, "read", read)
class TestConfigUpdater(unittest.TestCase):
    def setUp(self) -> None:
        # Reset LOCAL_CONFIG
        global LOCAL_CONFIG
        LOCAL_CONFIG = {
            "sync_with_remote": False,
            "azure": {
                "project_dataset_manager": {"env": "dev"},
                "environments": {
                    "dev": {
                        "subscription_id": "1234",
                        "resource_group": "rg1",
                        "machine_learning_workspace": "ws1",
                        "key_vault": "kv1",
                        "workspace_region": "westus2",
                        "data_lake": "lake",
                        "service_principal": {
                            "client_id": "1234",
                            "tenant_id": "1234",
                        },
                    }
                },
            },
        }

    def test_remove_config_key(self):
        self.config_updater = ConfigContentUpdater()
        keys_to_delete = ["azure", "project_dataset_manager"]
        self.config_updater.remove_config_key(LOCAL_CONFIG, keys_to_delete)
        assert "project_dataset_manager" not in LOCAL_CONFIG["azure"]

    def test_add_remove_project_dataset_manager(self):
        self.config_updater = ConfigContentUpdater()
        self.config_updater.remove_project_dataset_manager()
        assert "project_dataset_manager" not in LOCAL_CONFIG["azure"]
        self.config_updater.add_project_dataset_manager("dev")
        assert "project_dataset_manager" in LOCAL_CONFIG["azure"]

    def test_set_sync_with_remote(self):
        self.config_updater = ConfigContentUpdater()
        self.config_updater.set_sync_with_remote("test")
        assert LOCAL_CONFIG["sync_with_remote"] == "test"

    def test_add_remove_data_lake_named_folder(self):
        self.config_updater = ConfigContentUpdater()
        self.config_updater.add_data_lake_named_folder(
            "dev", "test_data_lake", "folder_name", "container_name", "path"
        )
        assert (
            "folder_name"
            in LOCAL_CONFIG["azure"]["data_lake_named_folders"]["dev"]["test_data_lake"]
        )
        self.config_updater.remove_data_lake_named_folder(
            "dev", "test_data_lake", "folder_name"
        )
        assert (
            "folder_name"
            not in LOCAL_CONFIG["azure"]["data_lake_named_folders"]["dev"][
                "test_data_lake"
            ]
        )

    def test_add_remove_database_and_credentials(self):
        self.config_updater = ConfigContentUpdater()
        self.config_updater.add_key_vault("test_key_vault", "dev", "test_account_key")
        self.config_updater.add_database_credentials(
            "credentials_name", "username_key", "password_key", "test_key_vault"
        )
        assert (
            "credentials_name"
            in LOCAL_CONFIG["azure"]["key_vaults"]["test_key_vault"][
                "database_credentials"
            ]
        )
        self.config_updater.add_database(
            "database_reference", "server_address", "database_name", "credentials_name"
        )
        assert "database_reference" in LOCAL_CONFIG["databases"]

        self.config_updater.remove_database("database_reference")
        assert "database_reference" not in LOCAL_CONFIG["databases"]

        self.config_updater.remove_database_credentials(
            "credentials_name", "test_key_vault"
        )
        assert (
            "credentials_name"
            not in LOCAL_CONFIG["azure"]["key_vaults"]["test_key_vault"][
                "database_credentials"
            ]
        )

        self.config_updater.remove_key_vault("test_key_vault")
        assert "test_key_vault" not in LOCAL_CONFIG["azure"]["key_vaults"]

    def test_add_remove_project_dataset_manager(self):
        self.config_updater = ConfigContentUpdater()
        self.config_updater.remove_project_dataset_manager()
        assert "project_dataset_manager" not in LOCAL_CONFIG["azure"]
        self.config_updater.add_project_dataset_manager("dev")
        assert "project_dataset_manager" in LOCAL_CONFIG["azure"]

    def test_add_remove_environment(self):
        self.config_updater = ConfigContentUpdater()
        self.config_updater.add_environment(
            environment_name="test_add_remove_env",
            subscription_id="34567",
            resource_group="rg-test-2",
            client_id="789",
            tenant_id="123",
        )
        assert "test_add_remove_env" in LOCAL_CONFIG["azure"]["environments"]
        self.config_updater.remove_environment(environment_name="test_add_remove_env")
        assert "test_add_remove_env" not in LOCAL_CONFIG["azure"]["environments"]

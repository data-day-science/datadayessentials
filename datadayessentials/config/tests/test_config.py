import unittest
from unittest.mock import patch, Mock
import datadayessentials
from datadayessentials.config._execution_environment_manager import ExecutionEnvironment
from datadayessentials.config._config import Config, AzureConfigManager


class TestAzureConfigManager(unittest.TestCase):

    @patch('datadayessentials.config.ExecutionEnvironmentManager.get_execution_environment')
    @patch('azure.appconfiguration.AzureAppConfigurationClient.from_connection_string')
    def test_get_config_variable_from_cloud_prod(self, mock_client, mock_env):
        # Arrange
        mock_env.return_value = ExecutionEnvironment.PROD
        mock_config_client = Mock()
        mock_client.return_value = mock_config_client
        mock_setting = Mock()
        mock_setting.value = (Mock(value="config_value_1"), Mock(value="config_value_2"))
        mock_config_client.get_configuration_setting.return_value = mock_setting

        # Act
        config_manager = AzureConfigManager()
        result = config_manager.get_config_variable("config_key")

        # Assert
        mock_config_client.get_configuration_setting.assert_called_once()

    @patch('datadayessentials.config.ExecutionEnvironmentManager.get_execution_environment')
    @patch('azure.appconfiguration.AzureAppConfigurationClient.from_connection_string')
    def test_get_config_variable_from_cloud_dev(self, mock_client, mock_env):
        # Arrange
        mock_env.return_value = ExecutionEnvironment.DEV
        mock_config_client = Mock()
        mock_client.return_value = mock_config_client
        mock_setting = Mock()
        mock_setting.value = (Mock(value="config_value_1"), Mock(value="config_value_2"))
        mock_config_client.get_configuration_setting.return_value = mock_setting

        # Act
        config_manager = AzureConfigManager()
        result = config_manager.get_config_variable("config_key")

        # Assert
        mock_config_client.get_configuration_setting.assert_called_once()

    @patch('datadayessentials.config.ExecutionEnvironmentManager.get_execution_environment',
           return_value=ExecutionEnvironment.LOCAL)
    @patch.object(AzureConfigManager, 'get_client_via_authenticator', return_value=Mock())
    def test_get_config_variable_from_cloud_local(self, mock_get_client, mock_env):
        # Arrange
        datadayessentials.initialise_core_config("tenant_id", "base_url")
        mock_client = mock_get_client.return_value
        mock_configuration_setting = Mock()
        mock_configuration_setting.value = "config_value_1"
        mock_client.get_configuration_setting.return_value = mock_configuration_setting

        # Act
        config_manager = AzureConfigManager()
        variable_value = config_manager.get_config_variable("test_key")

        # Assert
        self.assertEqual("config_value_1", variable_value)
        mock_get_client.assert_called_once()
        mock_client.get_configuration_setting.assert_called_once_with(
            key="test_key", label="dev"
        )
        datadayessentials.utils.CoreCacheManager().remove_value_from_config("tenant_id")
        datadayessentials.utils.CoreCacheManager().remove_value_from_config("base_url")


class TestConfig(unittest.TestCase):

    @patch('datadayessentials.config.Config.get_environment_variable', return_value="cloud_value")
    def test_get_environment_variable_cloud_or_var(self, mock_get_config_var):
        # Arrange
        config = Config()

        # Act
        result = config.get_environment_variable("variable_name")

        # Assert
        self.assertEqual(result, "cloud_value", "Environment variable should match cloud_value")

    @patch('datadayessentials.config.Config.get_environment_variable', return_value="cloud_value")
    def test_set_default_variables(self, mock_get_env_var):
        # Arrange
        config = Config()

        # Act
        config.set_default_variables(["variable_1", "variable_2"])
        result_variable_1 = config.get_environment_variable("variable_1")
        result_variable_2 = config.get_environment_variable("variable_2")

        # Assert
        self.assertEqual(result_variable_1, "cloud_value", "variable_1 should match cloud_value")
        self.assertEqual(result_variable_2, "cloud_value", "variable_2 should match cloud_value")

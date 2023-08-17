import unittest
from unittest.mock import patch, Mock
from datadayessentials.config import ExecutionEnvironmentManager, ExecutionEnvironment

from datadayessentials.config import AzureConfigManager, Config


class TestAzureConfigManager(unittest.TestCase):

    @patch('datadayessentials.config.ExecutionEnvironmentManager.get_execution_environment')
    @patch('azure.appconfiguration.AzureAppConfigurationClient.from_connection_string')
    def test_get_config_variable_from_cloud_prod(self, mock_client, mock_env):
        # Setting the return value execution_env to ensure we go down the prod path
        mock_env.return_value = ExecutionEnvironment.PROD

        # Mocking the returned client of AzureAppConfigurationClient.from_connection_string
        mock_config_client = Mock()
        # Mocking the returned value of get_configuration_setting is a mocked client.
        mock_client.return_value = mock_config_client

        # Mocking the client.get_configuration_setting
        mock_setting = Mock()
        mock_setting.value = (Mock(value="config_value_1"), Mock(value="config_value_2"))
        mock_config_client.get_configuration_setting.return_value = mock_setting

        config_manager = AzureConfigManager()
        result = config_manager.get_config_variable("config_key")

        # Assert that get_configuration_setting was called once
        mock_config_client.get_configuration_setting.assert_called_once()

    @patch('datadayessentials.config.ExecutionEnvironmentManager.get_execution_environment')
    @patch('azure.appconfiguration.AzureAppConfigurationClient.from_connection_string')
    def test_get_config_variable_from_cloud_dev(self, mock_client, mock_env):
        # Setting the return value execution_env to ensure we go down the prod path
        mock_env.return_value = ExecutionEnvironment.DEV

        # Mocking the returned client of AzureAppConfigurationClient.from_connection_string
        mock_config_client = Mock()
        # Mocking the returned value of get_configuration_setting is a mocked client.
        mock_client.return_value = mock_config_client

        # Mocking the client.get_configuration_setting
        mock_setting = Mock()
        mock_setting.value = (Mock(value="config_value_1"), Mock(value="config_value_2"))
        mock_config_client.get_configuration_setting.return_value = mock_setting

        config_manager = AzureConfigManager()
        result = config_manager.get_config_variable("config_key")

        # Assert that get_configuration_setting was called once
        mock_config_client.get_configuration_setting.assert_called_once()

    @patch('datadayessentials.config.ExecutionEnvironmentManager.get_execution_environment')
    @patch('azure.appconfiguration.AzureAppConfigurationClient')
    def test_get_config_variable_from_cloud_local(self, mock_client, mock_env):
        # Setting the return value execution_env to ensure we go down the prod path
        mock_env.return_value = ExecutionEnvironment.LOCAL

        # Mocking the returned client of AzureAppConfigurationClient.from_connection_string
        mock_config_client = Mock()
        # Mocking the returned value of get_configuration_setting is a mocked client.
        mock_client.return_value = mock_config_client

        # Mocking the client.get_configuration_setting
        mock_setting = Mock()
        mock_setting.value = (Mock(value="config_value_1"), Mock(value="config_value_2"))
        mock_config_client.return_value = mock_setting

        config_manager = AzureConfigManager()
        result = config_manager.get_config_variable("config_key")

        # Assert that get_configuration_setting was called once
        mock_config_client.get_configuration_setting.assert_called_once()

    def test_get_config_variable_from_local(self):
        config_manager = AzureConfigManager(use_local_config=True)
        with self.assertRaises(NotImplementedError):
            config_manager.get_config_variable("config_key")

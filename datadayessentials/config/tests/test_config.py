import os
import unittest
from unittest.mock import patch, Mock, PropertyMock

from azure.appconfiguration import AzureAppConfigurationClient

from datadayessentials.authentications import DataLakeAuthentication
from datadayessentials.config._execution_environment_manager import ExecutionEnvironment
from datadayessentials.config._config import Config, AzureConfigManager


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

    def test_get_config_variable_from_local(self):
        config_manager = AzureConfigManager(use_local_config=True)
        with self.assertRaises(NotImplementedError):
            config_manager.get_config_variable("config_key")

    @patch('datadayessentials.config.ExecutionEnvironmentManager.get_execution_environment',
           return_value=ExecutionEnvironment.LOCAL)
    @patch.object(AzureConfigManager, 'get_client_via_authenticator', return_value=Mock())
    def test_get_config_variable_from_cloud_local(self, mock_get_client, mock_env):
        # Setup mocks for the returned client instance
        mock_client = mock_get_client.return_value
        mock_configuration_setting = Mock()
        mock_configuration_setting.next.return_value = Mock(value="config_value_1")
        mock_client.get_configuration_setting.return_value = mock_configuration_setting

        # Create an instance of AzureConfigManager
        config_manager = AzureConfigManager(base_url="temp")

        # Call the method
        variable_value = config_manager.get_config_variable("test_key")

        # Assert the results
        self.assertEqual(variable_value, "config_value_1")

        # Ensure the methods were called as expected
        mock_get_client.assert_called_once()
        mock_client.get_configuration_setting.assert_called_once_with(
            key="test_key", label="local"
        )


class TestConfig(unittest.TestCase):

    @patch('datadayessentials.config.AzureConfigManager.get_config_variable', return_value="cloud_value")
    def test_get_environment_variable_cloud_or_var(self, mock_get_config_var):
        config = Config()
        result = config.get_environment_variable("variable_name")

        self.assertEqual(result, "cloud_value")

    @patch('datadayessentials.config.Config.get_environment_variable', return_value="cloud_value")
    def test_set_default_variables(self, mock_get_env_var):
        config = Config()
        config.set_default_variables(["variable_1", "variable_2"])
        # assert called
        mock_get_env_var.assert_called()

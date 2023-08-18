import os
import unittest
from unittest.mock import patch, Mock, PropertyMock

from datadayessentials.config import ExecutionEnvironment
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

    # @patch('azure.appconfiguration.AzureAppConfigurationClient', autospec=True)
    # @patch('datadayessentials.config.AzureConfigManager.get_base_url', return_value="test_url")
    # @patch('datadayessentials.config.ExecutionEnvironmentManager.get_execution_environment')
    # @patch('datadayessentials.authentications.DataLakeAuthentication.get_credentials')
    # def test_get_config_variable_from_cloud_local(self, mock_data_lake_auth, mock_env, mock_get_base_url, mock_client):
    #     # Setting the return value execution_env to ensure we go down the local path
    #     mock_env.return_value = ExecutionEnvironment.LOCAL
    #
    #     # Mock the get_credentials method of DataLakeAuthentication
    #     mock_credentials = Mock()
    #     mock_data_lake_auth.return_value.get_credentials.return_value = mock_credentials
    #
    #     # Mocking the returned client of AzureAppConfigurationClient
    #     mock_config_client = Mock()
    #     # Mocking the returned value of get_configuration_setting is a mocked client.
    #     mock_client.return_value = mock_config_client
    #
    #     # Mocking the client.get_configuration_setting
    #     mock_setting = Mock()
    #     mock_setting.value = (Mock(value="config_value_1"), Mock(value="config_value_2"))
    #     mock_config_client.return_value = mock_setting
    #
    #     config_manager = AzureConfigManager()
    #     result = config_manager.get_config_variable("config_key")
    #
    #     # Assert that get_configuration_setting was called once
    #     mock_config_client.get_configuration_setting.assert_called_once()
    #
    #     # Add assertions for the constructor call if needed
    #     mock_client.assert_called_once_with(
    #         base_url='test_url',  # Assuming 'test_url' is returned by mock_get_base_url
    #         credential=mock_credentials  # Use the mocked credentials
    #     )

    def test_get_config_variable_from_local(self):
        config_manager = AzureConfigManager(use_local_config=True)
        with self.assertRaises(NotImplementedError):
            config_manager.get_config_variable("config_key")


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


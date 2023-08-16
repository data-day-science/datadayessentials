from datadayessentials.config._execution_environment_manager import ExecutionEnvironmentManager, ExecutionEnvironment
import unittest
from unittest.mock import patch


class TestExecutionEnvironmentManager(unittest.TestCase):

    @patch('platform.system', return_value="Windows")
    def test_get_execution_environment_windows_prod(self, mock_platform_system):
        result = ExecutionEnvironmentManager.get_execution_environment()
        print(result)
        self.assertEqual(result, ExecutionEnvironment.LOCAL)

    @patch('os.getenv', return_value="dev")
    @patch('platform.system', return_value="Linux")
    def test_get_execution_environment_dev(self, mock_platform_system, mock_os_getenv):
        result = ExecutionEnvironmentManager.get_execution_environment()
        self.assertEqual(result, ExecutionEnvironment.DEV)

    @patch('os.getenv', return_value="prod")
    @patch('platform.system', return_value="Linux")
    def test_get_execution_environment_dev(self, mock_platform_system, mock_os_getenv):
        result = ExecutionEnvironmentManager.get_execution_environment()
        self.assertEqual(result, ExecutionEnvironment.PROD)

    @patch('os.getenv', return_value=None)
    @patch('platform.system', return_value='Linux')
    def test_asser_error_no_environment_variable(self, mock_platform_system, mock_os_getenv):
        self.assertRaises(EnvironmentError, ExecutionEnvironmentManager.get_execution_environment)

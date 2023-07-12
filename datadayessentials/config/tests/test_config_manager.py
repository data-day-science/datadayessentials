
import unittest
from unittest.mock import patch, mock_open, MagicMock
import pytest
from datadayessentials.data_retrieval import ProjectDatasetManager
from .._config_manager import ConfigManager
import os


class TestConfigManager:
    @pytest.fixture
    def mock_local_config(self):
        local_config = MagicMock()
        local_config.read.return_value = {'key1': 'value1', 'key2': 'value2', 'sync_with_remote': False}
        return local_config

    @pytest.fixture
    def mock_project_dataset_manager(self):
        project_dataset_manager = MagicMock()
        project_dataset_manager.load_datasets.return_value = {'local_config': {'key1': 'value1', 'key2': 'value2'}}
        return project_dataset_manager

    @pytest.fixture
    def config_manager(self, mock_local_config, mock_project_dataset_manager):
        with patch('datadayessentials.config._config_manager.LocalConfig', return_value=mock_local_config):
            with patch('datadayessentials.config._config_manager.ProjectDatasetManager', return_value=mock_project_dataset_manager):
                yield ConfigManager()

    def test_pull_config_no_version(self, config_manager, mock_project_dataset_manager):
        config_manager.pull_config()
        mock_project_dataset_manager.load_datasets.assert_called_once_with()

    def test_pull_config_with_version(self, config_manager, mock_project_dataset_manager):
        config_manager.pull_config(version=1)
        mock_project_dataset_manager.load_datasets.assert_called_once_with(versions={'local_config': 1})

    def test_register_new_config(self, config_manager, mock_local_config, mock_project_dataset_manager):
        config_manager.register_new_config()
        mock_local_config.read.assert_called_once_with()
        mock_project_dataset_manager.register_dataset.assert_called_once_with(registered_dataset_name='local_config', data={'key1': 'value1', 'key2': 'value2', 'sync_with_remote': False})
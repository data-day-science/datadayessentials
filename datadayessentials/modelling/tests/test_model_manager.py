import pathlib
import shutil

import pytest
from azureml.core import Model

from datadayessentials.modelling.model_manager import ModelManager, ModelCacher
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock
from .utils import trigger_test_run
import sys

def remove_test_folder(folder_to_delete: str = None):
    folder_to_delete = pathlib.Path(folder_to_delete)
    if folder_to_delete.exists() and folder_to_delete.is_dir():
        shutil.rmtree(folder_to_delete)


class TestModelManager(unittest.TestCase):
    import unittest
    import shutil
    import pathlib

    def setUp(self):
        self.model_manager = ModelManager()
        self.run_id, self.run_instance = trigger_test_run()
        

    def test_get_model_files_from_run(self):
        # Download the files
        self.model_manager.get_model_files_from_run(self.run_id, "pickled_models")
        assert self.model_manager.workspace is not None

        # Verify that the downloaded folder exists
        downloaded_folder = pathlib.Path.cwd() / "pickled_models" / "model"
        downloaded_file = downloaded_folder / "model.cb"
        self.assertTrue(downloaded_folder.exists())
        self.assertTrue(downloaded_file.exists())

        remove_test_folder(str(downloaded_folder))

    def test_get_model_files_from_registered_model(self):
        model_to_get = "test_model"
        directory_name = "test_download_folder"

        # Download the files
        self.model_manager.get_model_files_from_registered_model(model_to_get, folder_to_save_model=directory_name)
        assert self.model_manager.workspace is not None

        # Verify that the downloaded folder exists
        print(pathlib.Path.cwd())
        downloaded_folder = pathlib.Path.cwd() / pathlib.Path(directory_name) / pathlib.Path("model")

        self.assertTrue(downloaded_folder.exists())
        #delete the test_download_folder at the end of the test
        remove_test_folder(directory_name)

    def test_register_model(self):
        self.model_manager.get_model_files_from_run(self.run_id, "pickled_models")
        model_name = "test_model_23"

        model_file = pathlib.Path.cwd() / "pickled_models" / "model" / "model.cb"
        self.model_manager.register_model_from_local_folder(model_name, str(model_file))
        assert model_name in [model for model in self.model_manager.workspace.models]

        Model(workspace=self.model_manager.workspace, name=model_name).delete()
        assert model_name not in [model for model in self.model_manager.workspace.models]


class TestModelCacher(unittest.TestCase):
    def setUp(self):
        self.model_name = "model"
        self.model_version = 1
        if sys.platform != "win32":
            home_dir = Path('/tmp/')
        else:
            home_dir = Path.home()
        self.cache_directory = home_dir / "cache"
        self.model_path = home_dir / "model"
        self.model_cache_path = self.cache_directory / f"{self.model_name}-{self.model_version}"
        self.model_cacher = ModelCacher(self.model_name, self.model_version)

    @patch("datadayessentials.modelling.model_manager.Config")
    def test_get_model_cache_path(self, mock_config):
        mock_config.return_value = MagicMock(cache_directory=self.cache_directory)
        self.assertEqual(self.model_cache_path, self.model_cacher._get_model_cache_path())

    @patch("datadayessentials.modelling.model_manager.Config")
    def test_is_model_cached(self, mock_config):
        self._clean_up()
        mock_config.return_value = MagicMock(cache_directory=self.cache_directory)
        self.assertFalse(self.model_cacher.is_model_cached())
        self.model_cache_path.mkdir()
        self.assertTrue(self.model_cacher.is_model_cached())
        self._clean_up()
    

    @patch("datadayessentials.modelling.model_manager.Config")
    def test_copy_model_to_cache(self, mock_config):
        self._clean_up()
        mock_config.return_value = MagicMock(cache_directory=self.cache_directory)
        self.model_path.mkdir()
        self.model_cacher.copy_model_folder_to_cache(self.model_path)
        self.assertTrue((self.model_cache_path).exists())
        self._clean_up()

    @patch("datadayessentials.modelling.model_manager.Config")
    def test_copy_model_from_cache(self, mock_config):
        self._clean_up()
        mock_config.return_value = MagicMock(cache_directory=self.cache_directory)
        self.model_cache_path.mkdir()
        self.model_cacher.copy_model_folder_from_cache(self.model_path)
        self.assertTrue(Path(self.model_path).exists())
        self._clean_up()

    def _clean_up(self):
        if self.model_path.exists():
            shutil.rmtree(self.model_path)
        if self.model_cache_path.exists():
            shutil.rmtree(self.model_cache_path)

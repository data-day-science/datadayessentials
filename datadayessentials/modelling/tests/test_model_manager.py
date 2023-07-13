import pathlib
import shutil

import pytest
from azureml.core import Model

from datadayessentials.modelling.model_manager import ModelManager
import unittest
from unittest.mock import patch


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

    def test_get_model_files_from_run(self):
        run_id = "1aed3f9b-0cee-430e-8676-5070ed572d27"

        # Download the files
        self.model_manager.get_model_files_from_run(run_id, "pickled_models")
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

    def test_register_model(self):
        self.model_manager.get_model_files_from_run("1aed3f9b-0cee-430e-8676-5070ed572d27", "pickled_models")
        model_name = "test_model_23"

        model_file = pathlib.Path.cwd() / "pickled_models" / "model" / "model.cb"
        self.model_manager.register_model_from_local_folder(model_name, str(model_file))
        assert model_name in [model for model in self.model_manager.workspace.models]

        Model(workspace=self.model_manager.workspace, name=model_name).delete()
        assert model_name not in [model for model in self.model_manager.workspace.models]



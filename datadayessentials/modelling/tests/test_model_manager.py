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
        run_id = "2cf2751a-e2b8-44a7-bad0-a3f2d923cb2e"

        # Download the files
        self.model_manager.get_model_files_from_run(run_id, "pickled_models")
        assert self.model_manager.workspace is not None

        # Verify that the downloaded folder exists
        downloaded_folder = pathlib.Path.cwd() / "pickled_models" / "model"
        downloaded_file = downloaded_folder / "model.pkl"
        self.assertTrue(downloaded_folder.exists())
        self.assertTrue(downloaded_file.exists())

        remove_test_folder(str(downloaded_folder))

    def test_get_model_files_from_registered_model(self):
        model_to_get = "prime-predictions-service"
        directory_name = "test_download_folder"

        # Download the files
        self.model_manager.get_model_files_from_registered_model(model_to_get, folder_to_save_model=directory_name)
        assert self.model_manager.workspace is not None

        # Verify that the downloaded folder exists
        print(pathlib.Path.cwd())
        downloaded_folder = pathlib.Path.cwd() / pathlib.Path(directory_name) / pathlib.Path("pickled_models")

        self.assertTrue(downloaded_folder.exists())
        assert "PP5_v1.cb" in [file.name for file in downloaded_folder.iterdir() if file.is_file()]

        remove_test_folder(directory_name)

    def test_register_model(self):
        self.model_manager.get_model_files_from_run("2cf2751a-e2b8-44a7-bad0-a3f2d923cb2e", "pickled_models")
        model_name = "unit_test_model"

        model_file = pathlib.Path.cwd() / "pickled_models" / "model" / "model.pkl"
        self.model_manager.register_model_from_local_folder(model_name, str(model_file))
        assert model_name in [model for model in self.model_manager.workspace.models]

        Model(workspace=self.model_manager.workspace, name="unit_test_model").delete()
        assert model_name not in [model for model in self.model_manager.workspace.models]



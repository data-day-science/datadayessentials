from ...authentications import DataLakeAuthentication
from .._project_dataset_manager import (
    ProjectDatasetManager,
    MLStudioProjectDatasetsHelper,
)
from .._save_data import BlobLocation
import pandas as pd
import logging
import time
import unittest
from unittest import mock
import pytest

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class EXAMPLE_DATA:
    def __init__(self):
        self.name = (
            ""  # "datascience_247money_scorecard/Datasets/scorecard_cra_dataset"
        )


class TestDatasetURIGenerator:
    def test_deconstruct_abfss_path(self):
        """_summary_"""

        test_path = "abfss://container@storage_account.dfs.core.windows.net/dir1/dir2/dir3/file.csv"
        blob_location = BlobLocation(
            account="storage_account",
            container="test",
            filepath="dir1/dir2/dir3",
            filename="file.csv",
        )
        actual_dict = blob_location._deconstruct_abfss_path(test_path)
        assert actual_dict["account"] == "storage_account"
        assert actual_dict["container"] == "container"
        assert actual_dict["file_path"] == "dir1/dir2/dir3"
        assert actual_dict["file_name"] == "file.csv"

    def test_deconstruct_https_path(self):
        """_summary_"""

        test_path = "https://storage_account.blob.core.windows.net/container/dir1/dir2/dir3/file.csv"
        blob_location = BlobLocation(
            account="storage_account",
            container="test",
            filepath="dir1/dir2/dir3",
            filename="file.csv",
        )
        actual_dict = blob_location._deconstruct_https_path(test_path)
        assert actual_dict["account"] == "storage_account"
        assert actual_dict["container"] == "container"
        assert actual_dict["file_path"] == "dir1/dir2/dir3"
        assert actual_dict["file_name"] == "file.csv"

    def test_get_single_uri(self):
        """_summary_"""
        auth = DataLakeAuthentication()
        dataset_manager = MLStudioProjectDatasetsHelper(auth)
        test_path = (
            "abfss://test@storage_account.dfs.core.windows.net/dir1/dir2/dir3/file.csv"
        )
        test_path = [{"path": test_path, "name": "test", "data_type": "csv"}]

        actual_blob_location = dataset_manager._convert_asset_paths_to_bloblocations(
            test_path
        )
        actual_blob_location = actual_blob_location[0]["path"]

        assert actual_blob_location.get_account() == "storage_account"
        assert actual_blob_location.get_container() == "test"
        assert actual_blob_location.get_filepath() == "dir1/dir2/dir3"
        assert actual_blob_location.get_filename() == "file.csv"
        assert (
            str(actual_blob_location)
            == "https://storage_account.blob.core.windows.net/test/dir1/dir2/dir3/file.csv"
        )

    def test_get_multiple_uris(self):
        auth = DataLakeAuthentication()
        dataset_manager = MLStudioProjectDatasetsHelper(auth)
        test_path = (
            "abfss://test@storage_account.dfs.core.windows.net/dir1/dir2/dir3/file.csv"
        )
        test_paths = [
            {"path": test_path, "name": "test", "data_type": "csv"},
            {"path": test_path, "name": "test", "data_type": "csv"},
            {"path": test_path, "name": "test", "data_type": "csv"},
        ]
        actual_blob_location = dataset_manager._convert_asset_paths_to_bloblocations(
            test_paths
        )

        assert len(actual_blob_location) == 3


class TestMLStudioProjectDatasetsHelper:
    def test_get_path_to_registered_dataset(self):
        project_assets = ["test"]
        auth = DataLakeAuthentication()
        ml_studio_helper = MLStudioProjectDatasetsHelper(credentials=auth)
        asset_paths = ml_studio_helper._get_path_to_registered_dataset(
            project_assets=project_assets, versions={}
        )
        expected = ""  # "https://ds247dldev.blob.core.windows.net/projects/test/Datasets/test/"

        assert isinstance(asset_paths, list)
        assert isinstance(asset_paths[0], dict)
        assert asset_paths[0]["name"] == "test"
        assert expected in list(asset_paths[0].values())[0]


class TestProjectDatasetManager(unittest.TestCase):
    def test_register_dataset_dataframe(self):
        data = {"col1": [0, 1, 2, 3, 4, 5], "col2": [0, 1, 2, 3, 4, 5]}
        data = pd.DataFrame(data)
        project = "test"

        dataset_manager = ProjectDatasetManager(project=project)
        dataset_manager.register_dataset(registered_dataset_name="test", data=data)

    def test_register_dataset_json(self):
        data = {"col1": [0, 1, 2, 3, 4, 5], "col2": [0, 1, 2, 3, 4, 5]}
        project = "test"

        dataset_manager = ProjectDatasetManager(project=project)
        dataset_manager.register_dataset(registered_dataset_name="test", data=data)

    def test_register_dataset_pickle(self):
        data = "Hello"
        project = "test"

        dataset_manager = ProjectDatasetManager(project=project)
        dataset_manager.register_dataset(registered_dataset_name="test", data=data)

    def test_register_and_pull_dataset(self):
        data = {"col1": [0, 1, 2, 3, 4, 5], "col2": [0, 1, 2, 3, 4, 5]}
        data = pd.DataFrame(data)
        project = "test"

        dataset_manager = ProjectDatasetManager(project=project)
        dataset_manager.register_dataset(registered_dataset_name="test", data=data)

        time.sleep(2)
        dataset = dataset_manager.load_datasets()

        pd.testing.assert_frame_equal(data, dataset["test"])

    def test_delete_dataset_works(self):
        data = {"col1": [0, 1, 2, 3, 4, 5], "col2": [0, 1, 2, 3, 4, 5]}
        data = pd.DataFrame(data)
        project = "test"

        dataset_manager = ProjectDatasetManager(project=project)

        file_system_client = dataset_manager.project_asset_loader.datalake_service.get_file_system_client(
            file_system="projects/test/Datasets"
        )
        directory_client = file_system_client.get_directory_client("test_destroy")
        assert directory_client.exists() == False

        dataset_manager.register_dataset(
            registered_dataset_name="test_destroy", data=data
        )
        time.sleep(1)
        directory_client = file_system_client.get_directory_client("test_destroy")
        assert directory_client.exists() == True

        dataset_manager.remove_dataset(dataset_name="test_destroy")
        dataset_manager.confirm_destroy(confirm_dataset_name="test_destroy")

        time.sleep(1)
        directory_client = file_system_client.get_directory_client("test_destroy")
        assert directory_client.exists() == False

    @mock.patch(
        "datadayessentials.data_retrieval._project_dataset_manager.DatalakeProjectAssetsHelper"
    )
    @mock.patch(
        "datadayessentials.data_retrieval._project_dataset_manager.DataLakePickleSaver"
    )
    def test_register_pickle(
        self,
        mock_data_lake_pickle_saver,
        mock_data_lake_project_assets_helper,
    ):
        dataset_manager = ProjectDatasetManager(project="test")
        data = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

        (
            data_type,
            save_blob,
        ) = ProjectDatasetManager._register_dataset_based_on_datatype(
            dataset_manager, data, "test_dataset", True
        )

        self.assertEqual(data_type, "pkl")
        mock_data_lake_project_assets_helper.assert_called_once()
        mock_data_lake_pickle_saver.assert_called_once()

    @mock.patch(
        "datadayessentials.data_retrieval._project_dataset_manager.DatalakeProjectAssetsHelper"
    )
    @mock.patch(
        "datadayessentials.data_retrieval._project_dataset_manager.DataLakeCSVSaver"
    )
    def test_register_csv(
        self,
        mock_data_lake_csv_saver,
        mock_data_lake_project_assets_helper,
    ):
        dataset_manager = ProjectDatasetManager(project="test")
        data = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

        (
            data_type,
            save_blob,
        ) = ProjectDatasetManager._register_dataset_based_on_datatype(
            dataset_manager, data, "test_dataset", False
        )

        self.assertEqual(data_type, "csv")
        mock_data_lake_project_assets_helper.assert_called_once()
        mock_data_lake_csv_saver.assert_called_once()

    @mock.patch(
        "datadayessentials.data_retrieval._project_dataset_manager.DatalakeProjectAssetsHelper"
    )
    @mock.patch(
        "datadayessentials.data_retrieval._project_dataset_manager.DataLakeJsonSaver"
    )
    def test_register_json(
        self,
        mock_data_lake_json_saver,
        mock_data_lake_project_assets_helper,
    ):
        dataset_manager = ProjectDatasetManager(project="test")
        data = {"col1": [1, 2, 3], "col2": [4, 5, 6]}

        (
            data_type,
            save_blob,
        ) = ProjectDatasetManager._register_dataset_based_on_datatype(
            dataset_manager, data, "test_dataset", False
        )

        self.assertEqual(data_type, "json")
        mock_data_lake_project_assets_helper.assert_called_once()
        mock_data_lake_json_saver.assert_called_once()

    def test_list_datasets(self):
        project = "datadayessentials"
        dataset_manager = ProjectDatasetManager(project=project)
        assets = dataset_manager.list_datasets()

        assert isinstance(assets, list)
        assert len(assets) > 0

    def test_list_dataset_descriptions(self):
        project = "datadayessentials"
        dataset_manager = ProjectDatasetManager(project=project)
        assets = dataset_manager.list_dataset_descriptions()

        assert isinstance(assets, dict)

        assets = dataset_manager.list_datasets()

        named_assets = dataset_manager.list_dataset_descriptions(
            datasets=assets[0:1] if len(assets) > 1 else assets[0]
        )
        assert isinstance(named_assets, dict)
        assert list(named_assets.keys())[0] in assets

        versions = {assets[-1]: 1}
        version_assets = dataset_manager.list_dataset_descriptions(version=versions)
        assert list(version_assets.keys())[0] in assets

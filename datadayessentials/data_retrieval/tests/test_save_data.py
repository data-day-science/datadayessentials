import os
import unittest
import pandas as pd
from ...authentications import DataLakeAuthentication
import unittest.mock as mock
from .._save_data import (
    DataLakeCSVSaver,
    BlobLocation,
    DataLakeJsonSaver,
    DataLakePickleSaver,
)
import logging
from .test_data import test_path
from ...config import Config
from azure.storage.filedatalake import DataLakeServiceClient, DataLakeFileClient

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class TestBlobLocation(unittest.TestCase):
    def test_equal(self):
        blob1 = BlobLocation("acc", "container", "folder", "filename")
        blob2 = BlobLocation("acc", "container", "folder", "filename")
        equality = blob1 == blob2
        assert equality == True

    def test_not_equals(self):
        blob1 = BlobLocation("acc", "container", "folder", "filename")
        blob2 = BlobLocation("acc2", "container", "folder", "filename")
        equality = blob1 == blob2
        assert equality == False


class TestDataLakeCSVSaver:
    @mock.patch("azure.storage.filedatalake.DataLakeFileClient.upload_data")
    def test_save(self, mock_save_file):
        mock_authentication = DataLakeAuthentication()
        mock_authentication.get_azure_credentials = mock.MagicMock()

        blob_location = BlobLocation(
            account=Config().get_environment_variable("data_lake"),
            container="test",
            filename="test2.csv",
            filepath=".",
        )
        data_saver = DataLakeCSVSaver(authentication=mock_authentication)

        test_df = pd.read_csv(os.path.join(test_path, "test_csv.csv"))

        logger.debug("read file")
        data_saver.save(blob_location=blob_location, df=test_df)
        assert mock_save_file.called


class TestDataLakeJsonSaver:
    @mock.patch("azure.storage.filedatalake.DataLakeFileClient.upload_data")
    def test_save(self, mock_save_file):
        mock_authentication = DataLakeAuthentication()
        mock_authentication.get_azure_credentials = mock.MagicMock()

        blob_location = BlobLocation(
            account=Config().get_environment_variable("data_lake"),
            container="test",
            filename="test2.json",
            filepath=".",
        )
        data_saver = DataLakeJsonSaver(authentication=mock_authentication)

        test_data = ["1", "2"]

        logger.debug("read file")
        data_saver.save(blob_location=blob_location, data=test_data)
        assert mock_save_file.called


class TestDataLakePickleSaver:
    @mock.patch("azure.storage.filedatalake.DataLakeFileClient.upload_data")
    def test_save(self, mock_save_file):
        mock_authentication = DataLakeAuthentication()
        mock_authentication.get_azure_credentials = mock.MagicMock()
        blob_location = BlobLocation(
            account=Config().get_environment_variable("data_lake"),
            container="test",
            filename="test2.json",
            filepath=".",
        )
        data_saver = DataLakePickleSaver(authentication=mock_authentication)

        test_data = {"1": [1, 2, 3], "2": {"234": [1, 2, 3]}}

        logger.debug("read file")
        data_saver.save(blob_location=blob_location, data=test_data)
        assert mock_save_file.called

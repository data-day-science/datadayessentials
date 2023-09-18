from .._base import IDataFrameLoader
import unittest
import os

from ...authentications import DatabaseAuthentication, DataLakeAuthentication
from .._base import IURIGenerator
import unittest.mock as mock
from unittest.mock import DEFAULT, MagicMock, patch
from .._load_data import (
    TableLoader,
    DataFrameTap,
    DataLakeCSVLoader,
    DataCacher,
    DataLakeJsonLoader,
    DataLakePickleLoader,
    DataLakeParquetLoader
)
from .._save_data import BlobLocation
import pandas as pd
import logging
from io import BytesIO
import pandas as pd
import copy
from dateutil.relativedelta import relativedelta
import pytest
from datetime import datetime
from ...config import Config
import pickle
import json
from types import SimpleNamespace


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class FakeURIGenerator(IURIGenerator):
    def __init__(self, n):
        self.n = n

    def get_uris(self):
        return [
            BlobLocation("storate_acc", "container_name", "filepath", "filename")
        ] * self.n

class content_settings:
    def __init__(self,content_type):
        self.content_type = content_type

class TestTableLoader(unittest.TestCase):
    @mock.patch("pandas.read_sql")
    @mock.patch("pyodbc.connect")
    @mock.patch("datadayessentials.authentications.SQLServerConnection", autospec=True)
    @mock.patch("datadayessentials.config.Config.get_environment_variable")
    def test_load(self, get_db_mock, run_sql_mock, pyodbc_mock, pd_mock):
        authentication = mock.MagicMock()
        authentication.get_credentials = mock.MagicMock(
            return_value={"USERNAME": "username", "PASSWORD": "password"}
        )
        sql_statement = "I AM AN SQL STATEMENT"
        table_loader = TableLoader(sql_statement, use_cache=False, authentication=authentication)
        df = table_loader.load()
        logger.debug(f"SQL statement called: {pyodbc_mock.call_args.args[0]}")
        assert pyodbc_mock.called
        assert pd_mock.called
        assert len(pd_mock.call_args.args) == 2


class TestDataLakeCSVLoader(unittest.TestCase):
    @mock.patch("azure.storage.filedatalake.DataLakeFileClient.download_file")
    @mock.patch(
        "azure.storage.filedatalake.DataLakeFileClient.get_file_properties",
        return_value=SimpleNamespace(last_modified=datetime(2022, 1, 1)),
    )
    def test_load_1_file(self, mock_file_properties, mock_download_file):
        # Prepare
        authentication = DataLakeAuthentication()
        csv_buffer = BytesIO()
        with open(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "test_data", "test_csv.csv"
            ),
            "rb",
        ) as f:
            csv_buf = BytesIO(f.read())

        csv_buf.seek(0)
        expected_df = pd.read_csv(copy.copy(csv_buf))
        mock_download_file.side_effect = [copy.deepcopy(csv_buf)]
        test_uri_generator = FakeURIGenerator(1)

        # Test
        csv_loader = DataLakeCSVLoader(authentication)
        output_df = csv_loader.load_from_uri_generator(test_uri_generator)

        # Evaluate
        assert output_df.equals(expected_df)

    @mock.patch("azure.storage.filedatalake.DataLakeFileClient.download_file")
    @mock.patch(
        "azure.storage.filedatalake.DataLakeFileClient.get_file_properties",
        return_value=SimpleNamespace(last_modified=datetime(2022, 1, 1)),
    )
    def test_load_2_file(self, mock_file_properties, mock_download_file):
        # Prepare
        authentication = mock.MagicMock()
        csv_buffer = BytesIO()
        with open(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "test_data", "test_csv.csv"
            ),
            "rb",
        ) as f:
            csv_buf = BytesIO(f.read())

        csv_buf.seek(0)
        df = pd.read_csv(copy.copy(csv_buf))
        expected_df = pd.concat([df, df])
        mock_download_file.side_effect = [
            copy.deepcopy(csv_buf),
            copy.deepcopy(csv_buf),
        ]
        test_uri_generator = FakeURIGenerator(2)

        # Test
        csv_loader = DataLakeCSVLoader(authentication)
        output_df = csv_loader.load_from_uri_generator(test_uri_generator)

        # Evaluate
        assert output_df.equals(expected_df)


class TestDataLakeJsonLoader(unittest.TestCase):
    @mock.patch("azure.storage.filedatalake.DataLakeFileClient.download_file")
    @mock.patch(
        "azure.storage.filedatalake.DataLakeFileClient.get_file_properties",
        return_value=SimpleNamespace(last_modified=datetime(2022, 1, 1)),
    )
    def test_load_file(self, mock_file_properties, mock_download_file):
        # Prepare
        authentication = DataLakeAuthentication()
        with open(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "test_data",
                "test_json.json",
            ),
            "rb",
        ) as f:
            json_buf = BytesIO(f.read())

        json_buf.seek(0)
        expected_dict = json.load(copy.copy(json_buf))
        mock_download_file.side_effect = [copy.deepcopy(json_buf)]

        # Test
        json_loader = DataLakeJsonLoader(authentication)
        output_dict = json_loader.load(BlobLocation("dsafs", "asd", "sadfa", "fasdf"))

        # Evaluate
        self.assertDictEqual(expected_dict, output_dict)

class TestDataLakeParquetLoader(unittest.TestCase):
    #This class should test
    #1. That the parquet loader can load a parquet file
    #2. That the parquet loader wont load a file if it isnt a parquet file
    #3. That the parquet Loader will throw an error if the file doesnt exist


    #if a test parquet file doesnt exist in the test_data folder, generate a test parquet file to add to the test_data folder
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data", "test_parquet.parquet")):
        parquet_file = pd.DataFrame({"col1": [1, 2, 3], "col2": [1, 2, 3]})
        parquet_file.to_parquet(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data", "test_parquet.parquet"))

    @mock.patch("azure.storage.filedatalake.DataLakeFileClient.download_file")
    @mock.patch("azure.storage.filedatalake.DataLakeFileClient.exists")
    @mock.patch(
        "azure.storage.filedatalake.DataLakeFileClient.get_file_properties",
        return_value=SimpleNamespace(last_modified=datetime(2022, 1, 1),content_settings= content_settings(content_type="application/octet-stream")),
    )
    def test_load_file(self, mock_file_properties, mock_file_exists, mock_download_file):
        # Prepare
        authentication = mock.MagicMock()
        authentication.get_credentials = mock.MagicMock(
            return_value={"USERNAME": "username", "PASSWORD": "password"}
        )
        with open(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "test_data",
                "test_parquet.parquet",
            ),
            "rb",
        ) as f:
            parquet_buf = BytesIO(f.read())

        parquet_buf.seek(0)
        expected_df = pd.read_parquet(copy.copy(parquet_buf))
        mock_file_exists.return_value = True
        mock_download_file.side_effect = [copy.deepcopy(parquet_buf)]

        # Test
        parquet_loader = DataLakeParquetLoader(authentication)
        output_df = parquet_loader.load(BlobLocation("dsafs", "asd", "sadfa", "fasdf"))

        # Evaluate
        assert output_df.equals(expected_df)

    #test 2
    @mock.patch("azure.storage.filedatalake.DataLakeFileClient.download_file")
    @mock.patch("azure.storage.filedatalake.DataLakeFileClient.exists")
    @mock.patch(
        "azure.storage.filedatalake.DataLakeFileClient.get_file_properties",
        return_value=SimpleNamespace(last_modified=datetime(2022, 1, 1),content_settings= content_settings(content_type="not_a_csv")),
    )

    def test_load_non_parquet_file(self, mock_file_properties, mock_file_exists, mock_download_file):
        # Prepare

        authentication = mock.MagicMock()
        authentication.get_credentials = mock.MagicMock(
            return_value={"USERNAME": "username", "PASSWORD": "password"}
        )

        with open(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "test_data",
                "test_csv.csv",
            ),
            "rb",
        ) as f:
            csv_buf = BytesIO(f.read())

        csv_buf.seek(0)
        mock_file_exists.return_value = True
        mock_download_file.side_effect = [copy.deepcopy(csv_buf)]

        # Test
        parquet_loader = DataLakeParquetLoader(authentication)
        with pytest.raises(ValueError):
            output_df = parquet_loader.load(BlobLocation("dsafs", "asd", "sadfa", "fasdf"))

    @mock.patch("azure.storage.filedatalake.DataLakeFileClient.download_file")
    @mock.patch("azure.storage.filedatalake.DataLakeFileClient.exists")
    @mock.patch(
        "azure.storage.filedatalake.DataLakeFileClient.get_file_properties",
        return_value=SimpleNamespace(last_modified=datetime(2022, 1, 1),content_settings= content_settings(content_type="application/octet-stream")),
    )
    def test_load_non_existent_file(self, mock_file_properties, mock_file_exists, mock_download_file):
        # Prepare
        authentication = mock.MagicMock()
        authentication.get_credentials = mock.MagicMock(
            return_value={"USERNAME": "username", "PASSWORD": "password"}
        )
        with open(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "test_data",
                "test_parquet.parquet",
            ),
            "rb",
        ) as f:
            parquet_buf = BytesIO(f.read())

        parquet_buf.seek(0)
        mock_file_exists.return_value = False
        mock_download_file.side_effect = [copy.deepcopy(parquet_buf)]

        # Test
        parquet_loader = DataLakeParquetLoader(authentication)
        with pytest.raises(FileNotFoundError):
            output_df = parquet_loader.load(BlobLocation("dsafs", "asd", "sadfa", "fasdf"))







        
    


class MockStorageStreamDownloader:
    def __init__(self, bytes):
        self.bytes = bytes

    def readall(self):
        return self.bytes

class TestDataLakePickleLoader(unittest.TestCase):
    @mock.patch("azure.storage.filedatalake.DataLakeFileClient.download_file")
    @mock.patch(
        "azure.storage.filedatalake.DataLakeFileClient.get_file_properties",
        return_value=SimpleNamespace(last_modified=datetime(2022, 1, 1)),
    )
    def test_load_file(self, mock_file_properties, mock_download_file):
        # Prepare
        authentication = DataLakeAuthentication()
        obj = pickle.dumps("Hello World!")
        obj_buf = MockStorageStreamDownloader(obj)
        obj_io = BytesIO(obj)
        obj_io.seek(0) 
        expected_obj = pickle.load(obj_io)
        mock_download_file.side_effect = [copy.deepcopy(obj_buf)]

        # Test
        pickle_loader = DataLakePickleLoader(authentication)
        unpickled_pickled_object = pickle_loader.load(
            BlobLocation("dsafs", "sdfa", "asdf", "sdg")
        )

        # Evaluate
        assert expected_obj == unpickled_pickled_object


class TestDataCacher(unittest.TestCase):
    def test_cache_df(self):
        cacher = DataCacher("https://example_file.com")
        if os.path.exists(cacher.file_path):
            os.remove(cacher.file_path)
        assert cacher.is_file_in_cache() == False
        example_df = pd.DataFrame({"col1": [1, 2, 3], "col2": [1, 2, 3]})
        cacher.save_df_to_cache(example_df)
        assert cacher.is_file_in_cache() == True
        loaded_df = cacher.get_df_from_cache()
        assert loaded_df.equals(example_df)

    def test_cache_json(self):
        cacher = DataCacher("https://example_file.com")
        if os.path.exists(cacher.file_path):
            os.remove(cacher.file_path)
        assert cacher.is_file_in_cache() == False
        example_json = {"col1": [1, 2, 3], "col2": [1, 2, 3]}
        cacher.save_json_to_cache(example_json)
        assert cacher.is_file_in_cache() == True
        loaded_json = cacher.get_json_from_cache()
        self.assertDictEqual(loaded_json, example_json)

    def test_cache_pickle(self):
        cacher = DataCacher("https://example_file.com")
        if os.path.exists(cacher.file_path):
            os.remove(cacher.file_path)
        assert cacher.is_file_in_cache() == False
        example_obj = {"col1": [1, 2, 3], "col2": [1, 2, 3]}
        cacher.save_pickle_to_cache(example_obj)
        assert cacher.is_file_in_cache() == True
        loaded_obj = cacher.get_pickle_from_cache()
        self.assertDictEqual(loaded_obj, example_obj)


class TestDataTap:
    @mock.patch(
        "datadayessentials.data_retrieval._base.IDataFrameLoader.load",
        return_value=1,
    )
    @mock.patch.object(IDataFrameLoader, "__abstractmethods__", set())
    @mock.patch("datadayessentials.data_transformation.DataFrameCaster.process")
    @mock.patch(
        "datadayessentials.data_retrieval._validate_data.DataFrameValidator.validate",
        return_value=2,
    )
    def test_run2(self, mock_validate, mock_process, mock_load):
        # data = Mock()
        # data.load.return_value = 1
        data = IDataFrameLoader()

        data_tap = DataFrameTap(data, DEFAULT, DEFAULT)
        results = data_tap.run()
        mock_validate.assert_called_once_with(1)
        mock_process.assert_called_once_with(2)

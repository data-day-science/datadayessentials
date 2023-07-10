from .._base import IDataFrameLoader
import unittest
import os

from ...authentications import DatabaseAuthentication, DataLakeAuthentication
from .._base import IURIGenerator
import unittest.mock as mock
from unittest.mock import DEFAULT
from .._load_data import (
    TableLoader,
    DataFrameTap,
    DataLakeCSVLoader,
    CreditDataURIGenerator,
    CreditDataLoader,
    DataCacher,
    DataLakeJsonLoader,
    DataLakePickleLoader,
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
from ...config import LocalConfig
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


class TestTableLoader(unittest.TestCase):
    @mock.patch("pandas.read_sql")
    @mock.patch("pyodbc.connect")
    def test_load(self, pyodbc_mock, pd_mock):
        authentication = DatabaseAuthentication()
        authentication.get_credentials = mock.MagicMock(
            return_value={"USERNAME": "username", "PASSWORD": "password"}
        )
        sql_statement = "I AM AN SQL STATEMENT"
        table_loader = TableLoader(sql_statement, use_cache=False, authentication=authentication)
        df = table_loader.load()
        logger.debug(f"SQL statement called: {pyodbc_mock.call_args.args[0]}")
        assert pyodbc_mock.called
        assert pd_mock.called
        assert (
            pyodbc_mock.call_args.args[0]
            == "DRIVER={ODBC Driver 17 for SQL Server};SERVER=cjzbghrawq2.database.windows.net;DATABASE=DW;ENCRYPT=yes;UID=username;PWD=password"
        )
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
        authentication = DatabaseAuthentication()
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
        obj_buf = BytesIO(obj)

        obj_buf.seek(0)
        expected_obj = pickle.load(copy.copy(obj_buf))
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


class TestCreditDataURIGenerator(unittest.TestCase):
    def test_get_uris(self):
        start_date = pd.to_datetime("30-01-2022")
        end_date = pd.to_datetime("30-10-2022")

        credit_uri_generator = CreditDataURIGenerator(start_date, end_date)
        credit_uris = credit_uri_generator.get_uris()

        expected_output = [
            "epoch_1464.csv",
            "epoch_1465.csv",
            "epoch_1466.csv",
            "epoch_1467.csv",
            "epoch_1468.csv",
            "epoch_1469.csv",
            "epoch_1470.csv",
            "epoch_1471.csv",
            "epoch_1472.csv",
            "epoch_1473.csv",
        ]
        cra_data_location = LocalConfig.get_data_lake_folder(
            named_folder="cra_data", use_current_environment=False
        )
        expected_output = [
            BlobLocation(
                cra_data_location["data_lake"],
                cra_data_location["container"],
                cra_data_location["path"],
                filepath,
            )
            for filepath in expected_output
        ]
        logger.debug(f"The expected output blob location is {expected_output}")
        logger.debug(f"The actual blob location is {credit_uris}")
        self.assertListEqual(credit_uris, expected_output)

    def test_get_uris_max_end_date(self):
        start_date = pd.to_datetime("30-01-2022")
        end_date = datetime.now() + relativedelta(months=1)
        with pytest.raises(ValueError):
            credit_uri_generator = CreditDataURIGenerator(start_date, end_date)


class TestCreditDataLoader(unittest.TestCase):
    @mock.patch(
        "datascience_core.data_retrieval._load_data.DataLakeCSVLoader.load_from_uri_generator"
    )
    def test_raises_missing_col_error(self, mock_csv_loader):
        current_datetime = datetime.now() - relativedelta(months=1)
        example_credit_df = pd.DataFrame(
            {"col1": [1, 2, 3, 4, 5], "App ApplicationDate": [current_datetime] * 5}
        )

        mock_csv_loader.load.return_value = example_credit_df

        mock_authentication = DataLakeAuthentication()
        mock_authentication.get_azure_credentials = mock.MagicMock()

        end_date = datetime.now()
        start_date = pd.to_datetime("30-01-2022")

        with pytest.raises(ValueError):
            test_output = CreditDataLoader(
                authentication=mock_authentication, start_time=start_date, end_time=end_date, use_cache=False
            ).load()

    @mock.patch(
        "datascience_core.data_retrieval._load_data.DataLakeCSVLoader.load_from_uri_generator"
    )
    def test_load(self, mock_csv_load_func):
        current_datetime = datetime.now() - relativedelta(months=1)
        example_credit_df = pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "App.ApplicationDate": ["2020-11-01 01:14:00"] * 5,
            }
        )
        logger.debug(
            f"The dtype of the examplmsge_credit_df is {example_credit_df.dtypes}"
        )
        mock_csv_load_func.return_value = example_credit_df

        mock_authentication = DataLakeAuthentication()
        mock_authentication.get_azure_credentials = mock.MagicMock()
        end_date = pd.to_datetime("02/11/2020", format="%d/%m/%Y")
        start_date = pd.to_datetime("01/11/2020", format="%d/%m/%Y")

        test_output = CreditDataLoader(
            start_date, end_date, use_cache=False, authentication=mock_authentication
        ).load()
        assert mock_csv_load_func.called
        logger.debug(f"The expected df is {example_credit_df}")
        logger.debug(f"The actual df is {test_output}")
        assert test_output.equals(example_credit_df)

    @pytest.mark.skip(
        reason="For local testing only - credit data takes too long to load in CI"
    )
    def test_load_live(self):
        start_date = datetime(2021, 2, 1)
        end_date = datetime(2021, 2, 25)
        data_lake_authentication = DataLakeAuthentication()
        credit_data_loader = CreditDataLoader(
            data_lake_authentication, start_date, end_date
        )
        credit_data = credit_data_loader.load(use_cache=False)
        logger.debug(f"The loaded credit data has shape {credit_data.shape}")
        assert credit_data.shape[0] > 100


class TestDataTap:
    @mock.patch(
        "datascience_core.data_retrieval._base.IDataFrameLoader.load",
        return_value=1,
    )
    @mock.patch.object(IDataFrameLoader, "__abstractmethods__", set())
    @mock.patch("datascience_core.data_transformation.DataFrameCaster.process")
    @mock.patch(
        "datascience_core.data_retrieval._validate_data.DataFrameValidator.validate",
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

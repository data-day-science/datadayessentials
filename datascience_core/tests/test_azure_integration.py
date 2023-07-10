import unittest

import pytest
from ..authentications import DatabaseAuthentication, DataLakeAuthentication
import logging
from ..data_retrieval import (
    DataLakeCSVLoader,
    DataLakeCSVSaver,
    BlobLocation,
    QueryFactory,
    TableLoader,
)
from ..data_retrieval import IURIGenerator
import pandas as pd
from ..config import LocalConfig
from dateutil.relativedelta import relativedelta
from datetime import datetime


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class FakeURIGenerator(IURIGenerator):
    def __init__(self, blob_location):
        self.blob_location = blob_location

    def get_uris(self):
        return [self.blob_location]


class TestAzureIntegration(unittest.TestCase):
    def test_get_dwh_credentials(self):
        dwh_credentials = DatabaseAuthentication().get_credentials()
        assert isinstance(dwh_credentials, dict)
        assert "USERNAME" in dwh_credentials
        assert "PASSWORD" in dwh_credentials
        assert dwh_credentials["USERNAME"] is not None
        assert dwh_credentials["PASSWORD"] is not None

    @pytest.mark.skip(reason="Azure issue uploading data with VPN")
    def test_data_lake_connection(self):
        credentials = DataLakeAuthentication()
        saver = DataLakeCSVSaver(credentials)
        loader = DataLakeCSVLoader(credentials)

        test_df = pd.DataFrame({"col2": [2, 3, 4, 5], "col2": [1, 2, 3, 4]})
        blob_location = BlobLocation(
            LocalConfig.get_data_lake(), "test", "folder1", "test.csv"
        )
        uri_generator = FakeURIGenerator(blob_location)
        saver.save(blob_location, test_df)

        returned_df = loader.load_from_uri_generator(uri_generator)
        logger.debug(f"The returned_df is {returned_df}")
        logger.debug(f"The expected_df is {test_df}")


class TestDWHIntegration(unittest.TestCase):
    def test_load_table(self):
        current_date = datetime.now() - relativedelta(months=1)
        yesterday = current_date - relativedelta(days=1)
        one_day_query = QueryFactory.get_allocated_volume_query(yesterday, current_date)
        table_loader = TableLoader(one_day_query)
        data = table_loader.load()
        logger.debug(f"The data retrieved from the warehouse is {data.head()}")

    def test_credit_checks_db(self):
        current_date = datetime.now() - relativedelta(months=1)
        yesterday = current_date - relativedelta(days=1)
        example_query = "SELECT TOP 100 * FROM FACT.ApplicationIdToCreditCheckId"
        table_loader = TableLoader(
            example_query, server_name="credit_check_prod"
        )
        data = table_loader.load()
        logger.debug(f"The data retrieved from the warehouse is {data.head()}")

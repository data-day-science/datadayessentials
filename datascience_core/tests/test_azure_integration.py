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



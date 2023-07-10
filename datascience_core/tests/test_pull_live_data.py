import pytest
from datascience_core.data_retrieval import QueryFactory, TableLoader
from datascience_core.authentications import DatabaseAuthentication, DataLakeAuthentication
import datetime
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TestDefaultLabelPulling:
    def test_default_data(self):

        start_date = datetime.datetime(2021, 1, 1)
        end_date = datetime.datetime(2021, 2, 10)

        sql_statement = QueryFactory.get_default_label_query(start_date, end_date)
        sql_statement_short = "SELECT TOP(10) " + sql_statement[7:]
        default_label_perf_loader = TableLoader(sql_statement_short)
        df = default_label_perf_loader.load()
        assert df.shape[0] == 10
        assert df.shape[1] > 0


class TestAllocationsPulling:
    def test_allocations_data(self):
        start_date = datetime.datetime(2021, 1, 1)
        end_date = datetime.datetime(2021, 2, 10)

        sql_statement = QueryFactory.get_allocated_volume_query(start_date, end_date)
        sql_statement_short = "SELECT TOP(9) " + sql_statement[7:]
        default_label_perf_loader = TableLoader(sql_statement_short)
        df = default_label_perf_loader.load()
        logger.debug(df)
        assert df.shape[0] == 9
        assert df.shape[1] > 0


class TestAppPulling:
    def test_allocations_data(self):

        sql_statement = QueryFactory.get_app_payload_query(["10430432", "10446711"])
        default_label_perf_loader = TableLoader(sql_statement)
        logger.debug(sql_statement)
        df = default_label_perf_loader.load()
        logger.debug(df)
        assert df.shape[0] == 2
        assert df.shape[1] > 0


class TestPolicyCheck:
    def test_allocations_data(self):
        sql_statement = QueryFactory.get_policy_check_query([10430432, 10446711])
        default_label_perf_loader = TableLoader(sql_statement)
        logger.debug(sql_statement)
        df = default_label_perf_loader.load()
        logger.debug(df)
        assert df.shape[0] == 2
        assert df.shape[1] > 0

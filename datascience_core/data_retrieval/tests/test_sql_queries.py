import unittest
from ...data_retrieval import QueryFactory
import pytest
from datetime import datetime


def to_datetime(str_datetime):
    return datetime.strptime(str_datetime, "%Y-%m-%d")


class TestQueryFactory(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    # Need to check validation of input parameters and that the correct query is being returned
    def test_get_default_label_query(self):
        start_date = to_datetime("1999-01-01")
        end_date = to_datetime("2022-01-01")
        formatted_query = QueryFactory.get_default_label_query(start_date, end_date)
        print(formatted_query)
        expected_result = """SELECT 
[Broker Reference] AS ApplicationId,
ApplicationDate,
[Start Date],
DefaultedInFirst12Months AS [Retro Three_Plus_In_Twleve],
NumDaysOpen,
OpenGT370Days
FROM analyst.DefaultModelling
WHERE [Start Date] between '1999-01-01' and '2022-01-01'
"""
        assert formatted_query.replace("\t", "") == expected_result

    def test_get_default_label_query_parameter_validation(self):
        start_date = "hello"
        end_date = to_datetime("2022-01-01")
        with pytest.raises(ValueError):
            formatted_query = QueryFactory.get_default_label_query(start_date, end_date)

        start_date = to_datetime("2022-01-01")
        end_date = "hello"
        with pytest.raises(ValueError):
            formatted_query = QueryFactory.get_default_label_query(start_date, end_date)

        start_date = 20
        end_date = to_datetime("2022-02-02")
        with pytest.raises(ValueError):
            formatted_query = QueryFactory.get_default_label_query(start_date, end_date)

    def test_get_app_payload_query(self):
        app_ids = ["123098"]
        formatted_query = QueryFactory.get_app_payload_query(app_ids)
        assert (
            formatted_query
            == "EXEC [DataScience].[Get_AppPayload_Scorecard] @AppIds = ['123098']\n"
        )

    def test_get_app_payload_query_parameter_validation(self):
        app_ids = 12809
        with pytest.raises(TypeError):
            formatted_query = QueryFactory.get_app_payload_query(app_ids)

        app_ids = [235412]
        with pytest.raises(TypeError):
            formatted_query = QueryFactory.get_app_payload_query(app_ids)

    def test_get_allocated_volume_query(self):
        start_date = to_datetime("1999-01-01")
        end_date = to_datetime("2022-01-01")
        formatted_query = QueryFactory.get_allocated_volume_query(start_date, end_date)

        
        expected_start_date = '1999-01-01'
        expected_end_date = '2022-01-01'
        assert expected_start_date in formatted_query
        assert expected_end_date in formatted_query

    def test_get_allocated_volume_quuery_parameter_validation(self):
        start_date = "hello"
        end_date = to_datetime("2022-01-01")
        with pytest.raises(ValueError):
            formatted_query = QueryFactory.get_allocated_volume_query(
                start_date, end_date
            )

        start_date = to_datetime("2022-01-01")
        end_date = "hello"
        with pytest.raises(ValueError):
            formatted_query = QueryFactory.get_allocated_volume_query(
                start_date, end_date
            )

        start_date = 20
        end_date = to_datetime("2022-02-02")
        with pytest.raises(ValueError):
            formatted_query = QueryFactory.get_allocated_volume_query(
                start_date, end_date
            )

    def test_get_policy_check_data_type_err(self):
        application_ids = "111"
        with pytest.raises(TypeError):
            QueryFactory.get_policy_check_query(application_ids=application_ids)

    def test_get_policy_check_data(self):
        application_ids = ["1", "2", "3"]
        query = QueryFactory.get_policy_check_query(application_ids=application_ids)
        expected_query = "EXEC DataScience.Get_PolicyCheckByDecision  @AppicationIdCSLList = ['1', '2', '3']"
        assert expected_query == query

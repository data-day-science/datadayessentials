from abc import abstractmethod
from ._base import ISQLQueryFormatter
import os
from typing import List
import re
import logging
from datetime import date, datetime
from datascience_core.data_retrieval import ProjectDatasetManager


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



class SQLQueryFormatter(ISQLQueryFormatter):
    """Responsible for loading and formatting queries based on the input parameters."""

    def __init__(self, raw_query: str, params: dict) -> None:
        """Creates a SQLFormatter instance

        Args:
            query_path (str): Location of the query to format, containing format strings '{param1}' that can be formatted by the params
            params (dict): Key value pairs of parameters for formatting the query
        """
        self.raw_query = raw_query
        self.params = params

    

    def _format_query(self, not_formatted_query: str) -> str:
        """Formats a query string with the parameter dictionary

        Args:
            not_formatted_query (str): Raw query string with format strings

        Returns:
            str: Formatted query with the parameters
        """
        formatted_query = not_formatted_query.format(**self.params)
        return formatted_query

    def get_query(self) -> str:
        """Function for loading and formatting a query

        Returns:
            str: Formatted query
        """
        logger.debug("Formatting query with parameters")
        formatted_query = self._format_query(self.raw_query)
        logger.debug(f"Formatted query: \n {formatted_query}")
        return formatted_query


class QueryFactory:
    """
    Responsible for managing the parameter validation needed for speciric queries. Each query stored in the sql_queries folder will have a different function in this class.
    Example Use Case:
    In the below example the query factory is used to parameterise a few queries. 
    ```python
    from datascienc_core.data_retrieval import QueryFactory
    from datetime import datetime

    # Return the deault labels for all applicaions between the given dates
    start_time = datetime.strptime('2022-01-01', '%Y-%m-%d')
    end_time = datetime.strptime('2022-02-01', '%Y-%m-%d')
    default_sql = QueryFactory.get_default_label_query(start_time, end_time)

    # Return the app fields for specific application IDs
    app_ids = ['123', '1234']
    app_payload_sql = QueryFactory.get_app_payload_query(app_ids)

    # Return the allocated volume for applications within a time period
    start_time = datetime.strptime('2022-01-01', '%Y-%m-%d')
    end_time = datetime.strptime('2022-02-01', '%Y-%m-%d')
    allocated_sql = QueryFactory.get_default_label_query(start_time, end_time)
    ```

    """

    @classmethod
    def get_default_label_query(
        cls,
        start_date: datetime = datetime.strptime("1980-01-01", "%Y-%m-%d"),
        end_date: datetime = datetime.strptime("2100-01-01", "%Y-%m-%d"),
    ) -> str:
        """Returns a query for loading in the default data between the provided dates

        Args:
            start_date (datetime, optional): Start date of the default data. Defaults to datetime.strptime("1980-01-01", "%Y-%m-%d").
            end_date (datetime, optional): End date of the default data. Defaults to datetime.strptime("2100-01-01", "%Y-%m-%d").

        Raises:
            ValueError: Raise if the Start or End date is not the correct type

        Returns:
            str: Formtted query
        """
        logger.debug("Validating Parameters for get_default_label_query")
        if not isinstance(start_date, datetime):
            raise ValueError("Start date must be of datetime type")
        if not isinstance(end_date, datetime):
            raise ValueError("End date must be of datetime type")

        params = {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
        }
        # Retrieve query path from config:
        query_manager = ProjectDatasetManager("datascience_core_queries")
        raw_query = query_manager.load_datasets("default_label")["default_label"]
        return SQLQueryFormatter(raw_query, params).get_query()

    @classmethod
    def get_app_payload_query(cls, application_ids: List[str]) -> str:
        """Returns a query for retrieving credit check payloads based on applicationId

        Args:
            application_ids (List[str]): List of application ID's to retrieve payloads for_

        Raises:
            TypeError: Raised if the app ids are not of the correct type

        Returns:
            str: Formatted Query
        """
        logger.debug("Validating Parameters for get_app_payload_query")
        if not isinstance(application_ids, List):
            raise TypeError("Argument application_ids must be of List[str] type")
        for elem in application_ids:
            if not isinstance(elem, str):
                raise TypeError("Argument application_ids must be of List[str] type")
        params = {"application_ids": str(application_ids)}

        query_manager = ProjectDatasetManager("datascience_core_queries")
        raw_query = query_manager.load_datasets("app_sproc")["app_sproc"]
        return SQLQueryFormatter(raw_query, params).get_query()

    @classmethod
    def get_allocated_volume_query(
        cls,
        start_date: datetime = datetime.strptime("1980-01-01", "%Y-%m-%d"),
        end_date: datetime = datetime.strptime("2100-01-01", "%Y-%m-%d"),
    ) -> str:
        """Returns a query for retrieving allocations between a provided date range

        Args:
            start_date (datetime, optional): Start date for query. Defaults to datetime.strptime("1980-01-01", "%Y-%m-%d").
            end_date (datetime, optional): End date for query. Defaults to datetime.strptime("2100-01-01", "%Y-%m-%d").

        Raises:
            ValueError: Raised if input dates are not of the correct type

        Returns:
            str: Formatted Query
        """
        logger.debug("Validating parameters for get_allocated_volume_query")
        if not isinstance(start_date, datetime):
            raise ValueError("Start date must be of datetime type")
        if not isinstance(end_date, datetime):
            raise ValueError("End date must be of datetime type")
        params = {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
        }
        query_manager = ProjectDatasetManager("datascience_core_queries")
        raw_query = query_manager.load_datasets("allocations")["allocations"]
        return SQLQueryFormatter(raw_query, params).get_query()

    @classmethod
    def get_policy_check_query(cls, application_ids: List[str]) -> str:
        """Returns a query for retrieving the policy check information for provided application ID's (affordability checks etc.)

        Args:
            application_ids (List[str]): ApplicationIds to retrieve policy check information for

        Raises:
            TypeError: Raised if arguments are of the incorrect type

        Returns:
            str: Formatted query
        """
        logger.debug("Validating parameters for policy check query")
        if not isinstance(application_ids, list):
            raise TypeError("application_ids must be a list")

        params = {"application_ids": f"{str(application_ids)}"}
        query_manager = ProjectDatasetManager("datascience_core_queries")
        raw_query = query_manager.load_datasets("policy_check")["policy_check"]
        return SQLQueryFormatter(raw_query, params).get_query()

    @classmethod
    def get_policy_check_date_range_query(
        cls, start_date: datetime, end_date: datetime
    ) -> str:
        """Returns a query for getting policy check information for applications between a provided date range

        Args:
            start_date (datetime): Start date of the applications
            end_date (datetime): End date of the applications

        Raises:
            TypeError: Raised if the input arguments are of the incorrect type

        Returns:
            str: Formatted Query
        """
        logger.debug("Validating parameters for policy check query")
        if not isinstance(start_date, datetime):
            raise TypeError("Start date must be a datetime")

        if not isinstance(end_date, datetime):
            raise TypeError("end date must be a datetime")

        params = {
            "start_date": f"{start_date.strftime('%Y%m%d')}",
            "end_date": f"{end_date.strftime('%Y%m%d')}",
        }
        query_manager = ProjectDatasetManager("datascience_core_queries")
        raw_query = query_manager.load_datasets("policy_check_date_range")["policy_check_date_range"]
        return SQLQueryFormatter(raw_query, params).get_query()

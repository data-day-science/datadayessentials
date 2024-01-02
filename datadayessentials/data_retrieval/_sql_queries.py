from ._base import ISQLQueryFormatter
import os
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SQLQueryFormatter(ISQLQueryFormatter):
    """Responsible for loading and formatting queries based on the input parameters."""

    query_folder = os.path.join(os.path.dirname(__file__), "sql_queries")

    def __init__(self, query_name: str, params: dict) -> None:
        """Creates a SQLFormatter instance

        Args:
            query_path (str): Location of the query to format, containing format strings '{param1}' that can be formatted by the params
            params (dict): Key value pairs of parameters for formatting the query
        """
        self.query_name = query_name
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
        raw_query = self._load_query(self.query_name)
        logger.debug("Formatting query with parameters")
        formatted_query = self._format_query(raw_query)
        logger.debug(f"Formatted query: \n {formatted_query}")
        return formatted_query

    def _load_query(self, query_name):
        """
        Loads a query from the query folder
        """
        query_path = os.path.join(self.query_folder, query_name + ".sql")
        with open(query_path, "r") as query_file:
            query = query_file.read()
        return query

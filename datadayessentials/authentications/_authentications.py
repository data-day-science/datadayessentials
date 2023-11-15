from ._base import IAuthentication, ISQLServerConnection
import pandas as pd
import pyodbc
import os
import pandas as pd
import logging
import os
from azure.keyvault.secrets import SecretClient
from datadayessentials.config._config import Config


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DatabaseAuthentication(IAuthentication):
    """
    Class for authenticating with DWH, our data warehouse.
    Example Use Case:
    ```
    from datadayessentials.authentications import DatabaseAuthentication, DataLakeAuthentication, SQLServerConnection

    # Create a DatabaseAuthentication object
    authentication = DatabaseAuthentication()
    credentials_dict = database_authentication.get_credentials()
    ```
    """

    def __init__(self, database_reference: str = "readable-secondary") -> None:
        self.database_reference = database_reference

    def get_credentials(self, primary=False) -> dict:
        """Fetches username and password for connecting to a database

        Retrieves the DWH lgin credentials from our secret manager in azure

        Returns:
            dict: Login credentials for DWH
        """
        logger.debug("Fetching Credentials")
        credentials = super().get_azure_credentials()

        username = Config().get_environment_variable(
            f"{self.database_reference}-username"
        )
        password = Config().get_environment_variable(
            f"{self.database_reference}-password"
        )

        return {"USERNAME": username, "PASSWORD": password}


class SQLServerConnection(ISQLServerConnection):
    """
    Responsible for managing connections to an SQL database, and running sql through those connections.
    ```
    from datadayessentials.authentications import DWHAuthentication, DataLakeAuthentication, SQLServerConnection

    # Example use case for creating an SQLServerConnection - the server details are specified in the config file
    sql_server_connection = SQLServerConnection(dwh_credentials_dict, server='readable_secondary')
    ```
    """

    def __init__(
        self,
        credentials: dict,
        database_reference: str = "readable_secondary",
    ) -> None:
        """Creates a connection object, the server name corresponds to options in the config file.

        Args:
            credentials (dict): Credentials generated from DWHAuthentication object
            server_name (str, optional): The name of the server to connect to (must have atching details in the config file). Defaults to "readable_secondary".

        Raises:
            ValueError: Raised when the server name is not available in the config file
        """

        self.credentials = credentials
        raise ValueError(f"The credentials are: {self.credentials}")

        try:
            Config().get_environment_variable(f"databases_{database_reference}")
        except KeyError:
            raise ValueError(f"The server was not recognised")
        self.database_reference = database_reference
        self.connect()

    def run_sql(self, sql_statement: str) -> pd.DataFrame:
        """Runs SQL against the data warehouse

        Args:
            sql_statement (str): SQL statement to execute

        Returns:
            pd.DataFrame: Results from the SQL server, as a pandas dataframe.
        """
        logger.debug("Running SQL")
        df = pd.read_sql(sql_statement, self.cnxn)
        return df

    def connect(self):
        """Connect to the SQL server"""
        logger.debug(f"Connecting to database")

        database_info = Config().get_environment_variable(
            f"databases_{self.database_reference}"
        )

        server = database_info["server"]
        database = database_info["database"]
        port = database_info["port"] if "port" in database_info else None
        application_intent = (
            database_info["application_intent"]
            if "application_intent" in database_info
            else None
        )

        connection_string = (
            "DRIVER={ODBC Driver 17 for SQL Server};SERVER="
            + server
            + ";DATABASE="
            + database
            + ";ENCRYPT=yes;UID="
            + self.credentials["USERNAME"]
            + ";PWD="
            + self.credentials["PASSWORD"]
            + ";Trusted_Connection=yes"
            + ";TrustServerCertificate=yes"
        )
        if port:
            connection_string += f";PORT={port}"
        if application_intent:
            connection_string += f";ApplicationIntent={application_intent}"

        self.cnxn = pyodbc.connect(connection_string)


class DataLakeAuthentication(IAuthentication):
    """
    Class for authenticating with our Azure DataLake.
    Example Use Case:
    ```
    from datadayessentials.authentications import DWHAuthentication, DataLakeAuthentication, SQLServerConnection

    # Create a DataLakeAuthentication object
    dl_authentication = DataLakeAuthentication()
    dl_credentials = dl_authentication.get_credentials()
    ```
    """

    def get_credentials(self):
        """Retrieves azure credentials, for using cloud resources. This object is needed by many other parts of core that rely on cloud services.

        Returns:
            __type__: Azure credential chain (ethods for authenticating login)
        """
        credentials = super().get_azure_credentials()
        return credentials

from ._base import IAuthentication, ISQLServerConnection
import pandas as pd
import pyodbc
import os
import pandas as pd
import logging
import os
from azure.keyvault.secrets import SecretClient

from ..config import LocalConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DatabaseAuthentication(IAuthentication):
    """
    Class for authenticating with DWH, our data warehouse. 
    Example Use Case:
    ```
    from datascience_core.authentications import DatabaseAuthentication, DataLakeAuthentication, SQLServerConnection

    # Create a DatabaseAuthentication object
    authentication = DatabaseAuthentication()
    credentials_dict = database_authentication.get_credentials()
    ```
    """
    
    def __init__(self, database_reference: str = "cf247") -> None:
        """Creates a DatabaseAuthentication object

        Args:
            database_reference (str, optional): The reference of the database to connect to that maps to the login details in the LocalConfig. Defaults to "dwh".
        """
        self.database_reference = database_reference
        self.database_credential_keys = LocalConfig.get_database_credentials(self.database_reference)

    def get_credentials(self) -> dict:
        """Fetches username and password for connecting to a database

        Retrieves the DWH lgin credentials from our secret manager in azure

        Returns:
            dict: Login credentials for DWH
        """
        logger.debug("Fetching Credentials")
        credentials = super().get_azure_credentials()
        key_vault = LocalConfig.get_key_vault()
        KVUri = f"https://{key_vault['key_vault_name']}.vault.azure.net"

        client = SecretClient(vault_url=KVUri, credential=credentials)
        username = client.get_secret(self.database_credential_keys["username_key"]).value
        password = client.get_secret(self.database_credential_keys["password_key"]).value
        logger.debug(f"Retrieved database Credentials for {self.database_reference}")
        return {"USERNAME": username, "PASSWORD": password}


class SQLServerConnection(ISQLServerConnection):
    """
    Responsible for managing connections to an SQL database, and running sql through those connections.
    ```
    from datascience_core.authentications import DWHAuthentication, DataLakeAuthentication, SQLServerConnection

    # Example use case for creating an SQLServerConnection - the server details are specified in the config file
    sql_server_connection = SQLServerConnection(dwh_credentials_dict, server='readable_secondary')
    ```
    """

    def __init__(
        self, credentials: dict, database_reference: str = "readable_secondary"
    ) -> None:
        """Creates a connection object, the server name corresponds to options in the config file.

        Args:
            credentials (dict): Credentials generated from DWHAuthentication object
            server_name (str, optional): The name of the server to connect to (must have atching details in the config file). Defaults to "readable_secondary".

        Raises:
            ValueError: Raised when the server name is not available in the config file
        """
        self.credentials = credentials
        
        try:
            LocalConfig.get_database(database_reference)
        except KeyError:
            raise ValueError(
                f"The server name must be one of {LocalConfig.list_available_databases()}"
            )
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
        logger.debug(
            f"Connecting to database"
        )  
        database = LocalConfig.get_database(self.database_reference)
        server = database["server"]
        database = database["database"]
        

        self.cnxn = pyodbc.connect(
            "DRIVER={ODBC Driver 17 for SQL Server};SERVER="
            + server
            + ";DATABASE="
            + database
            + ";ENCRYPT=yes;UID="
            + self.credentials["USERNAME"]
            + ";PWD="
            + self.credentials["PASSWORD"]
        )


class DataLakeAuthentication(IAuthentication):
    """
    Class for authenticating with our Azure DataLake.
    Example Use Case:
    ```
    from datascience_core.authentications import DWHAuthentication, DataLakeAuthentication, SQLServerConnection

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

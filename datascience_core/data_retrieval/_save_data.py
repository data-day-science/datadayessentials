from abc import abstractmethod
import pandas as pd
from ._base import ICSVSaver, IBlobLocation, IAuthentication, IJsonSaver, IPickleSaver
from azure.storage.filedatalake import DataLakeServiceClient
import logging
import json
import pickle
import os
import io
from typing import Union, List, Any
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BlobLocation(IBlobLocation):
    """Standardised object for identifying blobs in an Azure Datalake

    The object holds the account, container, path and filename.  The object returns a formatted path to the
    blob
    Example Use Case:
    ```python
    from datascience_core.data_retrieval import BlobLocation 

    storage_acc = 'account_name'
    container = 'container_name'
    folder = '/folder/path/'
    file = 'file.csv'
    blob = BlobLocation(storage_acc, container, folder, file)
    ```
    """

    def __init__(self, account: str, container: str, filepath: str, filename: str):
        """Instantiates a blob location object when passed an account, container, path and filename

        Args:
            account (str): Azure storage account (Datalake gen2)
            container (str): storage container
            filepath (str): path within the container
            filename (str): filename to obtain
        """
        super().__init__(account, container, filepath, filename)

    @classmethod
    def _deconstruct_abfss_path(cls, uri: str) -> dict:
        """Extracts the components of an abfss path to get the account, container, patha nd filename

        Args:
            uri (str): Single abfss path string

        Returns:
            dict: Dictionary of path components
        """
        initial_split = uri.split(sep="/")
        initial_split.remove("abfss:")
        initial_split.remove("")
        initial_split

        container = [x.split("@")[0] for x in initial_split if "@" in x][0]
        if len(container) == 0:
            raise
        account = [x.split("@")[1].split(".")[0] for x in initial_split if "@" in x][0]
        if len(account) == 0:
            raise
        path = [x for x in initial_split if "@" not in x]

        file_name = path[-1]
        path.remove(file_name)
        # file_name = file_name.split(".")[0]
        file_path = ""
        for component in path:
            file_path += component + "/"

        path_components = {
            "account": account,
            "container": container,
            "file_path": file_path[:-1],
            "file_name": file_name,
        }

        return path_components

    @classmethod
    def _deconstruct_https_path(cls, uri: str) -> dict:
        """Extracts the components of an https path to get the account, container, patha nd filename

        Args:
            uri (str): Single https path string

        Returns:
            dict: Dictionary of path components
        """
        initial_split = uri.split(sep="/")
        initial_split.remove("https:")
        initial_split.remove("")

        account = [x.split(".")[0] for x in initial_split if "." in x][0]
        if len(account) == 0:
            raise
        container = initial_split[1]
        path = [x for x in initial_split if account + ".blob.core.windows.net" not in x]
        file_name = path[-1]
        path.remove(file_name)
        path.remove(container)
        # file_name = file_name.split(".")[0]
        file_path = ""
        for component in path:
            file_path += component + "/"
        path_components = {
            "account": account,
            "container": container,
            "file_path": file_path[:-1],
            "file_name": file_name,
        }
        return path_components

    @classmethod
    def from_abfss_path(cls, uri):
        """Extracts the components of an abfss path to get the account, container, patha nd filename

        Args:
            uri (str): Single abfss path string

        Returns:
            dict: Dictionary of path components
        """
        path_components = cls._deconstruct_abfss_path(uri=uri)

        account = path_components["account"]
        container = path_components["container"]
        filepath = path_components["file_path"]
        filename = path_components["file_name"]
        # return super().__init__(
        #     account=account, container=container, filepath=filepath, filename=filename
        # )
        return cls(
            account=account, container=container, filepath=filepath, filename=filename
        )

    @classmethod
    def from_https_path(cls, uri):
        """Extracts the components of an abfss path to get the account, container, patha nd filename

        Args:
            uri (str): Single https path string

        Returns:
            dict: Dictionary of path components
        """
        path_components = cls._deconstruct_https_path(uri=uri)

        account = path_components["account"]
        container = path_components["container"]
        filepath = path_components["file_path"]
        filename = path_components["file_name"]
        # return super().__init__(
        #     account=account, container=container, filepath=filepath, filename=filename
        # )
        return cls(
            account=account, container=container, filepath=filepath, filename=filename
        )


class DataLakeCSVSaver(ICSVSaver):
    """
    Saves a pandas dataframe to an Azure Blob
    ```python
    from datascience_core.authentication import DataLakeAuthentication
    from datascience_core.data_retrieval import BlobLocation, DataLakeCSVSaver

    authentication = DataLakeAuthentication()
    storage_acc = 'account_name'
    container = 'container_name'
    folder = '/folder/path/'
    file = 'file.csv'
    blob = BlobLocation(storage_acc, container, folder, file)

    df_to_save = pd.DataFrame({'col1': [1, 2, 3], 'col2': [1, 2, 3]})
    saver = DataLakeCSVSaver(authentication)

    saver.save(blob, df_to_save)
    ```
    """

    def __init__(self, authentication: IAuthentication):
        """Create a DataLakeCSVSaver instance

        Args:
            authentication (IAuthentication): Authentication object for retrieving azure credentials (see authentications) module
        """
        self.credentials = authentication.get_azure_credentials()

    def save(
        self,
        blob_location: BlobLocation,
        df: pd.DataFrame,
    ):
        """Connect to Azure Data Lake and save the input data to a csv file

        Args:
            blob_location (BlobLocation): Location to save the input dataframe to
            df (pd.DataFrame): Pandas dataframe to save
        """

        account = blob_location.get_account()
        container = blob_location.get_container()
        directory = blob_location.get_filepath()
        file = blob_location.get_filename()

        account_url = f"https://{account}.dfs.core.windows.net/"
        datalake_service = DataLakeServiceClient(
            account_url=account_url, credential=self.credentials
        )
        file_client = datalake_service.get_file_client(
            file_system=container, file_path=os.path.join(directory, file)
        )
        logger.debug("created client")

        df_buffer = io.StringIO()
        df_buffer = df.to_csv(index=False)

        file_client.upload_data(
            df_buffer, overwrite=True, timeout=10000, chunk_size=10 * 1024 * 1024
        )
        logger.debug("created file")


class DataLakeJsonSaver(IJsonSaver):
    """
    For saving pandas dataframes as JSON files in Azure Blob storage
    Example Use Case:
    ```python
    from datascience_core.authentication import DataLakeAuthentication
    from datascience_core.data_retrieval import BlobLocation, DataLakeJsonSaver

    authentication = DataLakeAuthentication()
    storage_acc = 'account_name'
    container = 'container_name'
    folder = '/folder/path/'
    file = 'file.json'
    blob = BlobLocation(storage_acc, container, folder, file)

    df_to_save = pd.DataFrame({'col1': [1, 2, 3], 'col2': [1, 2, 3]})
    saver = DataLakeJsonSaver(authentication)

    saver.save(blob, df_to_save)
    ```
    """

    def __init__(self, authentication: IAuthentication):
        """Creates a DataLakeJsonSaver instance

        Args:
            authentication (IAuthentication): Authentication object for retrieving azure login credentials (see the authentications module)
        """
        self.credentials = authentication.get_azure_credentials()

    def save(
        self,
        blob_location: IBlobLocation,
        data: Union[pd.DataFrame, dict, pd.Series, str, List],
    ):
        """Connect to Azure Data Lake and save the input data to a json file

        Args:
            blob_location (IBlobLocation): BlobLocation containing details of the azure account, container etc.
            data (pd.DataFrame): Data to be saved as JSON in azure
        """

        account = blob_location.get_account()
        container = blob_location.get_container()
        directory = blob_location.get_filepath()
        file = blob_location.get_filename()

        account_url = f"https://{account}.dfs.core.windows.net/"
        datalake_service = DataLakeServiceClient(
            account_url=account_url, credential=self.credentials
        )
        file_client = datalake_service.get_file_client(
            file_system=container, file_path=os.path.join(directory, file)
        )
        logger.debug("created client")

        file_client.upload_data(json.dumps(data), overwrite=True)
        logger.debug("created file")


class DataLakePickleSaver(IPickleSaver):
    """
    For saving python objects to Azure DataLake as a pickle file
    """

    def __init__(self, authentication: IAuthentication):
        """Creates a DataLakePickleSaver instance

        Args:
            authentication (IAuthentication): Authentication object for retrieving azure login credentials (see the authentications module)
        """
        self.credentials = authentication.get_azure_credentials()

    def save(
        self,
        blob_location: IBlobLocation,
        data: Any,
    ):
        """Connect to Azure Data Lake and save the input data to a json file

        Args:
            blob_location (IBlobLocation): BlobLocation containing details of the azure account, container etc.
            data (pd.DataFrame): Data to be saved as JSON in azure
        """

        account = blob_location.get_account()
        container = blob_location.get_container()
        directory = blob_location.get_filepath()
        file = blob_location.get_filename()

        account_url = f"https://{account}.dfs.core.windows.net/"
        datalake_service = DataLakeServiceClient(
            account_url=account_url, credential=self.credentials
        )
        file_client = datalake_service.get_file_client(
            file_system=container, file_path=os.path.join(directory, file)
        )
        logger.debug("created client")

        file_client.upload_data(pickle.dumps(data), overwrite=True)
        logger.debug("created file")

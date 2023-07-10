from azure.storage.filedatalake import DataLakeServiceClient
from ._base import IDataLakeDirectoryDeleter
from ._save_data import BlobLocation
from ..authentications import IAuthentication
import pandas as pd


class DataLakeDirectoryDeleter(IDataLakeDirectoryDeleter):
    def __init__(self, authentication: IAuthentication):
        self.credentials = authentication.get_credentials()

    # Get the connection string from an environment variable
    def delete_directory(
        self,
        blob_location: BlobLocation,
    ):
        """
        Deletes a directory from a Data Lake Storage Gen2 account.

        Args:
            blob_location (BlobLocation): The location of the directory to delete.
        """

        account = blob_location.get_account()
        container = blob_location.get_container()
        directory = blob_location.get_filepath()

        account_url = f"https://{account}.dfs.core.windows.net/"
        datalake_service = DataLakeServiceClient(
            account_url=account_url, credential=self.credentials
        )
        directory_client = datalake_service.get_directory_client(
            container, directory
        )
        directory_client.delete_directory()
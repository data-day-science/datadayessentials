from datetime import datetime, date
from ._save_data import BlobLocation
from ._base import IURIGenerator, IBlobLocation, IAuthentication
from typing import List
from ..config import LocalConfig
from azure.storage.filedatalake import DataLakeServiceClient
from azure.storage.blob import BlobServiceClient
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class URIGenerator(IURIGenerator):
    """Creates a generator using all the files in a folder"""

    def get_uris(
        self, blob: IBlobLocation, authentication: IAuthentication
    ) -> List[IBlobLocation]:
        """Retrieve the Azure Blob URI's for all the files in the `blob` input

        Args:
            blob (IBlobLocation): Folder in blob storage
            authentication (IAuthentication): Azure authentication credentials (see authentications module)

        Returns:
            List[IBlobLocation]: List of blobs that are all the files in the input folder
        """
        blob_service = BlobServiceClient(
            account_url=blob.get_account_url(), credential=authentication
        )
        container_client = blob_service.get_container_client(blob.container)
        files = container_client.list_blobs(blob.filepath)
        blobs = []
        for blob_properties in files:
            filename = blob_properties["name"].split("/")[-1]
            new_blob = BlobLocation(
                blob.account, blob.container, blob.filepath, filename
            )
            blobs.append(new_blob)
        return blobs


class CreditDataURIGenerator(IURIGenerator):
    """
    Generate all the URI's needed to load in all the credit data between two dates. This involves converting the input dates into epoch numbers and using the epoch numbers to get to the filenames.
    """

    def __init__(self, start_date: datetime, end_date: datetime):
        """Initialise CreditDataURIGenerator with the date range for the credit URI's to output in the `get_uris` function. The dates are validated to ensure they are not invalid.

        Args:
            start_date (datetime): Start date for the credit data URI's
            end_date (datetime): End date for the credit data URI's

        Raises:
            ValueError: Raised if the end date is greater than the current date
        """
        self.start_date = start_date
        self.end_date = end_date
        self.start_epoch = self._epoch_from_date(self.start_date)
        self.end_epoch = self._epoch_from_date(self.end_date)
        if self.end_date > datetime.now():
            raise ValueError("End date cannot be past the present date")

    def _epoch_from_date(self, date: date) -> int:
        """Converts a date into an epoch number

        Args:
            date (date): Input date or datetime

        Returns:
            int: Epoch number
        """
        return (12 * date.year) + date.month - 22801

    def get_uris(self) -> List[BlobLocation]:
        """Given the parent_URI passed in init, generate a list of epoch files that correspond to the start/end times

        Returns:
            List[BlobLocation]: List of blob locations
        """
        uris = []
        for epoch in range(self.start_epoch, self.end_epoch + 1):

            epoch_path = "epoch_" + str(epoch) + ".csv"
            cra_data = LocalConfig.get_data_lake_folder(
                "cra_data", use_current_environment=False
            )
            blob = BlobLocation(
                cra_data["data_lake"],
                cra_data["container"],
                cra_data["path"],
                epoch_path,
            )
            uris.append(blob)
        return uris

from datetime import datetime, date
from ._save_data import BlobLocation
from ._base import IURIGenerator, IBlobLocation, IAuthentication
from typing import List
from ..config import Config
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

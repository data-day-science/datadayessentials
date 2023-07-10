from datascience_core.authentications import DataLakeAuthentication 
from unittest import mock
from datascience_core.config import LocalConfig
from datascience_core.data_retrieval import BlobLocation
from datascience_core.data_retrieval._delete_data import DataLakeDirectoryDeleter

class TestDataLakeDirectoryDeleter:
    @mock.patch("azure.storage.filedatalake.DataLakeDirectoryClient.delete_directory")
    def test_save(self, mock_delete_directory):

        mock_authentication = DataLakeAuthentication()
        mock_authentication.get_azure_credentials = mock.MagicMock()

        blob_location = BlobLocation(
            account=LocalConfig.get_data_lake(),
            container="test",
            filename="",
            filepath="folder",
        )
        data_deleter = DataLakeDirectoryDeleter(authentication=mock_authentication)

        data_deleter.delete_directory(blob_location)
        assert mock_delete_directory.called
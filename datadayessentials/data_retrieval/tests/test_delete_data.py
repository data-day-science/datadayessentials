from datadayessentials.authentications import DataLakeAuthentication 
from unittest import mock
from datadayessentials.config import Config
from datadayessentials.data_retrieval import BlobLocation
from datadayessentials.data_retrieval._delete_data import DataLakeDirectoryDeleter

class TestDataLakeDirectoryDeleter:
    @mock.patch("azure.storage.filedatalake.DataLakeDirectoryClient.delete_directory")
    def test_save(self, mock_delete_directory):

        mock_authentication = DataLakeAuthentication()
        mock_authentication.get_azure_credentials = mock.MagicMock()

        blob_location = BlobLocation(
            account=Config().get_environment_variable("data_lake"),
            container="test",
            filename="",
            filepath="folder",
        )
        data_deleter = DataLakeDirectoryDeleter(authentication=mock_authentication)

        data_deleter.delete_directory(blob_location)
        assert mock_delete_directory.called
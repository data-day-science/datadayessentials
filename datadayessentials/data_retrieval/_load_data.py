from io import StringIO, BytesIO, BufferedWriter
from typing import List, Any, Union
from urllib.parse import uses_fragment
from ..authentications import (
    DataLakeAuthentication,
    DatabaseAuthentication,
    SQLServerConnection,
)
from ._base import (
    ITableLoader,
    ICSVLoader,
    IURIGenerator,
    IAuthentication,
    IDataFrameTap,
    IDataFrameLoader,
    IDataFrameCacher,
    IAzureBlobLoader,
    IBlobLocation,
    IJsonLoader,
    IPickleLoader,
    IParquetLoader,
)

from ._validate_data import DataFrameValidator
from ._uri_generators import URIGenerator
import pandas as pd
import datetime
import logging
from ..config._config import Config
from azure.storage.filedatalake import (
    DataLakeServiceClient,
)
from azure.storage.blob import BlobServiceClient, BlobClient
from datetime import datetime
from ._save_data import BlobLocation
import os
from pathlib import Path
import re
import hashlib, base64
from azure.core.exceptions import ResourceNotFoundError
import json, pickle

from azure.core.exceptions import HttpResponseError


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DataCacher:
    """A class for caching dataframes retrieved from the datawarehouse or cloud storage.
    The dataframes are saved to the cache directory specified in the config yaml. This directory is always in the user's home path.
    """

    def __init__(
        self, file: str, last_modified: datetime = None, sub_folder: str = None
    ):
        """Initialises the Cacher with file name to be checked, if the last_modified parameter is passed then include this in the filename

        Args:
            file (str): file name to check against the cache
        """

        file = re.sub(
            r"[^\w]", "_", file
        )  # Replace all non-alphanumeric characters with '_'

        if last_modified:
            self.file = file + "__" + last_modified.strftime("%Y_%m_%d")
        else:
            self.file = file

        home = str(Path.home())
        if sub_folder:
            print(f"subfolder is {sub_folder}")
            self.file_dir = os.path.join(
                home,
                Config().get_environment_variable(variable_name="local_cache_dir"),
                sub_folder,
            )
        else:
            print(f"subfolder is None")
            self.file_dir = os.path.join(
                home, Config().get_environment_variable(variable_name="local_cache_dir")
            )
        self.file_path = os.path.join(self.file_dir, self.file)
        if not os.path.exists(self.file_dir):
            os.makedirs(self.file_dir)

    def is_file_in_cache(self) -> bool:
        """Check if the file is in the cache directory.

        Returns:
            bool: Does the file exist in the cache
        """
        print(f"Searching cache for {self.file_path}")
        return os.path.exists(self.file_path)

    def is_dir_in_cache(self) -> str:
        """Check if the directory is in the cache.

        Returns:
            bool: Does the directory exist in the cache
        """

        return os.path.exists(self.file_dir)

    def get_dir_from_cache(self) -> str:
        """Retrieve the directory from the cache.

        Returns:
            str: cached directory
        """
        return self.file_dir

    def get_df_from_cache(self) -> pd.DataFrame:
        """Retrieve the dataframe from the cache.

        Returns:
            pd.DataFrame: cached file
        """
        try:
            df = pd.read_csv(self.file_path, index_col="index")
        except FileNotFoundError:
            df = pd.DataFrame()
        return df

    def get_pickle_from_cache(self) -> Any:
        """Retrieve the pickled file from the cache.

        Returns:
            Any: any python object
        """
        try:
            with open(self.file_path, "rb") as f:
                obj = pickle.loads(f.read())
        except FileNotFoundError:
            obj = ""
        return obj

    def get_json_from_cache(self) -> dict:
        """Retrieve the json from the cache.

        Returns:
            dict: Json dictionary
        """
        try:
            with open(self.file_path, "r") as f:
                json_out = json.loads(f.read())
        except FileNotFoundError:
            json_out = {}
        return json_out

    def get_pq_from_cache(self) -> pd.DataFrame:
        """Retrieve the parquet file from the cache.

        Returns:
            pd.DataFrame: cached file
        """
        try:
            df = pd.read_parquet(self.file_path)
        except FileNotFoundError:
            df = pd.DataFrame()
        return df

    def save_df_to_cache(self, df: pd.DataFrame, size: int = 0):
        """Save the dataframe to the cache.

        Args:
            df (pd.DataFrame): dataframe to store
        """
        if not df.empty:
            df.to_csv(self.file_path, index_label="index")

    def save_pq_to_cache(self, df: pd.DataFrame):
        """Save the dataframe to the cache.

        Args:
            df (pd.DataFrame): dataframe to store
        """
        if not df.empty:
            df.to_parquet(self.file_path)

    def save_pickle_to_cache(self, obj: Any):
        """Save the python object as a pickle file in the cache.

        Args:
            obj (Any): Python object for pickling
        """
        with open(self.file_path, "wb") as f:
            pickle.dump(obj, f)

    def save_json_to_cache(self, json_dict: Union[dict, List]):
        """Save the python dictionary as a json in the cache.

        Args:
            json (Union[dict, List]): Json serializable python object for saving
        """
        try:
            with open(self.file_path, "w") as f:
                json.dump(json_dict, f)
        except TypeError as e:
            raise TypeError(
                "The object passed for saving to the cache as JSON is not JSON serializable."
            )


class TableLoader(ITableLoader):
    """A class for connecting to an SQL Server and running SQL
    This can either be used directly with an SQL statement as below:
    ```python
    from datadayessentials.authentication import DWHAuthentication
    from datadayessentials.data_retrieval import TableLoader

    authentication = DWHAuthentication()
    SQL_statement = 'SELECT * FROM Allocations'

    loader = TableLoader(authentication, SQL_statement)
    df = loader.load()
    ```
    Or the QueryFactory can be used to parameterise an SQL statement (this is better for reusability, input validation can be performed on these queries)
    """

    def __init__(
        self,
        sql_statement: str,
        server_name: str = "readable-secondary",
        authentication: IAuthentication = None,
        use_cache: bool = True,
    ):
        """Creates a TableLoader instance

        Args:
            authentication (DWHAuthentication): DWHAuthentication instance for retrieving login credentials. Overides the default authentication method.
            sql_statement (str): SQL Statement to run against the SQL server
            server_name (str, optional): Name of the server, used to retrieve database details from the config file. Defaults to "readable-secondary".
            use_cache (bool, optional): If true, cache the results by the SQL statement used. If false then run against the SQL Server and cache the results after for future use. Defaults to True.
        """
        logger.debug(f"Creating TableLoader Class")
        self.use_cache = use_cache
        if not authentication:
            authentication = DatabaseAuthentication(database_reference=server_name)
        self.authetication = authentication
        self.sql_statement = sql_statement
        self.server_name = server_name
        credentials = authentication.get_credentials()
        self.connection = SQLServerConnection(
            credentials, database_reference=server_name
        )

    def load(self) -> pd.DataFrame:
        """Run the SQL statement against the SQL Server

        Returns:
            pd.DataFrame: Results from the SQL server stored in a pandas `pd.DataFrame`
        """
        logger.debug(f"Running sql to load table")
        # Check if the blob has already been cached
        hashed_sql = base64.urlsafe_b64encode(
            hashlib.md5(self.sql_statement.encode("utf-8")).digest()
        ).decode("ascii")
        cacher = DataCacher(hashed_sql)
        if cacher.is_file_in_cache() and self.use_cache:
            return cacher.get_df_from_cache()
        else:
            df = self.connection.run_sql(self.sql_statement)
            cacher.save_df_to_cache(df)
        return df


class DataLakeCSVLoader(ICSVLoader):
    """Retrieves CSV's from the data lake based on either a single blob path (using the load method) or multiple blob paths (then concatenates them, using the load_from_uri_generator).
    It will cache the results locally by default, to disable this behaviour, set use_cache = False.
    It will also ignore if a blob is not available when using the load_from_uri_generator method. To disable this set errors='raise'.
    Example Use Case:
        ```python
        from datadayessentials.authentications import DataLakeAuthentication
        from datadayessentials.data_retrieval import BlobLocation, DataLakeCSVLoader

        authentication = DataLakeAuthentication()
        # Details of the blob
        storage_acc = 'account_name'
        container = 'container_name'
        folder = '/folder/path/'
        file = 'file.csv'
        blob = BlobLocation(storage_acc, container, folder, file)

        csv_loader = DataLakeCSVLoader(authentication)
        df = csv_loader.load(blob)
        ```
    """

    def __init__(
        self,
        authentication: IAuthentication = None,
        use_cache=True,
        errors: str = "ignore",
    ):
        """Create a `DataLakeCSVLoader` instance, for loading CSV's stored in azure into a pandas dataframe

        Args:
            authentication (IAuthentication): Authentication instance for authenticating with azure, could be `DataLakeAuthentication` from the authentications module. Overrides the default authentication method.
            use_cache (bool, optional): When true, cache the results locally so that future invocations dont need to re-download, set to False to overwrite the cache. Defaults to True.
            errors (str, optional): Set to either 'ignore' or 'raise', when set to raise. Only relevant for when using the `DataLakeCSVLoader.load_from_uri_generator` function, if one of
                the CSV files cannot be found then no error will be raised if this parameter is set to 'ignore'. Defaults to "ignore".
        """
        self.use_cache = use_cache
        self.errors = errors
        if not authentication:
            authentication = DataLakeAuthentication()
        self.credential = authentication.get_azure_credentials()

    def load_from_uri_generator(self, uris: IURIGenerator) -> pd.DataFrame:
        """Connect to Azure Data Lake and retrieve the csvs provided by the IURIGenerator instance and concatenate them together

        Args:
            uris (IURIGenerator): Class instance of IURIGenerator that is an iterator over multiple URI's

        Raises:
            ResourceNotFoundError: Error is raised if a URI cannot be found. Can be suppressed by setting errors='ignore' when instantiating the class

        Returns:
            pd.DataFrame: Dataframe with all the CSV's concatentated
        """
        dfs = []
        for i, uri in enumerate(uris.get_uris()):
            logger.debug(f"Loading file {i} from ure {uri}")
            try:
                df = self.load(uri)
                dfs.append(df)
            except ResourceNotFoundError as e:
                if self.errors == "ignore":
                    logger.warning(
                        f"Blob at uri {uri} not found, continuing to load the rest."
                    )
                    continue
                else:
                    raise e

        combined_df = pd.concat(dfs)
        return combined_df

    def load(self, blob: BlobLocation, **kwargs) -> pd.DataFrame:
        """Load a CSV file stored in azure blob storage into a pandas dataframe

        Args:
            blob (BlobLocation): Blob to load
            kwargs: Additional arguments to pass to the `pandas.read_csv` function

        Returns:
            pd.DataFrame: Dataframe loaded from the blob CSV
        """
        if not hasattr(self, "datalake_service"):
            account_url = f"https://{blob.get_account()}.dfs.core.windows.net/"
            self.datalake_service = DataLakeServiceClient(
                account_url=account_url, credential=self.credential
            )

        file_client = self.datalake_service.get_file_client(
            blob.get_container(), blob.get_path_in_container()
        )
        properties = file_client.get_file_properties()

        cacher = DataCacher(str(blob) + ".csv", properties.last_modified)
        # Check if the blob has already been cached
        if cacher.is_file_in_cache() and self.use_cache:
            return cacher.get_df_from_cache()
        else:
            download = file_client.download_file()
            buffer = BytesIO(download.read())
            buffer.seek(0)
            df = pd.read_csv(buffer, **kwargs)
            cacher.save_df_to_cache(df)
            return df


class DataFrameTap(IDataFrameTap):
    """
    Class for loading, validating and casting data. It uses any loading class that inherits from IDataFrameLoader. The data_schema is used to remove any invalid values from the data (invalid categorical values or numerical values out of range). It then casts the data to the target schema.

    This loads, validates and then casts data. THe reason for this seperation is the following:

    1. Source data tends to have a schema, if there are values that dont conform to theis schema then they should be removed at this point.
    2. The schema that you want your data in is often different to the source schema (integers might be originally loaded as string type)

    In the below example the source and target schemas are loaded using the SchemaFetcher, which looks for a schema of the format (name.json) stored in the schemas folder in the data_retrieval module. Please see the SchemaFetcher class for the format schemas should follow.
    ```python
    from datadayessentials.data_retrieval import DataFrameTap, TableLoader, SchemaFetcher
    from datadayessentials.data_retrieval import TableLoader

    authentication = DWHAuthentication() # optional to pass in authentication.  Table loader will use default authentication if not passed in
    sql_statement = "SELECT * FROM table_1"
    loader = TableLoader(authentication, sql_statement)

    source_schema = SchemaFetcher('source_schema')
    target_schema = SchemaFetcher('target_schema')

    data_tap = DataFrameTap(authentication, source_schema, target_schema)
    casted_data = data_tap.run()
    ```
    """

    def __init__(
        self, data: IDataFrameLoader, data_schema: dict, target_schema: dict
    ) -> None:
        """Creates a DataFrameTap instance

        Args:
            data (IDataFrameLoader): Loading class for retrieving data (from locally, cloud or DataWarehouse locations)
            data_schema (dict): Schema to use for validating the loaded data
            target_schema (dict): Schema to use for casting the data once loaded
        """
        self.data_loader = data
        self.validator: DataFrameValidator = DataFrameValidator(data_schema)

    def run(self) -> pd.DataFrame:
        """Runs the Tap (load, validate, cast) and returns a pandas dataframe

        Returns:
            pd.DataFrame: Output casted data
        """
        data = self.data_loader.load()
        validated_data = self.validator.validate(data)

        return validated_data


class AzureBlobLoader(IAzureBlobLoader):
    """
    Responsible for downloading an Azure Blob, either a file (using the pull function) or a folder (using the pull_folder function).
    """

    def __init__(self, authentication: IAuthentication):
        """Creates an AzureBlobLoader class

        Args:
            authentication (IAuthentication): Authentication instance for retrieving login information from Azure (see authentications module)
        """
        self.authentication = authentication

    def _initialise_client(self, blob: IBlobLocation):
        """Initialize a blob client for a specific storage account

        Args:
            blob (IBlobLocation): BlobLocation containing the relevant storage account
        """
        if not hasattr(self, "blob_service"):
            self.blob_service = BlobServiceClient(
                account_url=blob.get_account_url(), credential=self.authentication
            )

    def pull(self, blob: IBlobLocation, save_file_as: str, use_cached: bool = True):
        """Download file from azure blob storage

        Args:
            blob (IBlobLocation): BlobLocation of the file to download
            save_file_as (str): Local path for saving the downloaded file
            use_cached (bool, optional): If the file already exists, then dont redownload if set to True. Defaults to True.
        """
        self._initialise_client(blob)
        self.blob_client = self.blob_service.get_blob_client(
            container=blob.container, blob=blob.get_path_in_container()
        )
        # Check Cache...
        if os.path.exists(save_file_as) and use_cached:
            return

        with open(save_file_as, "wb") as my_blob:
            download_stream = self.blob_client.download_blob()
            my_blob.write(download_stream.readall())

    def pull_folder(
        self, folder_blob: IBlobLocation, save_folder: str, use_cached=True
    ):
        """Download an entire folder from azure blob storage

        Args:
            folder_blob (IBlobLocation): BlobLocation containing the location of the folder to download
            save_folder (str): Local path for saving the folder
            use_cached (bool, optional): When True and the file has already been downloaded, then dont redownload. Defaults to True.
        """
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        self._initialise_client(folder_blob)
        uri_generator = URIGenerator()
        file_blobs = uri_generator.get_uris(folder_blob, self.authentication)
        for file_blob in file_blobs:
            self.pull(
                file_blob,
                os.path.join(save_folder, file_blob.filename),
                use_cached=use_cached,
            )


class DataLakePickleLoader(IPickleLoader):
    """
    Loads a pickle file stored in Azure blob (data lake) into a python object
    """

    def __init__(self, authentication: IAuthentication = None, use_cache=True):
        """Instantiate a DataLakePickleLoader

        Args:
            authentication (IAuthentication): Azure authentication credentials (see authentications module)
            use_cache (bool, optional): Option to load the file from the cache if available. Defaults to True.
        """
        self.use_cache = use_cache
        if not authentication:
            authentication = DataLakeAuthentication()
        self.credential = authentication.get_azure_credentials()

    def load(self, blob: BlobLocation) -> Any:
        """Loads the file from the blob (or cache)

        Args:
            blob (BlobLocation): The blob where the pickle file is

        Returns:
            Any: The python object inside the pickle file
        """
        self.account_url = f"https://{blob.get_account()}.dfs.core.windows.net/"
        self.datalake_service = DataLakeServiceClient(
            account_url=self.account_url, credential=self.credential
        )
        # Check if the blob has already been cached
        file_client = self.datalake_service.get_file_client(
            blob.get_container(), blob.get_path_in_container()
        )
        properties = file_client.get_file_properties()
        cacher = DataCacher(str(blob) + ".pkl", properties.last_modified)
        if cacher.is_file_in_cache() and self.use_cache:
            return cacher.get_pickle_from_cache()
        else:
            download = file_client.download_file()

            buffer = BytesIO(download.readall())
            buffer.seek(0)
            obj = pickle.load(buffer)
            cacher.save_pickle_to_cache(obj)
            return obj


class DataLakeJsonLoader(IJsonLoader):
    """
    Loads a json file from the Azure DataLake, using the cache by default.
    """

    def __init__(self, authentication: IAuthentication, use_cache=True):
        """Instantiate a DataLakeJsonLoader

        Args:
            authentication (IAuthentication): Authentication object for azure (see authentications module)
            use_cache (bool, optional): Option to use the object saved in the cache. Defaults to True.
        """
        self.use_cache = use_cache
        self.credential = authentication.get_azure_credentials()

    def load(self, blob: BlobLocation) -> dict:
        """Loads the file from azure or from the cache

        Args:
            blob (BlobLocation): Location in azure of the json to load

        Returns:
            dict: Python dictionary containing the loaded json
        """
        self.account_url = f"https://{blob.get_account()}.dfs.core.windows.net/"
        self.datalake_service = DataLakeServiceClient(
            account_url=self.account_url, credential=self.credential
        )
        # Check if the blob has already been cached
        file_client = self.datalake_service.get_file_client(
            blob.get_container(), blob.get_path_in_container()
        )
        properties = file_client.get_file_properties()
        cacher = DataCacher(str(blob) + ".json", properties.last_modified)
        if cacher.is_file_in_cache() and self.use_cache:
            return cacher.get_json_from_cache()
        else:
            download = file_client.download_file()
            buffer = BytesIO(download.read())
            buffer.seek(0)
            obj = json.load(buffer)
            cacher.save_json_to_cache(obj)
            return obj


class DataLakeParquetLoader(IParquetLoader):
    """
    Loads a parquet file from the Azure DataLake, using the cache by default.
    """

    def __init__(self, authentication: IAuthentication = None, use_cache=True):
        """Instantiate a DataLakeParquetLoader

        Args:
            authentication (IAuthentication): Authentication object for azure (see authentications module)
            use_cache (bool, optional): Option to use the object saved in the cache. Defaults to True.
        """
        self.use_cache = use_cache
        if not authentication:
            authentication = DataLakeAuthentication()
        self.credential = authentication.get_azure_credentials()

    def _get_file_client(self, blob: BlobLocation):
        """
        Helper function to get the file client for a given blob location
        """
        file_client = self.datalake_service.get_file_client(
            blob.get_container(), blob.get_path_in_container()
        )
        return file_client

    def _get_available_files(self, blob: BlobLocation) -> List[str]:
        """Get a list of available files from the specified BlobLocation.

        Args:
            blob (BlobLocation): The BlobLocation representing the location of the files.

        Returns:
            List[str]: A list of file paths available at the specified BlobLocation.
                    Returns an empty list if the location is not found or an error occurs.

        Raises:
            None

        Note:
            This function retrieves the available files at the specified BlobLocation
            by connecting to the corresponding Azure Data Lake Storage file system.
            It handles ResourceNotFoundError and HttpResponseError exceptions and
            prints informative messages in case of errors.
        """
        download_path = blob.get_filepath().replace("\\", "/")
        print(f"downloading files from {download_path}")
        try:
            file_system_client = self.datalake_service.get_file_system_client(
                blob.get_container()
            )
            available_files = file_system_client.get_paths(path=blob.get_filepath())
            return available_files
        except ResourceNotFoundError as e:
            print(f"File {download_path} not found. Skipping")
            return []
        except HttpResponseError as e:
            print(f"Error while getting files from {download_path}: {e.message}")
            return []

    def _load_file(
        self, blob: BlobLocation, cache_subfolder: str = None
    ) -> pd.DataFrame:
        """
        Load a file from Azure Blob Storage, optionally caching it for future use.

        Args:
            blob (BlobLocation): The BlobLocation representing the file to load.
            cache_subfolder (str, optional): A subfolder within the cache directory. Default is None.

        Returns:
            pd.DataFrame: The loaded data as a Pandas DataFrame.

        Raises:
            FileNotFoundError: If the specified blob does not exist in Azure Blob Storage.
            ValueError: If the specified blob is not a Parquet file.

        Note:
            This function loads a file from Azure Blob Storage using the provided BlobLocation.
            It checks if the file exists and if it's a Parquet file (content type).
            If the file is already cached and caching is enabled, it retrieves the data from the cache.
            Otherwise, it downloads the file, reads it as a Parquet file, and optionally caches it for future use.

        Example:
            To load a file from Azure Blob Storage and cache it in a subfolder:
            >>> blob_location = BlobLocation("container_name", "folder/file.parquet")
            >>> data = _load_file(blob_location, cache_subfolder="my_cache")
        """

        file_client = self._get_file_client(blob)

        if not file_client.exists():
            raise FileNotFoundError(f"File {blob} not found in azure.")
        properties = file_client.get_file_properties()
        # if the file type isnt parquet then raise a value error
        if properties.content_settings.content_type != "application/octet-stream":
            raise ValueError(f"File {blob} is not a parquet file.")

        cacher = DataCacher(
            str(blob), properties.last_modified, sub_folder=cache_subfolder
        )
        print(cacher)
        if cacher.is_file_in_cache() and self.use_cache:
            print(f"loading {blob.get_filepath()} from cache")
            return cacher.get_pq_from_cache()
        else:
            download = file_client.download_file()
            buffer = BytesIO(download.read())
            buffer.seek(0)
            df = pd.read_parquet(buffer)
            cacher.save_pq_to_cache(df)
            return df

    def _load_folder(self, blob: BlobLocation) -> pd.DataFrame:
        """
        Load all Parquet files within a folder in Azure Blob Storage and combine them into a single DataFrame.

        Args:
            blob (BlobLocation): The BlobLocation representing the folder containing Parquet files.

        Returns:
            pd.DataFrame: A DataFrame containing the concatenated data from all Parquet files in the folder.

        Note:
            This function retrieves a list of available files within the specified folder in Azure Blob Storage
            using the provided BlobLocation. It then iterates through the files, loading each Parquet file
            using the `_load_file` method and combining them into a single DataFrame.
            If no Parquet files are found, an empty DataFrame is returned.

        Example:
            To load all Parquet files from a folder in Azure Blob Storage:
            >>> folder_location = BlobLocation("account_name","container_name", "folder/")
            >>> data_frame = _load_folder(folder_location)
        """

        available_files = self._get_available_files(blob)

        dfs = []
        for file in available_files:
            print(f"Loading file {file.name}")
            if file.name.endswith(".parquet"):
                try:
                    individual_file_blob = BlobLocation(
                        blob.get_account(),
                        blob.get_container(),
                        blob.get_filepath(),
                        str(file.name).split("/")[-1],
                    )
                    df = self._load_file(
                        individual_file_blob,
                        cache_subfolder=individual_file_blob.get_filepath(),
                    )
                    dfs.append(df)
                except Exception as e:
                    print(f"Error loading file {file.name}: {str(e)}")
        if dfs:
            combined_df = pd.concat(dfs)
        else:
            combined_df = pd.DataFrame()  # Create an empty DataFrame
        return combined_df

    def load(self, blob: BlobLocation) -> pd.DataFrame:
        """
        Load data from Azure Blob Storage based on the provided BlobLocation.

        Args:
            blob (BlobLocation): The BlobLocation representing the data to be loaded.

        Returns:
            pd.DataFrame: A Pandas DataFrame containing the loaded data.

        Note:
            This function connects to Azure Blob Storage using the provided BlobLocation
            and loads data based on the type of BlobLocation:

            - If the `filename` attribute of the BlobLocation is set to None, it calls `_load_folder`
            to load all Parquet files within a folder in Azure Blob Storage and combines them into a single DataFrame.

            - If the `filename` attribute is not None, it calls `_load_file` to load a specific Parquet file.

            The function returns a Pandas DataFrame containing the loaded data.

        Example:
            To load data from Azure Blob Storage:

            1. Load a specific Parquet file:
            >>> blob_location = BlobLocation("container_name", "folder/file.parquet")
            >>> data_frame = load(blob_location)

            2. Load all Parquet files from a folder:
            >>> folder_location = BlobLocation("container_name", "folder/")
            >>> data_frame = load(folder_location)
        """

        self.account_url = f"https://{blob.get_account()}.dfs.core.windows.net/"
        self.datalake_service = DataLakeServiceClient(
            account_url=self.account_url, credential=self.credential
        )
        # check if the filename element of blob is set to None, if so then call _load_file and pass the blob.  If not then call _load_folder and pass the blob
        if blob.filename == None:
            return self._load_folder(blob)
        else:
            return self._load_file(blob)

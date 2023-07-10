"""
This module contains functions for retrieving, saving and validating data. This includes the following functionalities.

Loading data:
- Running queries against a SQL Server and loading the data into pandas (TableLoader)
- Loading pandas DataFrames from CSV files in Azure (DataLakeCSVLoader)
- Loading credit data payloads from Azure (CreditDataLoader)
- Loading datasets defined in MLStudio (ProjectDatasetManager)
- Loading, validating and casting data according to a schema (DataFrameTap)
- Downloading a file or folder from blob storage (AzureBlobLoader)

Saving data:
- Saving pandas dataframes as CSV's in Azure (DataLakeCSVSaver)
- Saving pandas dataframes as JSON in Azure (DataLakeJsonSaver)

Other:
- Formatting queries with parameters (QueryFactory)
- Representing azure blob locations (BlobLocation)
- Retrieving data schemas (SchemaFetcher)
- Validating data (replacing out of schema values) (DataFrameValidator)
"""
from ._validate_data import DataFrameValidator
from ._load_data import (
    TableLoader,
    DataLakeCSVLoader,
    CreditDataLoader,
    DataFrameTap,
    AzureBlobLoader,
)

from ._project_dataset_manager import (
    ProjectDatasetManager,
    DatalakeProjectAssetsHelper,
    MLStudioProjectDatasetsHelper,
)
from ._sql_queries import QueryFactory
from ._save_data import DataLakeCSVSaver, BlobLocation, DataLakeJsonSaver
from ._base import IURIGenerator, ICSVSaver, IBlobLocation, ICSVLoader, IProjectDataset
from ._schema_fetcher import SchemaFetcher
from ._delete_data import DataLakeDirectoryDeleter


__all__ = [
    DataFrameTap,
    TableLoader,
    QueryFactory,
    DataLakeCSVLoader,
    ProjectDatasetManager,
    DataLakeCSVSaver,
    DataLakeJsonSaver,
    BlobLocation,
    IBlobLocation,
    ICSVSaver,
    CreditDataLoader,
    SchemaFetcher,
    DatalakeProjectAssetsHelper,
    MLStudioProjectDatasetsHelper,
    DataFrameValidator,
    IURIGenerator,
    ICSVLoader,
    IProjectDataset,
    AzureBlobLoader,
    DataLakeDirectoryDeleter
]

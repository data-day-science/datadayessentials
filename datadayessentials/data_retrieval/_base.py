from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Any
import pandas as pd
from ..authentications import IAuthentication
import os
from pathlib import Path


class ISchemaFetcher(ABC):
    @abstractmethod
    def get_schema(self, name: str, model_version: str) -> dict:
        pass

    @abstractmethod
    def add_schema(self, schema_name: str, schema: dict) -> None:
        pass


class IBlobLocation(ABC):
    @abstractmethod
    def __init__(self, account, container, filepath, filename):
        self.account = account
        self.container = container
        self.filename = filename
        self.filepath = filepath

    def get_container(self) -> str:
        return self.container

    def get_filepath(self) -> str:
        return self.filepath

    def get_account(self) -> str:
        return self.account

    def get_filename(self) -> str:
        return self.filename

    def get_path_in_container(self) -> str:
        return os.path.join(self.filepath, self.filename)

    def get_account_url(self) -> str:
        return f"https://{self.account}.blob.core.windows.net"

    def __eq__(self, __o: object) -> bool:
        return (
            (self.container == __o.container)
            and (self.account == __o.account)
            and (self.filename == __o.filename)
            and (self.filepath == __o.filepath)
        )

    def __str__(self) -> str:
        return f"https://{self.account}.blob.core.windows.net/{self.container}/{self.filepath}/{self.filename}"

    def __repr__(self) -> str:
        return self.__str__()


class ISQLQueryFormatter(ABC):
    @abstractmethod
    def _format_query(self) -> str:
        pass

    @abstractmethod
    def get_query(self) -> str:
        pass

    @abstractmethod
    def _load_query(self, query_name: str) -> str:
        pass


class IDataFrameLoader(ABC):
    @abstractmethod
    def load(self, use_cache: bool = True) -> pd.DataFrame:
        pass


class IDataFrameCacher(ABC):
    @abstractmethod
    def is_file_in_cache(self) -> bool:
        pass

    @abstractmethod
    def get_df_from_cache(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def save_df_to_cache(self, df: pd.DataFrame):
        pass


class IDataFrameSaver(ABC):
    def __init__(self, authetication: IAuthentication):
        pass

    @abstractmethod
    def save(self, filepath: str, data: pd.DataFrame) -> str:
        """
        return folder path/uri
        """
        pass


class ITableLoader(IDataFrameLoader):
    def __init__(self, authentication: IAuthentication, sql_query: str):
        pass

    @abstractmethod
    def load(self) -> pd.DataFrame:
        pass


class IURIGenerator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_uris(self) -> List[IBlobLocation]:
        pass


class ICSVLoader(IDataFrameLoader):
    def __init__(self, authentication: IAuthentication):
        pass

    @abstractmethod
    def load(self) -> pd.DataFrame:
        pass


class IJsonLoader(IDataFrameLoader):
    def __init__(self, authentication: IAuthentication):
        pass

    @abstractmethod
    def load(self) -> dict:
        pass


class IPickleLoader(IDataFrameLoader):
    def __init__(self, authentication: IAuthentication):
        pass

    @abstractmethod
    def load(self) -> Any:
        pass


class IParquetLoader(IDataFrameLoader):
    def __init__(self, authentication: IAuthentication):
        pass

    @abstractmethod
    def load(self) -> pd.DataFrame:
        pass


class ICSVSaver(IDataFrameSaver):
    def __init__(self, authentication: IAuthentication):
        pass

    @abstractmethod
    def save(self, filepath: str, data: pd.DataFrame) -> pd.DataFrame:
        pass


class IJsonSaver(ABC):
    @abstractmethod
    def save(self, filepath: str, data: Any):
        pass


class IPickleSaver(ABC):
    @abstractmethod
    def save(self, filepath: str, data: Any):
        pass


class IDataFrameValidator(ABC):
    def __init__(self, schema: dict):
        pass

    @abstractmethod
    def validate(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


class IDataFrameTap(ABC):
    def __init__(self, data: IDataFrameLoader, data_schema: dict) -> None:
        super().__init__()

    @abstractmethod
    def run(self) -> pd.DataFrame:
        pass


class IAzureBlobLoader(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def initialise_client(self):
        pass

    @abstractmethod
    def pull(self, blob: IBlobLocation, save_file_as: str, use_cached: bool = True):
        pass

    @abstractmethod
    def pull_folder(
        self, blob: IBlobLocation, save_to_folder: str, use_cached: bool = True
    ):
        pass


class IProjectDataset(ABC):
    def __init__(self, project: str):
        self.project = project

    def load_data(self, quality: str, version: str) -> pd.DataFrame:
        pass


class IRegisterProjectDataset(ABC):
    def __init__(self, blob: IBlobLocation):
        self.blob = blob

    def create_or_update_dataset(self, data: pd.DataFrame):
        pass


class IDataLakeDirectoryDeleter(ABC):
    def __init__(self, authentication: IAuthentication):
        pass

    def delete_directory(self, directory: IBlobLocation):
        pass

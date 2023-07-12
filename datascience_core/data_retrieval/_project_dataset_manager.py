from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication, InteractiveLoginAuthentication

from ._load_data import DataLakeCSVLoader, DataLakePickleLoader, DataLakeJsonLoader
from ._base import IBlobLocation, IURIGenerator, ICSVLoader
from ..config import LocalConfig
from azure.storage.filedatalake import DataLakeServiceClient
from ._save_data import (
    BlobLocation,
    DataLakeCSVSaver,
    DataLakeJsonSaver,
    DataLakePickleSaver,
)
from ..authentications import IAuthentication, DataLakeAuthentication
from ._base import IProjectDataset

from typing import List, Optional, Any
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from typing import Union, Dict

import pandas as pd
import os
import numpy as np
from azure.ai.ml import MLClient

import uuid
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)



class DatalakeProjectAssetsHelper:
    """A class for interacting with resources in the Data lake.  This class is a helper for
    the ProjectDatasetManager and is not intended to be used as a standalone class
    """

    def __init__(self, credentials: DataLakeAuthentication, project: str):
        """Initialises the DatalakeProjectAssetsHelper with a data lake authenticator and
            cf247 project in the form datascience_<project>

        Example:
            auth = DataLakeAuthentication()
            dl_asset_helper = DatalakeProjectAssetsHelper(auth,"datascience_247money_scorecard")
        Args:
            credentials (DataLakeAuthentication): DWHAuthentication instance for retrieving login credentials
            project (str): name of the data science project
        """

        self.credential = credentials
        dataset_manager_env = LocalConfig.get_dataset_manager_environment()
        self.account_url = f"https://{dataset_manager_env['data_lake']}.dfs.core.windows.net/"
        self.datalake_service = DataLakeServiceClient(
            account_url=self.account_url,
            credential=self.credential.get_azure_credentials(),
        )

        self.project = project
        self.data_lake_name = dataset_manager_env['data_lake']

    def _get_project_assets(self) -> list:
        """for the specified project, identifies what datasets are available in the datalake that are associated with this project.

        The datasets for the project are named the same as the subdirectories in the datalake>projects>project>datasets fold

        Returns:
            list: subdirectories of the Datasets folder for each project.
        """
        container_name = LocalConfig.get_environment()['project_dataset_container']
        blob = BlobLocation(
            account=self.data_lake_name,
            container=container_name,
            filepath=self.project + "/Datasets",
            filename="",
        )
        # If the container does not exist then try to create it
        containers = self.datalake_service.list_file_systems()
        
        if container_name not in [container.name for container in containers]:
            self._create_container(container_name)


        file_system_client = self.datalake_service.get_file_system_client(
            file_system=blob.get_container()
        )
        self.datasets_path = []

        paths = file_system_client.get_paths(path=blob.get_filepath())
        self.testing_paths = paths

        project_assets = []
        for path in paths:
            self.datasets_path.append(path.name)
            temp = os.sep.join(os.path.normpath(path.name).split(os.sep)[:-1])
            project_assets.append(
                os.sep.join(os.path.normpath(temp).split(os.sep)[-1:])
            )

        project_assets = list(np.unique(project_assets))
        logging.debug(
            f"The project assets are {project_assets}. The blob is {blob.get_filepath()}, The container is {blob.container}"
        )
        project_assets.remove("Datasets")
        return project_assets
    
    def _create_container(self, container_name):
        self.datalake_service.create_file_system(file_system=container_name)
        

    def pull_project_datasets(self, uri_blob_locations: list, skip_datasets=[]) -> dict:
        """Returns all datasets associated with the supplied project name.

        Args:
            uri_blob_locations (list): list of dictionaries - key: datasetname, value: blobLocation object

        Returns:
            dict: dictionary of datasets - key: dataset name, value: pd.dataframe
        """

        datasets = {}
        for each_uri_blob_location in uri_blob_locations:
            path = each_uri_blob_location["path"]
            data_type = each_uri_blob_location["data_type"]
            name = each_uri_blob_location["name"]
            # Skip if requested
            if name in skip_datasets:
                continue
            print(f"pulling dataset {name}")
            if data_type == "csv":
                csv_loader = DataLakeCSVLoader(self.credential)
                datasets[name] = csv_loader.load(path)
            elif data_type == "pkl":
                pkl_loader = DataLakePickleLoader(self.credential)
                datasets[name] = pkl_loader.load(path)
            elif data_type == "json":
                json_loader = DataLakeJsonLoader(self.credential)
                datasets[name] = json_loader.load(path)
            else:
                raise ValueError(
                    f"Data type '{data_type}' loaded from dataset asset is not valid. It must be either 'csv', 'json' or 'pkl'"
                )
        return datasets

    def delete_project_dataset(self, dataset_name: str):
        """deletes the directory passes as dataset_name in the datalake.  The mlstudio dataset shares the name
        with the directory in Datasets for any project.

        Args:
            dataset_name (str): name of the dataset as it appears in mlstudio.
        """

        container_name = LocalConfig.get_environment()['project_dataset_container']
        blob = BlobLocation(
            account=self.data_lake_name,
            container=container_name,
            filepath=self.project + "/Datasets",
            filename="",
        )
        file_system_client = self.datalake_service.get_file_system_client(
            file_system=blob.get_container() + "/" + blob.get_filepath()
        )
        directory_client = file_system_client.get_directory_client(dataset_name)
        if directory_client.exists():
            directory_client.delete_directory()
        else:
            print("The dataset was not found in the specified container")


class MLStudioProjectDatasetsHelper:
    """A class for interacting with resources in MLStudio.  This class is a helper for
    the ProjectDatasetManager and is not intended to be used as a standalone class
    """

    def __init__(self, credentials: DataLakeAuthentication):
        """Initialises the MLStudioProjectDatasetsHelper with a data lake authenticator

        Example:
            auth = DataLakeAuthentication()
            mls_asset_helper = MLStudioProjectDatasetsHelper(auth)

        Args:
            credentials (DataLakeAuthentication): DWHAuthentication instance for retrieving login credentials
        """
        self.credential = credentials
        dataset_manager_env = LocalConfig.get_dataset_manager_environment()
        self.data_lake_name = dataset_manager_env['data_lake']

        self.ml_client = MLClient(
            credential=self.credential.get_azure_credentials(),
            subscription_id=dataset_manager_env["subscription_id"],
            resource_group_name=dataset_manager_env["resource_group"],
            workspace_name=dataset_manager_env["machine_learning_workspace"],
        )

    def _get_asset_overview(self, project_assets: List[str], version=None) -> Dict:
        """returns the dataset name, description, tags and version number of a dataset

        Args:
            project_assets (List[str]): list of datasets to pull information for
            version (_type_, optional): dictionary of dataset name and version. Defaults to None.

        Returns:
            Dict: output dictionary of dataset name, version, description and tags
        """
        outputs = {}
        for each_project in project_assets:
            if version:
                dataset_version = version[each_project]
                asset = self.ml_client.data.get(each_project, version=dataset_version)
            else:
                asset = self.ml_client.data.get(each_project, label="latest")
            outputs[each_project] = {
                "description": asset.description,
                "version": asset.version,
                "tags": asset.tags,
            }
        return outputs

    def _get_path_to_registered_dataset(
            self, project_assets: List[str], versions: Dict[str, str]
    ) -> List:
        """Returns a list of paths to data assets in MLStudio.  The function will get one path per asset
        based on the requested version.  This path will correspond to a location in the datalake where the
        asset is stored

        Args:
            project_assets (list): List of dataset names
            version (Union[int,str]): either version number or

        Returns:
            _type_: list of strings (asset paths)
        """
        assets_in_mlstudio = []
        asset_list = self.ml_client.data.list()
        [assets_in_mlstudio.append(x.name) for x in asset_list]
        available_assets = [x for x in project_assets if x in assets_in_mlstudio]
        missing_project_assets = [
            x for x in project_assets if x not in assets_in_mlstudio
        ]

        asset_paths = []
        for each_asset in available_assets:
            # If the version number is given then get that specific version, else get the latest version
            if each_asset in versions.keys():
                version = versions[each_asset]
                dataset_info = self.ml_client.data.get(
                    name=each_asset,
                    version=version,
                )
            else:
                dataset_info = self.ml_client.data.get(name=each_asset, label="latest")

            if "data_type" not in dataset_info.tags:
                raise ValueError(
                    f"Missing the correct tags inside the registered dataset {each_asset}. It must have the data_type attribute"
                )
            logging.debug(f"The dataset path is {dataset_info}")
            asset_paths.append(
                {
                    "path": dataset_info.path,
                    "data_type": dataset_info.tags["data_type"],
                    "name": each_asset,
                }
            )

        if missing_project_assets:
            logger.info(
                f"missing the following assets from mlstudio: {missing_project_assets}"
            )

        return asset_paths

    def _convert_asset_paths_to_bloblocations(self, azure_dataset_uris: list) -> list:
        """Converts the https:// or abfss:// path into a list of BlobLocation objects

        Args:
            azure_dataset_uris (list): list of asset paths in the format https:// or abfss://

        Raises:
            ValueError: raises error if the asset isnt in the format https:// or abfss://

        Returns:
            list: list of blob location objects for the supplied asset paths
        """
        logging.debug(f"The azure_dataset_uris are {azure_dataset_uris}")
        uris = []
        for uri_dict in azure_dataset_uris:
            uris.append(
                {
                    "path": self._uri_to_blob_location(uri_dict["path"]),
                    "data_type": uri_dict["data_type"],
                    "name": uri_dict["name"],
                }
            )
        return uris

    def _uri_to_blob_location(self, uri: str) -> BlobLocation:
        """Convert a single Azure URI path into a BlobLocation object depending on if it is an abfs:// or https:// format

        Args:
            uri (str): Blob URI path

        Raises:
            ValueError: Raised if path isnt in the correct format

        Returns:
            BlobLocation: BlobLocation representing the input URI
        """
        if "https://" in uri:
            logging.debug(f"\n\n\n running https: {uri} \n\n\n")
            blob = BlobLocation.from_https_path(uri=uri)
            logging.debug(f"\n\n\n running https complete: {uri} \n\n\n")
            return blob
        elif "abfss://" in uri:
            logging.debug(f"\n\n\n running abfss: {uri} \n\n\n")
            blob = BlobLocation.from_abfss_path(uri=uri)
            logging.debug(f"\n\n\n running abfss complete: {uri} \n\n\n")
            return blob
        else:
            raise ValueError("uri must be either https format or abfss format")

    def get_path_to_dataset(
            self, project_assets: list, versions: Dict[str, str]
    ) -> List[BlobLocation]:
        """Gets the bloblocation objects for each supplied project dataset

        Args:
            project_assets (list): list of dataset names as they appear in MLStudio
            version (int): resuested version.  Leave as None for "latest"

        Returns:
            List[BlobLocation]: list of blob location objects for each data asset
        """
        azure_dataset_uris = self._get_path_to_registered_dataset(
            project_assets=project_assets, versions=versions
        )
        return self._convert_asset_paths_to_bloblocations(azure_dataset_uris)

    def generate_dataset_path(
            self, registered_dataset_name: str, project: str
    ) -> BlobLocation:
        """Generates a BlobLocation object with uuid file name for a dataframe to be registered in Azure

        Args:
            registered_dataset_name (str): as will be shown in MLStudio
            project (str): name of the data science project

        Returns:
            BlobLocation: blob location object for file to be saved in the data lake
        """
        file_uuid = uuid.uuid4()
        container_name = LocalConfig.get_environment()['project_dataset_container']
        return BlobLocation(
            account=self.data_lake_name,
            container=container_name,
            filepath=project + "/Datasets/" + registered_dataset_name,
            filename=str(file_uuid),
        )

    def register_dataset(self, data: Data):
        """Function to register a dataset in MLStudio.  This function is intended to be used by a
        ProjectDatasetManager, and should not be used as a stand alone class

        Args:
            data (Data): MLClient data object
        """
        self.ml_client.data.create_or_update(data)


class ProjectDatasetManager(IProjectDataset):
    """A Class to manage acquiring project datasets and registering them in Azure.  This class can be used in
    a notebook or in a script to acquire all datasets connected with a given project.  The class will also
    handle registering a new dataset in mlstudio and simultaneously creating a directory and saving in the
    data lake


    Example Use Case:
    ```python
    # To load all datasets for a project
    from datascience_core.data_retrieval import ProjectDatasetManager
    project = "test"

    dataset_manager = ProjectDatasetManager(project=project)
    datasets = dataset_manager.load_datasets()
    datasets.keys()

    datasets = dataset_manager.load_datasets(get_these_datasets=['test1','test2'])

    # To obtain information about the datasets registered to the project
    from datascience_core.data_retrieval import ProjectDatasetManager
    project = "test"

    dataset_manager = ProjectDatasetManager(project=project)
    dataset_manager.list_datasets()

    dataset_manager.list_dataset_descriptions()

    datasets = ['PP4_true_prime_motonova_label']
    data_manager.list_dataset_descriptions(datasets=datasets)

    version = {
    'PP4_bad_features':"1",
    'PP5_test_dataset':"1"
    }
    data_manager.list_dataset_descriptions(version=version)


    # To register a dataset in MLStudio
    data = pd.DataFrame({'col1':[0,1,2,3,4,5],'col2':[0,1,2,3,4,5],'col3':[0,1,2,3,4,5],'col4':[0,1,2,3,4,5]})

    #either register the file
    dataset_name = 'testa'
    dataset_manager.register_dataset(dataset_name,data)

    #or pass the path to the file
    dataset_name = 'testb'
    path = 'https://ds247dldev.blob.core.windows.net/projects/test/Datasets/testb/6078b5f1-2626-4dc8-8b35-c8bf55b2b28a'
    dataset_manager.register_dataset(dataset_name,path)

    ```
    """

    def __init__(self, project: str):
        """initialises a ProjectDatasetManager

        Args:
            project (str): name of the data science project
        """
        self.credential = DataLakeAuthentication()

        self.project = project
        self.project_asset_loader = DatalakeProjectAssetsHelper(
            self.credential, self.project
        )
        self.MLStudio_asset_helper = MLStudioProjectDatasetsHelper(self.credential)

        self.stage_for_delete_dataset_name = None

    def list_datasets(self) -> List:
        """Returns a list of all datasets registered to the specified project

        Returns:
            list: datasets
        """
        return self.project_asset_loader._get_project_assets()

    def list_dataset_descriptions(
            self, datasets: List = [], version: Dict[str, str] = {}
    ) -> Dict:
        """Lists the descriptions of each dataset, version number and its tags

        Args:
            datasets (List, optional): named list of datasets to get information for. Defaults to [].
            version (Dict[str, str], optional): dictionary of datasets and version number.  Supersedes datasets argument. Defaults to {}.

        Returns:
            dict: dataset information
        """

        assets = self.project_asset_loader._get_project_assets()
        if datasets and not version:
            assets = [x for x in assets if x in datasets]
        if version:
            assets = [x for x in assets if x in list(version.keys())]

        return self.MLStudio_asset_helper._get_asset_overview(
            project_assets=assets, version=version
        )

    def load_datasets(
            self,
            get_these_datasets: list = None,
            versions: Dict[str, str] = {},
            skip_datasets: list = [],
    ) -> dict:
        """Loads the registered dataset for the project from Azure

        Args:
            version (dict, optional): dictionary of dataset: version.  Leave as None to get the latest Defaults to None.

        Returns:
            dict: dictionary of datsetname : pd.DataFrame value pairs
        """
        self.versions = versions
        project_assets = self.project_asset_loader._get_project_assets()
        uri_blob_locations = self.MLStudio_asset_helper.get_path_to_dataset(
            project_assets=project_assets, versions=versions
        )

        if get_these_datasets:
            uri_blob_locations = [
                uri_blob_location
                for uri_blob_location in uri_blob_locations
                if uri_blob_location["name"] in get_these_datasets
            ]

        return self.project_asset_loader.pull_project_datasets(
            uri_blob_locations, skip_datasets=skip_datasets
        )

    def register_dataset(
            self,
            registered_dataset_name: str,
            data: Any,
            description: str = None,
            tags: dict = {},
            register_as_pickle: bool = True
    ):
        """Saves the supplied data in the data lake under the registered dataset name and registers as a data asset in ML Studio.

        Args:
            registered_dataset_name (str): The name of the registered dataset, which will appear as a subdirectory in the Datalake
                    "../projects/<PROJECT>/Datasets" folder, and as the registered asset name in ML Studio.
            data (Union[pd.DataFrame, str, dict, list]): The data to register. This can be a Pandas DataFrame or a path to a blob.
            description (Optional[str]): A description to associate with the dataset. Default is None.
            tags (Optional[Dict[str, Any]]): Additional tags to associate with the dataset. Default is an empty dictionary.
            register_as_pickle (bool): Whether to register the data as a pickle file or not. Defaults to True.

        Raises:
            TypeError: If no data is passed to register.
        """

        if data is None:
            raise TypeError("You need to pass something to register")

        tags["data_type"], save_blob = self._register_dataset_based_on_datatype(data,
                                                                                registered_dataset_name,
                                                                                register_as_pickle,
                                                                                )

        dataset = Data(
            path=save_blob,
            type=AssetTypes.URI_FILE,
            description=description,
            name=registered_dataset_name,
            tags=tags,
        )
        self.MLStudio_asset_helper.register_dataset(dataset)

    def _register_dataset_based_on_datatype(self,
                                            data: Any,
                                            registered_dataset_name: str,
                                            register_as_pickle: bool):

        """Calculates and applies the logical workflow required by the register_dataset function.

                    :param registered_dataset_name: The name of the registered dataset, which will appear as a subdirectory in
                     the Datalake "../projects/<PROJECT>/Datasets" folder, and as the registered asset name in ML Studio.
                    :type registered_dataset_name: str
                    :param data: The data to register. This can be a Pandas DataFrame or a path to a blob.
                    :type data: [pd.DataFrame, str, dict, list]
                    :param register_as_pickle: Whether to register the data as a pickle file or not. Defaults to True.
                    :type register_as_pickle: bool

                """
        # 
        save_blob = self.MLStudio_asset_helper.generate_dataset_path(
            registered_dataset_name=registered_dataset_name, project=self.project)

        if register_as_pickle:
            data_lake_saver = DataLakePickleSaver(self.credential)
            data_lake_saver.save(blob_location=save_blob, data=data)
            data_type = "pkl"
            return data_type, save_blob

        elif isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            data_lake_saver = DataLakeCSVSaver(self.credential)
            data_lake_saver.save(blob_location=save_blob, df=data)
            data_type = "csv"
            return data_type, save_blob

        elif isinstance(data, dict) or isinstance(data, List):
            data_lake_saver = DataLakeJsonSaver(self.credential)
            data_lake_saver.save(blob_location=save_blob, data=data)
            data_type = "json"
            return data_type, save_blob

    def remove_dataset(self, dataset_name: str):
        """Initiates a deletion for a dataset.  This function stages the dataset for deletion, but must be used with <confirm_destroy()>
        for the dataset to be deleted

        Args:
            dataset_name (str): name of the dataset as it appears in mlstudio.
        """
        self.stage_for_delete_dataset_name = dataset_name

        print(f"{dataset_name} staged for deletion")
        print(f"Do you want to delete {dataset_name}?  This action is permanent")
        print(
            "To delete, run ProjectDatasetManager.confirm_destroy() and pass the dataset"
        )

    def confirm_destroy(self, confirm_dataset_name: str):
        """Confirmation of deletion.  remove_dataset() must be called first to stage the dataset for deletion.
        If an unmatched dataset is passed here, staging will be reset.  If a matching dataset is passed, deletion will be
        confirmed

        Args:
            confirm_dataset_name (str): dataset to be deleted
        """
        if self.stage_for_delete_dataset_name:
            if self.stage_for_delete_dataset_name == confirm_dataset_name:
                self.project_asset_loader.delete_project_dataset(
                    dataset_name=confirm_dataset_name
                )
                self.stage_for_delete_dataset_name
                print(f"{self.stage_for_delete_dataset_name} was deleted successfully")
                self.stage_for_delete_dataset_name = None
            else:
                print(
                    "Dataset names don't match, cancelling operation.  Re-run ProjectDatasetManager.remove_dataset to try again"
                )
                self.stage_for_delete_dataset_name = None
        else:
            print(
                "No file staged for deletion.  Run ProjectDatasetManager.remove_dataset() first"
            )



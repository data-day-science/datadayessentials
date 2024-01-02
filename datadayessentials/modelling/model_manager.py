import os
import shutil
from azureml.core import Workspace, Model, Run
from azureml.core.authentication import (
    ServicePrincipalAuthentication,
    InteractiveLoginAuthentication,
)
from datadayessentials.modelling._base import IModelManager
from datadayessentials.config import Config
from .utils import get_workspace
from pathlib import Path
from typing import Dict


class ModelCacher:
    """
    Class to cache the model files in the core cache.

    Example Use Case:
        model_cacher = ModelCacher("model", 1)
        if not model_cacher.is_model_cached():
            model_cacher.cache_model("/tmp/model")
        else:
            ... download model to /tmp/model
            model_cacher.copy_model_from_cache("/tmp/model")
    """

    def __init__(self, model_name: str, model_version: int, cache_location):
        self.model_name = model_name
        self.model_version = model_version
        self.cache_location = Path(cache_location)

    def is_model_cached(self):
        return self._get_model_cache_path().exists()

    def _get_model_cache_path(self):
        return self.cache_location / f"{self.model_name}-{self.model_version}"

    def copy_model_folder_to_cache(self, model_path: str):
        shutil.copytree(model_path, self._get_model_cache_path())

    def copy_model_folder_from_cache(self, model_path: str):
        shutil.copytree(self._get_model_cache_path(), model_path)


class ModelManager(IModelManager):
    """
    Class to manage Azure ML models.

    Args:
        ml_studio_name (str): Name of the Azure ML studio
        subscription_id (str): Subscription id of the Azure ML studio
        resource_group (str): Resource group of the Azure ML studio
    """

    def __init__(self) -> None:
        self.workspace = get_workspace()

    def get_model_files_from_run(self, run_id: str, folder_to_save_model: str = None):
        """
        Downloads the model files from the run

        Args:
            run_id (str): Run id of the model
            folder_to_save_model (str): Folder to save the model files
        """

        folder_to_save_model = (
            f"{run_id}_model_files"
            if folder_to_save_model is None
            else folder_to_save_model
        )
        self.workspace.get_run(run_id).download_files(
            output_directory=folder_to_save_model, timeout_seconds=60
        )
        return folder_to_save_model

    def get_model_files_from_registered_model(
        self,
        model_name: str,
        model_version: int = None,
        folder_to_save_model: str = None,
    ):
        """
        Downloads the model files from the registered model

        Args:
            model_name (str): Name of the model
            model_version (int): Version of the model
            folder_to_save_model (str): Folder to save the model files

        Example
            mlflow_manager.get_model_files_from_registered_model("model", "1", "/tmp/model")

        """

        if not model_version:
            model_version = Model.list(self.workspace, name=model_name, latest=True)[
                0
            ].version
        print(f"Downloading model {model_name} version {model_version}")

        folder_to_save_model = (
            f"{model_name}-{model_version}"
            if folder_to_save_model is None
            else folder_to_save_model
        )

        if os.path.exists(folder_to_save_model):
            shutil.rmtree(folder_to_save_model)

        model = Model(self.workspace, model_name, version=model_version)
        cache_location = os.path.join(
            os.path.expanduser("~"),
            Config().get_environment_variable("local_cache_dir"),
        )

        model_cacher = ModelCacher(model_name, model_version, cache_location)
        if model_cacher.is_model_cached():
            model_cacher.copy_model_folder_from_cache(folder_to_save_model)
            return folder_to_save_model
        max_retries = 5
        for retries in range(1, max_retries + 1):
            try:
                download_folder = model.download(
                    target_dir=folder_to_save_model, exist_ok=True
                )
                # Check if folder is empty
                if not os.listdir(folder_to_save_model):
                    raise Exception("model not downloaded")
            except Exception as e:
                if retries == max_retries:
                    raise e
                print(f"failed to download model: attempt {retries}. Error was {e}")
        model_cacher.copy_model_folder_to_cache(folder_to_save_model)
        return folder_to_save_model

    def get_model_properties_from_run(self, run_id: str):
        """
        Gets the model properties from the run

        Args:
            run_id (str): Run id of the model

        Example:
            mlflow_manager.get_model_properties_from_run("2cf2751a-e2b8-44a7-bad0-a3f2d923cb2e")
        """
        return self.workspace.get_run(run_id).properties

    def get_model_properties_from_registered_model(
        self, model_name: str, model_version: int = None
    ):
        """
        Gets the model properties from the registered model

        Args:
            model_name (str): Name of the model
            model_version (int): Version of the model

        Example:
            mlflow_manager.get_model_properties_from_registered_model("model", "1")
        """
        model = Model(self.workspace, model_name, version=model_version)
        return model.properties

    def get_model_run_id_from_registered_model(
        self, model_name: str, model_version: int = None
    ):
        """
        Gets the model job id from the registered model

        Args:
            model_name (str): Name of the model
            model_version (int): Version of the model

        Example:
            mlflow_manager.get_model_job_id_from_registered_model("model", "1")
        """
        return Model(self.workspace, model_name, version=model_version).run_id

    def register_model_from_local_folder(
        self,
        model_name: str,
        model_path: str,
        model_version: int = None,
        properties: dict = None,
        tags: dict = {},
    ) -> Model:
        """
        Registers the model in the Azure ML studio

        Args:
            model_name (str): Name of the model
            model_path (str): Path of the model
            model_version (int): Version of the model
            properties (dict): Properties to be added to the model

        Example:
            mlflow_manager.register_model("model", "/tmp/model", 1)

        """
        return Model.register(
            workspace=self.workspace,
            model_name=model_name,
            model_path=model_path,
            tags={"version": model_version, **tags},
            properties=properties,
        )

    def register_model_from_run_id(
        self,
        run_id: str,
        model_name: str,
        model_path: str = "model",
        tags: dict = {},
        properties: dict = None,
    ) -> Model:
        """
        Registers the model from the run id in the Azure ML studio, without downloading the model files locally

        Args:
            run_id (str): Run id of the model
            model_name (str): Name of the model
            model_path (str): Path of the model
            tags (dict): Tags to be added to the model
            properties (dict): Properties to be added to the model

        Example:
            mlflow_manager.register_model_from_run("2cf2751a-e2b8-44a7-bad0-a3f2d923cb2e", "model")

        """
        return self.workspace.get_run(run_id).register_model(
            model_name=model_name,
            model_path=model_path,
            tags=tags,
            properties=properties,
        )

    def register_ensemble_model_from_run_ids(
        self,
        run_ids: Dict[str, str],
        ensemble_model_name: str,
        properties: dict = None,
        tags: dict = {},
    ):
        """From the run ids provided, downloads all the run ids to a local temporary folder and uploads them all as a single model to the model registry.

        Args:
            run_ids (Dict[str, str]): Dictionary containing the name of the individual models and their run ids
            model_name (str): Name of the model to be registered
        """
        ensemble_model_folder = os.path.join(
            Config().get_environment_variable("local_cache_dir"), "ensemble_model_files"
        )
        if os.path.exists(ensemble_model_folder):
            shutil.rmtree(ensemble_model_folder)
        os.makedirs(ensemble_model_folder)
        for model_name, run_id in run_ids.items():
            model_folder_name = ensemble_model_folder + "/" + model_name
            self.get_model_files_from_run(run_id, model_folder_name)
        self.register_model_from_local_folder(
            ensemble_model_name, ensemble_model_folder, properties=properties, tags=tags
        )
        shutil.rmtree(ensemble_model_folder)

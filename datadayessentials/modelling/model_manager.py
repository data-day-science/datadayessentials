import os
from azureml.core import Workspace, Model, Run
from azureml.core.authentication import (
    ServicePrincipalAuthentication,
    InteractiveLoginAuthentication,
)
from datadayessentials.modelling._base import IModelManager
from datadayessentials.config import LocalConfig
from .utils import get_workspace


class ModelManager(IModelManager):
    """
    Class to manage Azure ML models.

    Args:
        ml_studio_name (str): Name of the Azure ML studio
        subscription_id (str): Subscription id of the Azure ML studio
        resource_group (str): Resource group of the Azure ML studio
    """

    def __init__(
        self
    ) -> None:
        raise ValueError
        self.workspace = get_workspace()


    def get_model_files_from_run(self, run_id: str, folder_to_save_model: str = None):
        """
        Downloads the model files from the run

        Args:
            run_id (str): Run id of the model
            folder_to_save_model (str): Folder to save the model files

        Examples:
            mlflow_manager.workspace.get_run("2cf2751a-e2b8-44a7-bad0-a3f2d923cb2e").download_files(output_directory="/tmp/model")
        """

        folder_to_save_model = (
            f"{run_id}_model_files"
            if folder_to_save_model is None
            else folder_to_save_model
        )
        self.workspace.get_run(run_id).download_files(
            output_directory=folder_to_save_model
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

        folder_to_save_model = (
            f"{model_name}-{model_version} model files"
            if folder_to_save_model is None
            else folder_to_save_model
        )

        Model(self.workspace, model_name, version=model_version).download(
            target_dir=folder_to_save_model, exist_ok=True
        )
        return folder_to_save_model

    def register_model_from_local_folder(
        self, model_name: str, model_path: str, model_version: int = None
    ) -> Model:
        """
        Registers the model in the Azure ML studio

        Args:
            model_name (str): Name of the model
            model_path (str): Path of the model
            model_version (int): Version of the model

        Example:
            mlflow_manager.register_model("model", "/tmp/model", 1)

        """
        return Model.register(
            workspace=self.workspace,
            model_name=model_name,
            model_path=model_path,
            tags={"version": model_version},
        )

    def register_model_from_run_id(
        self, run_id: str, model_name: str, model_path: str = 'model', tags: dict = None
    ) -> Model:
        """
        Registers the model from the run id in the Azure ML studio, without downloading the model files locally

        Args:
            run_id (str): Run id of the model
            model_name (str): Name of the model
            model_path (str): Path of the model
            tags (dict): Tags to be added to the model

        Example:
            mlflow_manager.register_model_from_run("2cf2751a-e2b8-44a7-bad0-a3f2d923cb2e", "model")

        """
        return self.workspace.get_run(run_id).register_model(
            model_name=model_name, model_path=model_path, tags=tags
        )

import sys

import mlflow.version as mlflow_version
import azureml.core as azure_ml_core
from azureml.core import Workspace, Experiment, Run

from datadayessentials.authentications import DataLakeAuthentication
from azureml.core.authentication import (
    ServicePrincipalAuthentication,
    InteractiveLoginAuthentication,
)
from datadayessentials.modelling.models._base import IModelSavingLoadingAttribute
from datadayessentials.modelling._base import IExperimentManager
from datadayessentials.config import LocalConfig
import os
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
import uuid

import mlflow
from pathlib import Path
import shutil
from typing import Tuple
from .utils import get_workspace


class ExperimentManager(IExperimentManager):
    mlflow_folder_name = "./mlruns"

    def __init__(
        self,
        experiment_name: str,
        experiment_description: str = None,
    ):
        """A Class to handle submitting data from machine learning runs to mlflow

        Args:
            ml_studio_name (str): Azure ML Studio name
            subscription_id (str): Subscription ID
            resource_group (str): Resource Group
            experiment_name (str): Experiment name as appears in the Jobs section of ML Studio
            'model_save_path (str): directory in which to save the model in MLFlow
        """
        self.experiment_name = experiment_name
        self.experiment_description = experiment_description
        self.workspace = get_workspace()

    def _create_or_update_experiment(
        self, experiment_name: str, experiment_description: str = None
    ) -> Experiment:
        """Creates or updates an Azure Experiment"""
        azure_experiment = Experiment(self.workspace, experiment_name)

        if experiment_description:
            azure_experiment.set_description(experiment_description)
        return azure_experiment

    def submit_run(
        self,
        datasets_used: dict,
        model: IModelSavingLoadingAttribute = None,
        run_name: str = None,
        train_model_metrics: dict = None,
        test_model_metrics: dict = None,
        validate_model_metrics: dict = None,
        other_model_metrics: dict = None,
        tags: dict = None,
        params: dict = None,
        artifacts_folder: str = None,
    ) -> Tuple[str, str]:
        """
        Submits a run to the Azure Experiment
        
        Args:
            datasets_used (dict): dictionary of datasets used in the run, where the key is the name of the registered dataset and the value is the version of the dataset
            model (IModelSavingLoadingAttribute): model to be saved
            run_name (str): name of the run, if not provided a uuid will be generated
            train_model_metrics (dict): dictionary of training metrics
            test_model_metrics (dict): dictionary of testing metrics
            validate_model_metrics (dict): dictionary of validation metrics
            other_model_metrics (dict): dictionary of other metrics
            tags (dict): dictionary of tags
            params (dict): dictionary of parameters used to train the model
            artifacts_folder (str): path to the artifacts folder containing any images or other artifacts to be logged
        
        Returns:
            Tuple[str, str]: tuple of the run id and run name
        """
        azure_experiment: Experiment = self._create_or_update_experiment(
            self.experiment_name, self.experiment_description
        )

        if not run_name:
            run_name = uuid.uuid1()
        azure_run = azure_experiment.start_logging(
            outputs=None, display_name=run_name, snapshot_directory=None
        )
        try:
            print("submission started")
            self._add_data_to_run(
                azure_run,
                tags,
                params,
                train_model_metrics,
                validate_model_metrics, 
                test_model_metrics,
                other_model_metrics,
                artifacts_folder,
                datasets_used,
            )
            if model:
                self._add_model_to_run(
                    azure_run,
                    model,
                )
        finally:
            azure_run.complete()
        return azure_run.id, azure_run.name

    def _add_data_to_run(
        self,
        azure_run: Run,
        tags: dict,
        params: dict,
        train_model_metrics: dict,
        validate_model_metrics: dict,
        test_model_metrics: dict,
        other_model_metrics: dict,
        artifacts_folder: str,
        versioning_meta_data: dict,
    ):
        """
        Adds meta data to the azure experirment run
        """
        if not versioning_meta_data:
            raise ValueError("No data versioning input found")

        if tags:
            print("tags found,... submitting")
            azure_run.set_tags(tags)

        if params:
            print("model params found,... submitting")
            azure_run.add_properties(params)

        if train_model_metrics:
            print("training metrics found,... submitting")
            for key, val in train_model_metrics.items():
                azure_run.log("train_" + key, val)

        if validate_model_metrics:
            print("validation metrics found,... submitting")
            for key, val in validate_model_metrics.items():
                azure_run.log("validate_" + key, val)

        if test_model_metrics:
            print("test metrics found,... submitting")
            for key, val in test_model_metrics.items():
                azure_run.log("test_" + key, val)
        
        if other_model_metrics:
            print("other metrics found,... submitting")
            for key, val in other_model_metrics.items():
                azure_run.log(key, val)

        if artifacts_folder:
            print("artifacts found,... submitting")
            azure_run.upload_folder(name="artifacts", path=artifacts_folder)

        azure_run.add_properties({"datasets_used": versioning_meta_data})
        
        print("versioning_meta_data found,... submitting")

    def _add_model_to_run(self, azure_run: Run, model: IModelSavingLoadingAttribute):
        """
        Creates an mlflow locally and uploads this to the azure experiment run

        Args:
            azure_run (Run): azure experiment run
            model (IModelSendingAttribute): model object
        """
        local_cache_dir = LocalConfig.get_local_cache_dir()
        model_save_folder = os.path.join(local_cache_dir, "temp_model")
        shutil.rmtree(model_save_folder, ignore_errors=True)

        _ = model._save_model_to_folder(model_save_folder)
        azure_run.upload_folder(name="model", path=model_save_folder)
        shutil.rmtree(model_save_folder, ignore_errors=True)

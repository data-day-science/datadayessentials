from azureml.core import Experiment

from datadayessentials.modelling.models._base import IModelSavingLoadingAttribute
from datadayessentials.modelling.model_manager import ModelManager
from datadayessentials.config import LocalConfig
import os
from datadayessentials.modelling.models._base import IModelFactory


class ModelFactory(IModelFactory):
    """
    Class for loading models stored as either azure runs or an azure registered model. Can be used to load these models into any one of the model wrapping classes in datadayessentials.modelling.models package.

    Example use case:
        # Load from run id (for an experiment that has no registered model)
        run_id = "some_run_id"
        model_factory = ModelFactory(CatboostClassifierPipeline)
        model = model_factory.create_from_run(run_id)

        # Or load from model name and version

        model_name = "some_model_name"
        model_version = 1
        model_factory = ModelFactory(CatboostClassifierPipeline)
        model = model_factory.create_from_registered_model(model_name, model_version)

    """

    def create_from_run_id(
        self, run_id: str, model_path_in_job: str = "model"
    ) -> IModelSavingLoadingAttribute:
        """
        Loads the model from the run id.

        Args:
            run_id (str): Run id of the model
            model_path_in_job (str): Path to the location of the model folder stored in the run artifacts

        Returns:
            IModelSavingLoadingAttribute: Model loaded from the run id, the type of model is specified in the __init__ method of the ModelFactory class
        """
        model_manager = ModelManager()
        model_files_download_location = os.path.join(
            LocalConfig.get_local_cache_dir(), "temp_model_files"
        )
        model_location = os.path.join(model_files_download_location, model_path_in_job)

        model_manager.get_model_files_from_run(run_id, model_files_download_location)
        return self.model_class.load_model_from_folder(model_location)

    def create_from_registered_model(
        self,
        model_name: str,
        model_version: str = None,
        model_path_in_job: str = "model",
    ) -> IModelSavingLoadingAttribute:
        """
        Loads the model from the model reigstry.

        Args:
            model_name (str): Name of the model
            model_version (str): Version of the model
            model_path_in_job (str): Path to the location of the model folder stored in the model artifacts

        Returns:
            IModelSavingLoadingAttribute: Model loaded from the model registry, the type of model is specified in the __init__ method of the ModelFactory class
        """
        model_manager = ModelManager()
        model_files_download_location = os.path.join(
            LocalConfig.get_local_cache_dir(), "temp_model_files"
        )
        model_location = os.path.join(model_files_download_location, model_path_in_job)

        model_manager.get_model_files_from_registered_model(
            model_name,
            model_version=model_version,
            folder_to_save_model=model_files_download_location,
        )
        return self.model_class.load_model_from_folder(model_location)

    @staticmethod
    def get_latest_run_id_from_experiment(experiment_name: str) -> str:
        """
        Gets the latest run id from the experiment.

        Args:
            experiment_name (str): Name of the experiment

        Returns:
            str: Latest run id from the experiment
        """
        model_manager = ModelManager()
        experiments = Experiment(model_manager.workspace, experiment_name)
        runs = experiments.get_runs()
        next_run = next(runs)
        experiment_id = next_run.id

        return experiment_id

from abc import ABC, abstractmethod
from typing import Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlflow.models.model import ModelInfo
from typing import Union, Tuple, List
from datadayessentials.modelling.models._base import IModelSavingLoadingAttribute


class ICatboostMetric(ABC):
    @abstractmethod
    def is_max_optimal(self):
        pass

    @abstractmethod
    def evaluate(self, approxes, target, weight):
        pass

    @abstractmethod
    def get_final_error(self, error, weight):
        pass


class IModelManager(ABC):
    @abstractmethod
    def register_model(
        self, model_name: str, model_path: str, model_version: int = None
    ) -> None:
        pass


class IExperimentManager(ABC):
    @abstractmethod
    def submit_run(
        self,
        experiment_name,
        experiment_description: str,
        model_metrics: dict,
        versioning_meta_data: dict,
        model: IModelSavingLoadingAttribute,
    ) -> str:
        pass


class IModelManager(ABC):
    @abstractmethod
    def register_model_from_local_folder(
        self, model_name: str, model_path: str, model_version: int = None
    ):
        pass

    @abstractmethod
    def get_model_files_from_registered_model(
        self,
        model_name: str,
        model_version: int = None,
        folder_to_save_model: str = None,
    ) -> str:
        pass

    @abstractmethod
    def get_model_files_from_run(
        self, run_id: str, folder_to_save_model: str = None
    ) -> str:
        pass


class IModelEvaluator(ABC):
    """A class to evaluate the outputs of a binary classifier."""

    def __init__(self, model=None):
        self.model = model

    def set_model(self, model):
        self.model = model

    @abstractmethod
    def make_predictions(
        self, y_proba: Union[pd.Series, list, np.array], boundary: float = 0.5
    ) -> np.array:
        """return the class predictions based off the supplied boundary

        Args:
            y_proba (Union[pd.Series,list,np.array]): class probabilities for each sample
            boundary (float, optional): boundary for class separation. Defaults to 0.5.

        Returns:
            np.array: array of predictions for the given input data
        """
        pass

    @abstractmethod
    def get_probas(self, X: pd.DataFrame) -> np.array:
        """returns the probas for the given input data

        Args:
            X (pd.DataFrame): model input data.  Should be processed and in the correct format for the model to accept

        Returns:
            np.array: array of probas for the given input data
        """
        pass

    @abstractmethod
    def calculate_metrics(
        self,
        y_pred: Union[list, np.array],
        y_proba: Union[list, np.array],
        y: Union[list, np.array],
    ) -> dict:
        """calculates the model metrics for the given predictions and truth labels.
        Args:
            y_pred (Union[list,np.array]): the class predictions
            y_proba (Union[list,np.array]): class probabilities
            y (Union[list,np.array]): truth labels

        Returns:
            dict: dictionary of calculated metrics
        """
        pass

    @abstractmethod
    def plot_figures(
        self,
        y_pred: Union[list, np.array],
        y_proba: Union[list, np.array],
        y: Union[list, np.array],
        metrics: dict,
        labels: list = ["positive", "negative"],
        boundary: float = 0.5,
    ) -> plt.figure:
        """creates an output plot
        Args:
            y_pred (Union[list,np.array]): the class predictions
            y_proba (Union[list,np.array]): class probabilities
            y (Union[list,np.array]): truth labels
            metrics (dict): dictionary of calculated metrics
            labels (list, optional): class labels. Defaults to ["positive", "negative"].
            boundary (float, optional): boundary for class separation. Defaults to 0.5.
        Returns:
            plt.figure: evaluation figure
        """
        pass

    @abstractmethod
    def print_summary(self, model_metrics: dict):
        """displays some summary information

        Args:
            model_metrics (dict): dictionary of calculated metrics
        """
        pass

    @abstractmethod
    def run(
        self, X: pd.DataFrame, y: Union[list, pd.Series, np.array], verbose=True
    ) -> dict:
        """runs the evaluator and returns a figure and summary output

        Args:
            X (pd.DataFrame): model input data.  Should be processed and in the correct format for the model to accept
            y (Union[list,pd.Series,np.array]): class truth labels
            verbose(bool): print outputs

        Returns:
            dict: returns a dictionary of the figure object and the dictionary of metrics
        """
        pass

   
    class IInferenceModel(ABC):
        """Interface for inference model"""
        
        feature_names_: List[str] = None
        feature_importances_: List[float] = None
        model = None
            
        @abstractmethod
        def __init__(self, model):
            if not model.model:
                raise ValueError("Model has not been fit.  Only a trained model should be passed to this class")
            

        @abstractmethod
        def predict(self, X: pd.DataFrame) -> np.array:
            """Predicts the class for the given input data

            Args:
                X (pd.DataFrame): model input data.  Should be processed and in the correct format for the model to accept

            Returns:
                np.array: array of predictions for the given input data
            """
            pass

        @abstractmethod
        def predict_proba(self, X: pd.DataFrame) -> np.array:
            """Predicts the class probabilities for the given input data

            Args:
                X (pd.DataFrame): model input data.  Should be processed and in the correct format for the model to accept

            Returns:
                np.array: array of class probabilities for the given input data
            """
            pass

        

from abc import ABC, abstractmethod
from mlflow.models.model import ModelInfo
from typing import Union, Tuple, List
from types import ModuleType
import pandas as pd
from typing import Type


class IModelSavingLoadingAttribute(ABC):
    @abstractmethod
    def _save_model_to_folder(self) -> ModelInfo:
        """Save the model to a local folder"""
        pass

    @abstractmethod
    def load_model_from_folder(self, model_folder: str) -> object:
        """Loads the model from mlflow

        Args:
            model_folder (str): path to the model folder
        """
        pass


class IModelFactory(ABC):
    def __init__(self, model_class: Type[IModelSavingLoadingAttribute]) -> None:
        self.model_class = model_class

    @abstractmethod
    def create_from_run_id(self, run_id) -> IModelSavingLoadingAttribute:
        """Creates a model from a run id

        Args:
            run_id ([type]): unique run id
        """
        pass

    @abstractmethod
    def create_from_registered_model(
        self, model_name: str, model_version: int = None
    ) -> IModelSavingLoadingAttribute:
        """Creates a model from a registered model

        Args:
            model_name (str): name of the model
            model_version (int, optional): version of the model. Defaults to None.
        """
        pass


class IModel(ABC):
    def fit(
        self, X: pd.DataFrame, y: Union[List, pd.Series], *args, **kwargs
    ) -> object:
        """Function to fit the model

        Args:
            X (pd.DataFrame): Features
            y (Union[List,pd.Series]): truth label

        Returns:
            object: fitted model
        """
        pass

    def predict(self, X: pd.DataFrame) -> Union[List, pd.Series]:
        """Function to predict on a dataset

        Args:
            X (pd.DataFrame): Features

        Returns:
            Union[List,pd.Series]: predicted labels
        """
        pass

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Function to predict probabilities on a dataset

        Args:
            X (pd.DataFrame): Features

        Returns:
            pd.DataFrame: predicted probabilities
        """
        pass


from types import ModuleType
from mlflow.models.model import ModelInfo
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from datascience_core.modelling._base import IModelEvaluator
from datascience_core.modelling.models._base import IModel, IModelSavingLoadingAttribute

from typing import Dict, List, Union
import os


class NeuralNetworkBinaryClassifierPipeline(IModel, IModelSavingLoadingAttribute):
    model: keras.Model

    def __init__(
        self,
        evaluator: IModelEvaluator = None,
        figure_dir = 'output_figures',
        model: keras.Model = None,
        **kw,
    ):
        self.figure_dir = figure_dir
        path = os.path.join(os.getcwd(), f"{self.figure_dir}")
        if not os.path.exists(path):
            os.makedirs(path)
        self.evaluator = evaluator
        
        if model is not None:
            self.model = model
        else:
            self.model = keras.Model(**kw)


    def evaluate_model(
        self, X: pd.DataFrame, y_true: pd.Series, verbose: bool = True
    ) -> Dict:
        """runs the model evaluator class

        Args:
            X (pd.DataFrame): Features
            y_true (pd.Series): Truth label
            verbose (bool): print graphs and metrics

        Raises:
            AttributeError: Raises an error if no evaluator was passed to the pipeline at instantiation

        Returns:
            dict: dictionary of evaluation metrics
        """
        if self.evaluator:
            self.evaluator.set_model(model=self)
            return self.evaluator.run(X, y_true, verbose=verbose)
            return self.evaluator.run(X, y_true, verbose=verbose)
        else:
            raise AttributeError("No DSModelEvaluator set for this class")

    def fit(self, *args, **kwargs):
        """
        Wrapper for the fit function, then saving the history object into this class
        """
        history = self.model.fit(*args, **kwargs)
        self.model.model_history = history.history

    def plot_and_save_history(self) -> Dict[str, str]:
        """
        Plots the training history of the nerual network, and saves the figures locally  so that it can be uploaded to MLFlow

        Returns:
            dict[str, str]: Dictionary of the name of the figure and the save location
        """
        fig_names = {}
        for key in self.model.model_history.keys():
            if "val_" not in key:
                if "val_" + key not in self.model.model_history.keys():
                    continue
                fig = plt.figure()
                plt.plot(self.model.model_history[key], "b")
                plt.plot(self.model.model_history["val_" + key], "r")
                plt.title(f"{key} during training")
                plt.xlabel(f"Epoch Number")
                plt.ylabel(f"{key}")
                plt.legend(["train", "validation"])
                fig_name = os.path.join(os.getcwd(), f"{self.figure_dir}", f"{key}.png")
                fig_names[key] = fig_name
                fig.savefig(fig_name)
        return fig_names

    def plot_permutation_importance(
        self, X: pd.DataFrame, y: Union[List, pd.Series], verbose=True
    ) -> plt.figure:
        raise NotImplementedError("This method has not been implemented in this class")

    def predict_proba(self, x: Union[List, np.array]) -> np.array:
        """
        Predict the probability of belonging to a particular class so that the output is suitable for binary classification and consistent with other sklearn format models.

        Args:
            x (Union[List, np.array]): Array of payloads to score with the model.

        Returns:
            np.array: List of arrays with the binary class confidences
        """
        return np.asarray([[1 - score, score] for score in self.model.predict(x)])

    def predict(self, x):
        """
        Predict the probability of belonging to a particular class so that the output is suitable for binary classification and consistent with other sklearn format models.
        Args:
            x (Union[List, np.array]): Array of payloads to score with the model.

        Returns:
            np.array: List of binary predictions for each payload
        """
        return np.asarray(self.model.predict(x)).round()
    
    def _save_model_to_folder(self, mlflow: ModuleType, model: object, model_name: str, input_example: object = None) -> ModelInfo:
        raise NotImplementedError("This method has not been implemented in this class")
    
    def load_model_from_folder(self, model_uri: str) -> object:
        raise NotImplementedError("This method has not been implemented in this class")


from types import ModuleType
from typing import Union, Tuple, List
from mlflow.models.model import ModelInfo

import pandas as pd
import matplotlib.pyplot as plt


from datascience_core.data_retrieval import SchemaFetcher
from sklearn.inspection import permutation_importance
import numpy as np
import xgboost as xgb
from datascience_core.modelling._base import IModelEvaluator
from datascience_core.modelling.models._base import IModel, IModelSavingLoadingAttribute


class XGBoostClassifierPipeline(IModel, IModelSavingLoadingAttribute):
    """A wrapper class for XGBoost classifier.  This wrapper expands on the existing catboost class by adding
    an evaluator object and functions to run cross validation, permutation importance and feature importance.  The class
    is intsantiated the same was as a catboost classifier.  The evaluator should be instantiated separately and passed as an input
    """

    model: xgb.Booster
    def __init__(
        self,
        evaluator: IModelEvaluator = None,
        model: xgb.Booster = None,
        **kw
    ):
        self.evaluator = evaluator
        if model is not None:
            self.model = model
        else:
            self.model = xgb.Booster(**kw)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: dict = {},
        *args,
        **kwargs,
    ):
        """
        Fits an XGBoost model using their native API and saves it to the booster attribute. It forwards any arugments of the xgb.train api.
        """
        dtrain = xgb.DMatrix(data=X_train, label=y_train, enable_categorical=True)
        dval = xgb.DMatrix(data=X_val, label=y_val, enable_categorical=True)
        self.booster = xgb.train(params, dtrain, evals=[(dval, "val")], *args, **kwargs)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5):
        output = np.where(self.predict_proba(X) > threshold, 1, 0)
        return output

    def predict_proba(self, X):
        X = xgb.DMatrix(X, enable_categorical=True)
        output = self.booster.predict(X)
        return np.swapaxes(np.array([output, output]), 0, 1)

    def run_cross_validation(
        self,
        X: pd.DataFrame,
        y: Union[List, pd.Series],
        **kwargs,
    ) -> plt.figure:
        """Function to run cross validation on a dataset.

        Args:
            X (pd.DataFrame): Features
            y (Union[List,pd.Series]): truth label
            cv_folds (int, optional): Number of folds to perform. Defaults to None.

        Returns:
            plt.figure: corss validation results figure
        """

        raise NotImplementedError("This has not yet been implemented for this class.")

    def evaluate_model(
        self, X: pd.DataFrame, y_true: pd.Series, title:str ="Model Performance", verbose: bool = True
    ) -> dict:
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
            return self.evaluator.run(X, y_true, title=title,verbose=verbose)
        else:
            raise AttributeError("No DSModelEvaluator set for this class")

    def plot_permutation_importance(
        self, X: pd.DataFrame, y: Union[List, pd.Series], verbose=True
    ) -> plt.figure:
        """Calculates the permutation importance if 50 or fewer features are used.  Above this it takes ages

        Args:
            X (pd.DataFrame): Features
            y (Union[List,pd.Series]): Truth label
            verbose (bool): print outputs

        Returns:
            plt.figure: plot of permutation importances
        """
        raise NotImplementedError(
            "This function is not yet implemented for this class."
        )

    def plot_feature_importance(
        self,
        X: pd.DataFrame,
        y: Union[List, pd.Series],
        top_n_features: int = 50,
        verbose=True,
    ) -> plt.figure:
        """Calculates the feature importance of the top n features

        Args:
            X (pd.DataFrame): Features
            y (Union[List,pd.Series]): Truth label
            top_n_features (int, optional): number of features to include. Defaults to 50.
            verbose (bool): print outputs

        Returns:
            plt.figure:: figure of feature importances
        """
        raise NotImplementedError(
            "This functionality is not yet implemented for this class"
        )
    
    def _save_model_to_folder(self, mlflow: ModuleType, model: object, model_name: str, input_example: object = None) -> ModelInfo:
        raise NotImplementedError("This functionality is not yet implemented for this class")
    
    def load_model_from_folder(self, run_name: str) -> object:
        raise NotImplementedError("This functionality is not yet implemented for this class")

    
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from lightgbm import LGBMClassifier, cv, Dataset
from mlflow.models.model import ModelInfo
import mlflow

import os
import shutil
from typing import Union, List
from types import ModuleType
from pathlib import Path

from datadayessentials.config import LocalConfig
from datadayessentials.data_retrieval import SchemaFetcher
from datadayessentials.modelling.models._base import (
    IModel,
    IModelSavingLoadingAttribute,
)
from datadayessentials.modelling._base import IModelEvaluator


class lightgbmClassifierPipeline(IModel, IModelSavingLoadingAttribute):
    """A wrapper class for catboost classifier.  This wrapper expands on the existing catboost class by adding
    an evaluator object and functions to run cross validation, permutation importance and feature importance.  The class
    is intsantiated the same was as a catboost classifier.  The evaluator should be instantiated separately and passed as an input

    Example Use Case:
    '''python

    from lightgbm import LGBMClassifier
    from datadayessentials.models import lightgbmClassifierPipeline
    from datadayessentials.utils import ModelEvaluator
    from datadayessentials.data_transformation import ValueReplacer, CatTypeCoverter, DataFramePipe

    import pandas as pd

    X = pd.DataFrame({"col1":[0,1,2,3,4,5],"col2":[0,1,2,3,4,5],"col3":[0,1,2,3,4,5],"col4":[0,1,2,3,4,5],'y_true':[1,1,0,1,0,1]})
    y = X.pop('y_true')

    step1 = ValueReplacer(unwanted_values=[
                    "M",
                    "C",
                    "{ND}"
                ])
    catboost_preprocessor = DataFramePipe([step1])

    evaluator = ModelEvaluator()

    pipeline = lightgbmClassifierPipeline(
        preprocessor=catboost_preprocessor,
        evaluator=evaluator,
        iterations= 5,
        random_seed=42,
        task_type='CPU',
        logging_level='Silent',

        )

    X_processed = pipeline.preprocess(data=X)
    pipeline.fit(X=X_processed,y=y)
    metrics = pipeline.evaluate(X=X, y_true=y)
    '''
    """

    model: LGBMClassifier

    def __init__(
        self,
        evaluator: IModelEvaluator = None,
        model: LGBMClassifier = None,
        **kw,
    ):
        if model is not None:
            self.model = model
        else:
            self.model = LGBMClassifier(**kw)
        self.evaluator = evaluator

    def fit(self, X, y, *args, **kwargs):
        self.model.fit(X, y, *args, **kwargs)

    def predict(
        self,
        X,
        raw_score=False,
        start_iteration=0,
        num_iteration=None,
        pred_leaf=False,
        pred_contrib=False,
        validate_features=False,
        **kwargs,
    ):
        return self.predict(
            X,
            raw_score=False,
            start_iteration=0,
            num_iteration=None,
            pred_leaf=False,
            pred_contrib=False,
            validate_features=False,
            **kwargs,
        )

    def predict_proba(
        self,
        X,
        raw_score=False,
        start_iteration=0,
        num_iteration=None,
        pred_leaf=False,
        pred_contrib=False,
        validate_features=False,
        **kwargs,
    ):
        return self.model.predict_proba(
            X,
            raw_score,
            start_iteration,
            num_iteration,
            pred_leaf,
            pred_contrib,
            validate_features,
            **kwargs,
        )

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

        params = self.model.get_params()
        if "cat_features" in params.keys():
            cv_dataset = Dataset(
                data=X, label=y, cat_feature=self.model.get_params()["cat_features"]
            )
            params.pop("cat_features")
        else:
            cv_dataset = Dataset(data=X, label=y)

        if "loss_function" not in params.keys():
            params["loss_function"] = "Logloss"
            print(
                "loss function has not been specified for cross validation.  I'll default to Logloss"
            )

        scores = cv(cv_dataset, params, **kwargs)

        fig = plt.figure()
        plt.plot(
            scores["iterations"], scores[["test-Logloss-mean", "train-Logloss-mean"]]
        )
        plt.fill_between(
            scores["iterations"],
            scores["test-Logloss-mean"] - scores["test-Logloss-std"],
            scores["test-Logloss-mean"] + scores["test-Logloss-std"],
            alpha=0.5,
        )
        plt.fill_between(
            scores["iterations"],
            scores["train-Logloss-mean"] - scores["train-Logloss-std"],
            scores["train-Logloss-mean"] + scores["train-Logloss-std"],
            alpha=0.5,
        )
        plt.legend(["test", "train"])
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Cross Validation performance")

        return fig

    def evaluate_model(
        self, X: pd.DataFrame, y_true: pd.Series, verbose: bool = True
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
            self.evaluator.set_model(model=self.model)
            return self.evaluator.run(X, y_true, verbose=verbose)
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
        if len(self.model.feature_names_) < 51:
            fetcher = SchemaFetcher()
            app_and_cra_payload_schema = fetcher.get_schema("app_and_cra_payload")
            missing_keys = []

            for each_col in X.columns:
                if each_col not in app_and_cra_payload_schema.keys():
                    app_and_cra_payload_schema[each_col] = {
                        "description": "MISSNG - ADD TO SCHEMA",
                        "unique_categories": ["C", "M"],
                        "is_date": False,
                        "min_val": "01",
                        "max_val": "99",
                        "dtype": "str",
                    }
                    missing_keys.append(each_col)

            if missing_keys:
                print(f"schema is missing: {missing_keys}")

            result = permutation_importance(
                self, X, y, n_repeats=10, random_state=42, n_jobs=2
            )

            # feature importance
            importances = pd.DataFrame(
                {
                    "feature_name": self.model.feature_names_,
                    "feature_importance": self.model.feature_importances_,
                }
            )
            importances = importances.sort_values(
                by="feature_importance", ascending=False
            ).head(50)
            importances["description"] = (
                importances.apply(
                    lambda x: app_and_cra_payload_schema[x["feature_name"]][
                        "description"
                    ],
                    axis=1,
                )
                + ":"
                + " " * 15
                + importances["feature_name"]
            )

            df_permutation = pd.DataFrame(
                {
                    "feature_name": X.columns,
                    "permutation_mean": result["importances_mean"],
                    "permutation_std": result["importances_std"],
                }
            )
            df_permutation["description"] = (
                df_permutation.apply(
                    lambda x: app_and_cra_payload_schema[x["feature_name"]][
                        "description"
                    ],
                    axis=1,
                )
                + ":"
                + " " * 15
                + importances["feature_name"]
            )
            permutation_importance_fig = df_permutation.sort_values(
                by="permutation_mean", ascending=False
            ).plot(
                kind="barh",
                x="description",
                y="permutation_mean",
                yerr="permutation_std",
                figsize=(20, 10),
            )
            plt.xlabel("Mean decrease in impurity")
            plt.title("Permutation importance")
            permutation_importance_fig = permutation_importance_fig.get_figure()
        else:
            print("Too many features for permutation importance.")
            permutation_importance_fig = None

        return permutation_importance_fig

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
        fetcher = SchemaFetcher()
        app_and_cra_payload_schema = fetcher.get_schema("app_and_cra_payload")
        missing_keys = []

        for each_col in X.columns:
            if each_col not in app_and_cra_payload_schema.keys():
                app_and_cra_payload_schema[each_col] = {
                    "description": "MISSNG - ADD TO SCHEMA",
                    "unique_categories": ["C", "M"],
                    "is_date": False,
                    "min_val": "01",
                    "max_val": "99",
                    "dtype": "str",
                }
                missing_keys.append(each_col)

        print(f"schema is missing: {missing_keys}")

        # feature importance
        importances = pd.DataFrame(
            {
                "feature_name": self.model.feature_names_,
                "feature_importance": self.model.feature_importances_,
            }
        )
        importances = importances.sort_values(
            by="feature_importance", ascending=False
        ).head(top_n_features)
        importances["description"] = (
            importances.apply(
                lambda x: app_and_cra_payload_schema[x["feature_name"]]["description"],
                axis=1,
            )
            + ":"
            + " " * 15
            + importances["feature_name"]
        )

        feature_importance_fig = importances.sort_values(
            by="feature_importance", ascending=True
        ).plot(kind="barh", x="description", y="feature_importance", figsize=(20, 10))
        plt.xlabel("importance")
        plt.title("Feature importance")

        return feature_importance_fig.get_figure()
        # feature_importance_fig.savefig("feature_importance.png")

    def _save_model_to_folder(
        self, model_save_path: str, input_example: object = None
    ) -> ModelInfo:
        """Logs the model to the MLFlow server, if an input example is passed then infer the model signature.

        Args:
            mlflow (ModuleType): MLFlow module
            model_save_path (str): path in the mlflow file sytem to save the model files
            input_example (object, optional): model input example, used to infer the model signature for saving to mlflow
        Returns:
            ModelInfo: info object that contains the model URI and other details about the model
        """
        if input_example is not None:
            prediction = self.predict(input_example)
            signature = mlflow.models.signature.infer_signature(
                input_example, prediction
            )
            meta_data = mlflow.catboost.save_model(
                self.model, model_save_path, signature=signature
            )
        else:
            meta_data = mlflow.catboost.save_model(
                self.model,
                model_save_path,
            )
        return meta_data

    @classmethod
    def load_model_from_folder(cls, model_folder):
        model = mlflow.catboost.load_model(model_folder)
        return cls(model=model)

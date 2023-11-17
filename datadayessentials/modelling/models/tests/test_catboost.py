import pytest
import unittest.mock as mock
from catboost import cv
import pandas as pd
import os

from datadayessentials.modelling.experiment_manager import ExperimentManager

from ..catboost import CatBoostClassifierPipeline
from ...model_evaluator import ModelEvaluator
from catboost import cv, Pool
import matplotlib.pyplot as plt
import mlflow

from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication, InteractiveLoginAuthentication
from datadayessentials.config import Config

@pytest.fixture
def model_params():
    return {
        "iterations": 10,
        "random_seed": 42,
        "task_type": "CPU",
        "loss_function": "Logloss",
        "logging_level": "Silent",
        "l2_leaf_reg": 8,
        "random_strength": 0.6,
        "subsample": 0.8,
        "max_depth": 5,
        "learning_rate": 0.03,
        "early_stopping_rounds": 20,
        "custom_metric": ["NormalizedGini", "Logloss"],
    }


@pytest.fixture
def X():
    return pd.DataFrame(
        {
            "col1": [0, 1, 2, 3, 4, 5],
            "col2": [0, 1, 2, 3, 4, 5],
            "col3": [0, 1, 2, 3, 4, 5],
            "col4": ["zero", "one", "two", "three", "four", "five"],
            "Target": [0, 1, 1, 1, 0, 0],
        }
    )


class TestCatBoostClassifierPipeline:
    @mock.patch("datadayessentials.modelling.models.catboost.cv")
    def test_run_cross_validation(self, mock_cv_run, model_params, X):
        mock_cv_run.return_value = pd.DataFrame(
            {
                "iterations": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "train-Logloss-mean": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "train-Logloss-std": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "test-Logloss-mean": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "test-Logloss-std": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            }
        )
        y = X.pop("Target")

        pipe = CatBoostClassifierPipeline(cat_features=["col4"], **model_params)

        fig = pipe.run_cross_validation(
            X=X, y=y, fold_count=2, early_stopping_rounds=10, plot=True
        )
        assert type(fig) is type(plt.figure())
        mock_cv_run.assert_called()

    def test_evaluate_model(self, model_params, X):
        evaluator = ModelEvaluator()
        pipe = CatBoostClassifierPipeline(
            evaluator=evaluator, cat_features=["col4"], **model_params
        )
        y = X.pop("Target")
        pipe.fit(X=X, y=y)

        evaluation = pipe.evaluate_model(X, y, verbose=False)

        assert isinstance(evaluation, dict)
        assert set(list(evaluation.keys())) == set(
            ["model_metrics", "model_performance_figure"]
        )
        
        

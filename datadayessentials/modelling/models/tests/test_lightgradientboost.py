import pytest
import unittest.mock as mock
from catboost import cv
import pandas as pd
import numpy as np
import os

from datadayessentials.modelling.experiment_manager import ExperimentManager

from ..lightgradientboost import lightgbmClassifierPipeline
from ...model_evaluator import ModelEvaluator
from lightgbm import cv, Dataset
import matplotlib.pyplot as plt
import mlflow

from azureml.core import Workspace
from azureml.core.authentication import (
    ServicePrincipalAuthentication,
    InteractiveLoginAuthentication,
)
from datadayessentials.config import Config


@pytest.fixture
def model_params():
    return {
        "n_estimators": 100,
        "random_seed": 42,
        "l2_leaf_reg": 8,
        "subsample": 0.8,
        "max_depth": 5,
        "learning_rate": 0.03,
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


class TestLightGradientBoostPipeline:
    def test_init(self, model_params):
        pipe = lightgbmClassifierPipeline(**model_params)

        assert np.in1d(
            list(model_params.keys()), list(pipe.model.get_params().keys())
        ).all()

    def test_fit(self, model_params, X):
        pipe = lightgbmClassifierPipeline(**model_params)
        y = X.pop("Target")
        X["col4"] = X["col4"].astype("category")
        pipe.fit(X=X, y=y)
        assert pipe.model is not None

    @mock.patch("datadayessentials.modelling.models.lightgradientboost.cv")
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
        X["col4"] = X["col4"].astype("category")

        pipe = lightgbmClassifierPipeline(**model_params)

        fig = pipe.run_cross_validation(
            X=X, y=y, fold_count=2, early_stopping_rounds=10, plot=True
        )
        assert type(fig) is type(plt.figure())
        mock_cv_run.assert_called()

    def test_evaluate_model(self, model_params, X):
        evaluator = ModelEvaluator()
        pipe = lightgbmClassifierPipeline(evaluator=evaluator, **model_params)
        y = X.pop("Target")
        X["col4"] = X["col4"].astype("category")
        pipe.fit(X=X, y=y)

        evaluation = pipe.evaluate_model(X, y, verbose=False)

        assert isinstance(evaluation, dict)
        assert set(list(evaluation.keys())) == set(
            ["model_metrics", "model_performance_figure"]
        )

    def test_predict(self, model_params, X):
        pipe = lightgbmClassifierPipeline(**model_params)
        y = X.pop("Target")
        X["col4"] = X["col4"].astype("category")
        pipe.fit(X=X, y=y)

        predictions = pipe.predict(X=X)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (6,)

    def test_predict_proba(self, model_params, X):
        pipe = lightgbmClassifierPipeline(**model_params)
        y = X.pop("Target")
        X["col4"] = X["col4"].astype("category")
        pipe.fit(X=X, y=y)

        predictions = pipe.predict_proba(X=X)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (6, 2)

    def test_no_categorical_type_raises_valueerror(self, model_params, X):
        pipe = lightgbmClassifierPipeline(**model_params)
        y = X.pop("Target")

        with pytest.raises(ValueError):
            pipe.fit(X=X, y=y)

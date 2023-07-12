import sys
from types import ModuleType
import unittest
from mlflow.models.model import ModelInfo

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from datadayessentials.modelling.models.catboost import CatBoostClassifierPipeline

from ..experiment_manager import ExperimentManager
from azureml.core import Workspace, Experiment
import datetime
import mlflow
from datadayessentials.config import LocalConfig


def setup_experiment_manager():

    raise ValueError
    manager = ExperimentManager(
        experiment_name="test_datadayessentials"
    )
    return manager

def setup_workspace():
    env_config = LocalConfig().get_environment_from_name("dev")
    ws = Workspace.get(
        name=env_config["machine_learning_workspace"],
        subscription_id=env_config["subscription_id"],
        resource_group=env_config["resource_group"],
    )
    return ws


class TestExperimentManager(unittest.TestCase):
    def test_submit_run(self):
        experiment_manager = setup_experiment_manager()

        model_metrics = {
            "metrics": {
                "train": {
                    "OK_float": 1.23,
                },
                "validate": {
                    "OK_float": 1.23,
                },
                "test": {
                    "OK_float": 1.23,
                },
            },
        }
        params = {
            'iterations': 1
        }

        X = pd.DataFrame(
            {
                "col1": [0, 1, 2, 3, 4, 5],
                "col2": [0, 1, 2, 3, 4, 5],
                "col3": [0, 1, 2, 3, 4, 5],
                "col4": ["zero", "one", "two", "three", "four", "five"],
                "Target": [0, 1, 1, 1, 0, 0],
            }
        )
        y = X.pop("Target")
        
        versioned_meta_data = {"test_data": 2}

        test_model = CatBoostClassifierPipeline(cat_features=['col4'], **params)
        test_model.fit(X, y)
        run_id, run_name = experiment_manager.submit_run(
            versioned_meta_data,
            test_model,
            run_name='test_run_name',
            train_model_metrics=model_metrics['metrics']['train'],
            validate_model_metrics=model_metrics['metrics']['validate'],
            test_model_metrics=model_metrics['metrics']['test'],
            params=params
        )

        ws = setup_workspace()

        run = ws.get_run(run_id)
        assert run.get_metrics()["train_OK_float"] == 1.23
        assert run.get_metrics()["validate_OK_float"] == 1.23
        assert run.get_metrics()["test_OK_float"] == 1.23

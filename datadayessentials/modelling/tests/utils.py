from ..experiment_manager import ExperimentManager
import pandas as pd
from datadayessentials.modelling.models.catboost import CatBoostClassifierPipeline

from ..experiment_manager import ExperimentManager


def setup_experiment_manager():
    manager = ExperimentManager(experiment_name="test_datadayessentials")
    return manager


def trigger_test_run():
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
    params = {"iterations": 1}

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

    test_model = CatBoostClassifierPipeline(cat_features=["col4"], **params)
    test_model.fit(X, y)
    run_id, run_name = experiment_manager.submit_run(
        versioned_meta_data,
        test_model,
        run_name="test_run_name",
        train_model_metrics=model_metrics["metrics"]["train"],
        validate_model_metrics=model_metrics["metrics"]["validate"],
        test_model_metrics=model_metrics["metrics"]["test"],
        params=params,
    )

    ws = experiment_manager.workspace

    run = ws.get_run(run_id)
    return run_id, run

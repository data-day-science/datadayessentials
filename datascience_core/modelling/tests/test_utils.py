import pytest
import unittest.mock as mock
from ..utils import DataSplitter
from ..model_evaluator import ModelEvaluator
import pytest
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt


@pytest.fixture
def X():
    return pd.DataFrame(
        {
            "col1": list(range(0, 120)),
            "col2": list(range(0, 120)),
            "col3": list(range(0, 120)),
            "col4": [
                "zero",
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
            ]
            * 12,
            "Target": [0, 0, 0, 1, 0, 1, 0, 1, 0, 1] * 12,
        }
    )


class TestDataSplitter:
    def test_train_val_test_split_same_dist(self, X):
        splitter = DataSplitter()
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.train_val_test_split(
            data=X,
            target_list=["Target"],
            validation_fraction=0.2,
            num_test_rows=20,
            validation_is_holdout=False,
        )

        assert X_train.shape[0] == 80
        assert X_train.shape[1] == 4
        assert len(y_train) == 80

        assert X_val.shape[0] == 20
        assert X_val.shape[1] == 4
        assert len(y_val) == 20

        assert X_test.shape[0] == 20
        assert X_test.shape[1] == 4
        assert len(y_test) == 20

        X_train, X_val, X_test, y_train, y_val, y_test = splitter.train_val_test_split(
            data=X,
            target_list=["Target"],
            validation_fraction=0.2,
            num_test_rows=0,
        )
        assert X_test.empty
        assert y_test.empty

        X_train, X_val, X_test, y_train, y_val, y_test = splitter.train_val_test_split(
            data=X,
            target_list=["Target"],
            validation_fraction=0.2,
            num_test_rows=None,
        )
        assert X_test.empty
        assert y_test.empty

    def test_train_val_test_split_holdout_val(self, X):
        splitter = DataSplitter()
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.train_val_test_split(
            data=X,
            target_list=["Target"],
            validation_fraction=0.2,
            num_test_rows=20,
            validation_is_holdout=True,
        )

        assert X_val.iloc[0]["col1"] == 80
        assert X_val.iloc[19]["col3"] == 99

    def test_validation_faction_variants_passes(self, X):
        splitter = DataSplitter()

        # fraction is 0, holdout False
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.train_val_test_split(
            data=X,
            target_list=["Target"],
            validation_fraction=0,
            num_test_rows=20,
            validation_is_holdout=False,
        )

        assert X_train.shape[0] == 100
        assert X_val.shape[0] == 0
        assert X_test.shape[0] == 20

        # fraction is None, holdout False
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.train_val_test_split(
            data=X,
            target_list=["Target"],
            validation_fraction=None,
            num_test_rows=20,
            validation_is_holdout=False,
        )

        assert X_train.shape[0] == 100
        assert X_val.shape[0] == 0
        assert X_test.shape[0] == 20

        # fraction is 0, holdout True
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.train_val_test_split(
            data=X,
            target_list=["Target"],
            validation_fraction=0,
            num_test_rows=20,
            validation_is_holdout=True,
        )

        assert X_train.shape[0] == 100
        assert X_val.shape[0] == 0
        assert X_test.shape[0] == 20

        # fraction is None, holdout True
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.train_val_test_split(
            data=X,
            target_list=["Target"],
            validation_fraction=None,
            num_test_rows=20,
            validation_is_holdout=True,
        )

        assert X_train.shape[0] == 100
        assert X_val.shape[0] == 0
        assert X_test.shape[0] == 20

    def test_test_set_variants_passes(self, X):
        splitter = DataSplitter()

        # test_set is 0, holdout is False
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.train_val_test_split(
            data=X,
            target_list=["Target"],
            validation_fraction=0.2,
            num_test_rows=0,
            validation_is_holdout=False,
        )

        assert X_train.shape[0] == 96
        assert X_val.shape[0] == 24
        assert X_test.shape[0] == 0

        # test_set is None, holdout is False
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.train_val_test_split(
            data=X,
            target_list=["Target"],
            validation_fraction=0.2,
            num_test_rows=None,
            validation_is_holdout=False,
        )

        assert X_train.shape[0] == 96
        assert X_val.shape[0] == 24
        assert X_test.shape[0] == 0

        # test_set is 0, holdout is True
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.train_val_test_split(
            data=X,
            target_list=["Target"],
            validation_fraction=0.2,
            num_test_rows=0,
            validation_is_holdout=True,
        )

        assert X_train.shape[0] == 96
        assert X_val.shape[0] == 24
        assert X_test.shape[0] == 0

        # test_set is None, holdout is True
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.train_val_test_split(
            data=X,
            target_list=["Target"],
            validation_fraction=0.2,
            num_test_rows=None,
            validation_is_holdout=True,
        )

        assert X_train.shape[0] == 96
        assert X_val.shape[0] == 24
        assert X_test.shape[0] == 0

    def test_invalid_input_fails(self, X):
        splitter = DataSplitter()
        with pytest.raises(ValueError) as err:
            splitter.train_val_test_split(
                data=X[["Target"]],
                target_list=["Target"],
                validation_fraction=0.2,
                num_test_rows=2,
            )
            assert (
                err.value
                == "Data contains only 1 feature. Splitter needs features and a target column"
            )

        with pytest.raises(ValueError) as err:
            splitter.train_val_test_split(
                data=X,
                target_list=["Not_the_target"],
                validation_fraction=0.2,
                num_test_rows=2,
            )
            assert err.value == "Target column not found in the data"

    def test_resample_data(self, X):
        splitter = DataSplitter()
        data_out = splitter._resample_data(
            data=X, minority_samples=50, majority_samples=50, target="Target"
        )
        assert data_out.shape[0] == 100
        assert data_out["Target"].value_counts()[0] == 50
        assert data_out["Target"].value_counts()[1] == 50

        data_out = splitter._resample_data(
            data=X, minority_samples=100, majority_samples=50, target="Target"
        )
        assert data_out.shape[0] == 150
        assert data_out[data_out["Target"] == 1].shape[0] == 100
        assert data_out[data_out["Target"] == 0].shape[0] == 50

        data_out = splitter._resample_data(
            data=X, minority_samples=50, majority_samples=100, target="Target"
        )
        assert data_out.shape[0] == 150
        assert data_out[data_out["Target"] == 1].shape[0] == 50
        assert data_out[data_out["Target"] == 0].shape[0] == 100

    def test_no_sample_number_breaks_resample_data(self, X):
        splitter = DataSplitter()
        with pytest.raises(ValueError) as err:
            data_out = splitter._resample_data(
                data=X, minority_samples=0, majority_samples=50, target="Target"
            )
            assert (
                err.value
                == "Both minority and majority samples have to be greater than 0"
            )

        with pytest.raises(ValueError) as err:
            data_out = splitter._resample_data(
                data=X, minority_samples=100, majority_samples=0, target="Target"
            )
            assert (
                err.value
                == "Both minority and majority samples have to be greater than 0"
            )

    def test_balance_dataset(self, X):
        y = X.pop("Target")
        splitter = DataSplitter()
        X_out, y_out = splitter.balance_dataset(
            X=X, y=y, total_samples=10, majority_class_fraction=0.5
        )

        assert X_out.shape[0] == 10
        assert y_out.sum() == 5


class TestModelEvaluator:
    @mock.patch(
        "catboost.CatBoostClassifier.predict_proba",
        return_value=np.array(
            [
                [0.69, 0.31],
                [0.85, 0.15],
                [0.44, 0.56],
                [0.1, 0.9],
                [0.8, 0.2],
                [0.3, 0.7],
            ]
        ),
    )
    def test_make_prediction(self, mock_model_predict):
        data = pd.DataFrame(
            {
                "f1": [0, 1, 2, 3, 4, 5],
                "f2": [0, 1, 2, 3, 4, 5],
                "f3": [0, 1, 2, 3, 4, 5],
                "y": [0, 1, 0, 1, 0, 1],
            }
        )
        model = CatBoostClassifier()
        X = data
        y = X.pop("y")
        model.fit(X, y, verbose=False)
        evaluator = ModelEvaluator(model=model)
        y_proba = evaluator.get_probas(X)
        y_pred = evaluator.make_predictions(y_proba)
        assert set(y_pred) == set([0, 0, 1, 1, 0, 1])

    @mock.patch(
        "catboost.CatBoostClassifier.predict_proba",
        return_value=np.array(
            [
                [0.69, 0.31],
                [0.85, 0.15],
                [0.44, 0.56],
                [0.1, 0.9],
                [0.8, 0.2],
                [0.3, 0.7],
            ]
        ),
    )
    def test_get_probas(self, mock_model_predict):
        data = pd.DataFrame(
            {
                "f1": [0, 1, 2, 3, 4, 5],
                "f2": [0, 1, 2, 3, 4, 5],
                "f3": [0, 1, 2, 3, 4, 5],
                "y": [0, 1, 0, 1, 0, 1],
            }
        )
        model = CatBoostClassifier()
        X = data
        y = X.pop("y")
        model.fit(X, y)
        evaluator = ModelEvaluator(model=model)
        y_pred = evaluator.get_probas(X)
        assert list(y_pred) == [0.31, 0.15, 0.56, 0.9, 0.2, 0.7]

    def test_calculate_metrics(self):
        evaluator = ModelEvaluator()

        y = list(np.zeros(50)) + list(np.ones(50))
        y_proba = list(np.ones(25) * 0.1) + list(np.ones(75) * 0.9)
        y_pred = list(np.zeros(25)) + list(np.ones(75))

        actual = evaluator.calculate_metrics(y_pred=y_pred, y_proba=y_proba, y=y)

        expected = {
            "precision": 0.6666666666666666,
            "recall": 1.0,
            "f1": 0.8,
            "roc_auc": 0.75,
            "gini": 0.5,
            "confusion matrix": np.array([[25, 25], [0, 50]]),
            "false positive rates": np.array([0.0, 0.5, 1.0]),
            "true positive rates": np.array([0.0, 1.0, 1.0]),
        }

        assert actual["precision"] == expected["precision"]
        assert actual["recall"] == expected["recall"]
        assert actual["f1"] == expected["f1"]
        assert actual["roc_auc"] == expected["roc_auc"]
        assert actual["gini"] == expected["gini"]
        assert actual["confusion matrix"].all() == expected["confusion matrix"].all()
        assert (
            actual["false positive rates"].all()
            == expected["false positive rates"].all()
        )
        assert (
            actual["true positive rates"].all() == expected["true positive rates"].all()
        )

    def test_run(self):
        params = {
            "iterations": 10,
            "random_seed": 42,
            "task_type": "CPU",
            "loss_function": "Logloss",
            "logging_level": "Silent",
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
        model = CatBoostClassifier(cat_features=["col4"], **params)
        model.fit(X, y)
        evaluator = ModelEvaluator(model=model)

        dict_out = evaluator.run(X, y, verbose=False)
        assert "model_performance_figure" in dict_out.keys()
        assert "model_metrics" in dict_out.keys()

        assert dict_out["model_performance_figure"] is None
        assert isinstance(dict_out["model_metrics"], dict)


class TestModelPipeline:
    """The indvidual components of this orchestrator have been tested.  This test is an end to end test.

    TO DO at the end of the project once the orchestrator design is decided upon
    """

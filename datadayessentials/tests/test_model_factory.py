
import pandas as pd
import unittest
from datadayessentials.modelling.experiment_manager import ExperimentManager
from datadayessentials.modelling.model_manager import ModelManager
from datadayessentials.modelling.models import CatBoostClassifierPipeline, ModelFactory
from datadayessentials.config import LocalConfig

class TestModelFactory(unittest.TestCase):
    """
    This is an integration test that tests the model factory class, along with the
    experiment manager and model manager classes.

    It creates a model, submits it as a run to azure and then registers it as a model
     in azure. It then uses the model factory to reload these models into their original
     state.
    """ 
    @classmethod
    def setUpClass(cls):
        """
        Create a dummy model, submit it as a run to azure and then register it  as a model in azure."""
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
        model_params = {
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

        cls.test_model = CatBoostClassifierPipeline(cat_features=["col4"], **model_params)
        cls.test_model.fit(X, y)

        environment_config = LocalConfig().get_environment()

        experiment_manager = ExperimentManager('test_experiment_name')   

        cls.run_id, run_name = experiment_manager.submit_run(
            {'dataset_name': 'test_dataset_name'},
            cls.test_model,
            run_name='test_run_name',
            train_model_metrics=model_metrics['metrics']['train'],
            validate_model_metrics=model_metrics['metrics']['validate'],
            test_model_metrics=model_metrics['metrics']['test'],
            params=model_params
        )
        
        model_manager = ModelManager()
        
        cls.registered_model_name = 'test_model'
        model_info = model_manager.register_model_from_run_id(cls.run_id, cls.registered_model_name, "model")
        cls.model_version = model_info.version

    def test_create_from_run_id(self):
        
        model_factory = ModelFactory(CatBoostClassifierPipeline)
        
        loaded_model = model_factory.create_from_run_id(self.run_id)

        assert loaded_model.model.feature_names_ == self.test_model.model.feature_names_
    
    def test_create_from_registered_model(self):
        model_factory = ModelFactory(CatBoostClassifierPipeline)
        
        loaded_model = model_factory.create_from_registered_model(self.registered_model_name, self.model_version)

        assert loaded_model.model.feature_names_ == self.test_model.model.feature_names_

    def test_create_from_registered_model_no_version(self):
        model_factory = ModelFactory(CatBoostClassifierPipeline)
        
        loaded_model = model_factory.create_from_registered_model(self.registered_model_name)

        assert loaded_model.model.feature_names_ == self.test_model.model.feature_names_
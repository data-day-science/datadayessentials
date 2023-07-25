import unittest
from ..model_factory import ModelFactory


class TestGetLatestRunIdFromExperiment(unittest.TestCase):
    def test_get_latest_run_id_from_experiment_single_run(self):
        model_factory = ModelFactory("placeholder_pipeline")
        latest_run_id = model_factory.get_latest_run_id_from_experiment(
            "Ratesetter_auto_ml_run"
        )
        assert isinstance(latest_run_id, str)

    def test_get_latest_run_id_from_false_experiment(self):
        model_factory = ModelFactory("placeholder_pipeline")

        with self.assertRaises(Exception):
            latest_run_id = model_factory.get_latest_run_id_from_experiment(
                "IM NOT AN EXPERIMENT"
            )

    def test_get_latest_run_id_from_none(self):
        model_factory = ModelFactory("placeholder_pipeline")

        with self.assertRaises(Exception):
            latest_run_id = model_factory.get_latest_run_id_from_experiment(None)

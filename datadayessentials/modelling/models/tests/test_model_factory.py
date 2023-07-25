import unittest
from ..model_factory import ModelFactory


class TestGetLatestRunIdFromExperiment(unittest.TestCase):
    def test_get_latest_run_id_from_experiment_single_run(self):
        latest_run_id = ModelFactory.get_latest_run_id_from_experiment(
            "Ratesetter_auto_ml_run"
        )
        print(latest_run_id, "********")
        assert isinstance(latest_run_id, str)

    def test_get_latest_run_id_from_false_experiment(self):
        with self.assertRaises(Exception):
            latest_run_id = ModelFactory.get_latest_run_id_from_experiment(
                "IM NOT AN EXPERIMENT"
            )

    def test_get_latest_run_id_from_none(self):
        with self.assertRaises(Exception):
            latest_run_id = ModelFactory.get_latest_run_id_from_experiment(None)

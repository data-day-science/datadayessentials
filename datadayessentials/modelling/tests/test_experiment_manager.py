import sys
from types import ModuleType
import unittest
from mlflow.models.model import ModelInfo

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


import datetime
import mlflow
from datadayessentials.config import Config, ExecutionEnvironmentManager
from datadayessentials.modelling.tests.utils import trigger_test_run


class TestExperimentManager(unittest.TestCase):
    def test_submit_run(self):
        run_id, run = trigger_test_run()

        assert run.get_metrics()["train_OK_float"] == 1.23
        assert run.get_metrics()["validate_OK_float"] == 1.23
        assert run.get_metrics()["test_OK_float"] == 1.23

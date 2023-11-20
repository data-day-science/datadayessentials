#add imports
from ..model_inference import InferenceModel, lightgbmInferenceModel
import unittest
from unittest.mock import patch, MagicMock
import pytest
from lightgbm import LGBMClassifier


class TestInferenceModel(unittest.TestCase):
    def setUp(self):
        self.model = MagicMock()
        self.model.model = MagicMock()
        self.model.feature_names_ = ["a", "b", "c"]
        self.model.feature_importances_ = [0.1, 0.2, 0.3]
        self.inference_model = InferenceModel(self.model)

    def test_init(self):
        self.assertEqual(self.inference_model.model, self.model.model)
        self.assertEqual(
            self.inference_model.feature_names_, self.model.feature_names_
        )
        self.assertEqual(
            self.inference_model.feature_importances_,
            self.model.feature_importances_,
        )

    def test_predict(self):
        self.inference_model.model.predict = MagicMock(return_value=[1, 2, 3])
        X = MagicMock()
        self.assertEqual(self.inference_model.predict(X), [1, 2, 3])

    def test_predict_proba(self):
        self.inference_model.model.predict_proba = MagicMock(return_value=[1, 2, 3])
        X = MagicMock()
        self.assertEqual(self.inference_model.predict_proba(X), [1, 2, 3])

class TestlightgbmInferenceModel(unittest.TestCase):
    def setUp(self):
        self.model = MagicMock()
        self.model.model = MagicMock(spec=LGBMClassifier)

        self.model.model.fit = MagicMock()
        self.model.model.predict = MagicMock(return_value=[1, 2, 3])
        self.model.model.predict_proba = MagicMock(return_value=[1, 2, 3])


        self.model.feature_names_ = ["a", "b", "c"]
        self.model.feature_importances_ = [0.1, 0.2, 0.3]
        self.inference_model = lightgbmInferenceModel(self.model)

    def test_init(self):
        self.assertEqual(self.inference_model.model, self.model.model)
        self.assertEqual(
            self.inference_model.feature_names_, self.model.feature_names_
        )
        self.assertEqual(
            self.inference_model.feature_importances_,
            self.model.feature_importances_,
        )

    def test_predict(self):
        self.inference_model.model.predict = MagicMock(return_value=[1, 2, 3])
        X = MagicMock()
        self.assertEqual(self.inference_model.predict(X), [1, 2, 3])

    def test_predict_proba(self):
        self.inference_model.model.predict_proba = MagicMock(return_value=[1, 2, 3])
        X = MagicMock()
        self.assertEqual(self.inference_model.predict_proba(X), [1, 2, 3])
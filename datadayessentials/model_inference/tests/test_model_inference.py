import unittest
from io import StringIO
import os

import numpy as np
from datadayessentials.model_inference import (
    IServiceHitter,
    ServiceHitterCacher,
)
from unittest.mock import Mock, patch
from datadayessentials.model_inference._base import Models
import pytest
import pandas as pd

from .test_data import test_data_path



class TestIServiceHitter:
    @pytest.fixture
    def MockServiceHitter(self):
        with patch.object(IServiceHitter, "__init__", lambda _: None):
            with patch.object(IServiceHitter, "__abstractmethods__", set()):
                yield IServiceHitter()

    @patch("datadayessentials.data_retrieval.DataLakeCSVSaver.save")
    @patch("datadayessentials.data_retrieval.DataLakeDirectoryDeleter.delete_directory")
    def test_save_files_to_location(
        self, MockDLDeleter, MockDLCSVSave, MockServiceHitter
    ):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        models_to_invoke = ['1']
        account = "testaccount"
        container = "testcontainer"
        filepath = "testfilepath/nestedtestfilepath"
        MockServiceHitter.data_lake_authentication = Mock()
        actual_dir = MockServiceHitter._save_files_to_location(
            df, models_to_invoke, account, container, filepath
        )

        assert (
            actual_dir
            == "https://testaccount.blob.core.windows.net/testcontainer/testfilepath/nestedtestfilepath"
        )


class TestServiceHitterCacher:
    @pytest.fixture
    def combined_payload(self):
        example_df = pd.DataFrame({"ApplicationId": [1, 2, 3, 4, 5]})
        for i in range(1500):
            example_df[f"feature_{i}"] = [1, 2, 3, 4, 5]
        return example_df

    @pytest.fixture
    def model_predictions(self):
        return pd.DataFrame(
            {
                "ApplicationId": [1, 2, 3, 4, 5],
                "model1.score1": [1, 2, 3, 4, 5],
                "model1.score2": [1, 2, 3, 4, 5],
                "model2.score1": [1, 2, 3, 4, 5],
                "model2.score2": [1, 2, 3, 4, 5],
            }
        )

    def test_save_to_cache(self, combined_payload, model_predictions):
        # Save the above fixtures to the cache
        model_versions = ["model1", "model2"]
        cacher = ServiceHitterCacher(model_versions)
        model_1_cache_filename = os.path.join(cacher.cache_dir, "model1.csv")
        model_2_cache_filename = os.path.join(cacher.cache_dir, "model2.csv")
        if os.path.exists(model_1_cache_filename):
            os.remove(model_1_cache_filename)
        if os.path.exists(model_2_cache_filename):
            os.remove(model_2_cache_filename)
        cacher.save_to_cache(combined_payload, model_predictions)
        assert os.path.exists(model_1_cache_filename)
        assert os.path.exists(model_2_cache_filename)

        model_1_cache = cacher._load_model_cache("model1")
        model_2_cache = cacher._load_model_cache("model2")

        assert "hash" in model_1_cache.columns
        assert len(model_1_cache.columns) == 4
        assert "hash" in model_2_cache.columns
        assert len(model_2_cache.columns) == 4

        # Now test that the cache can be updated, with a single new entry and one already existing entry
        new_rows = pd.DataFrame(
            {
                "ApplicationId": [5, 7],
                "model1.score1": [5, 2],
                "model1.score2": [5, 2],
            }
        )

        new_payload = pd.DataFrame(
            {
                "ApplicationId": [5, 7],
            }
        )
        for i in range(1500):
            new_payload[f"feature_{i}"] = [5, 5]

        cacher.save_to_cache(new_payload, new_rows)

        model_versions = ["model1"]
        cacher = ServiceHitterCacher(model_versions)

        model_1_cache = cacher._load_model_cache("model1")

        assert len(model_1_cache) == 6

        os.remove(model_1_cache_filename)
        os.remove(model_2_cache_filename)

    def test_get_cached_results(self, combined_payload, model_predictions):
        # Save the above fixtures to the cache
        model_versions = ["model1", "model2"]
        cacher = ServiceHitterCacher(model_versions)
        model_1_cache_filename = os.path.join(cacher.cache_dir, "model1.csv")
        model_2_cache_filename = os.path.join(cacher.cache_dir, "model2.csv")

        if os.path.exists(model_1_cache_filename):
            os.remove(model_1_cache_filename)
        if os.path.exists(model_2_cache_filename):
            os.remove(model_2_cache_filename)

        cacher.save_to_cache(combined_payload, model_predictions)
        assert os.path.exists(model_1_cache_filename)
        assert os.path.exists(model_2_cache_filename)

        assert cacher._check_model_scores_cached("model1", combined_payload) == True
        assert cacher._check_model_scores_cached("model2", combined_payload) == True

        # Attempt to retrieve the model predictions from just the payload
        retrieved_model_predictions = cacher.get_cached_results(combined_payload)
        print(retrieved_model_predictions.head())
        print(combined_payload.head())
        assert retrieved_model_predictions.shape == (5, 5)

        # Attempt to retrieve the models predictions when some app ids havent been cached

        new_payload = pd.DataFrame({"ApplicationId": [1, 2, 3, 4, 6]})
        for i in range(1500):
            new_payload[f"feature_{i}"] = [1, 2, 3, 4, 6]

        assert cacher._check_model_scores_cached("model1", new_payload) == False
        assert cacher._check_model_scores_cached("model2", new_payload) == False

        with pytest.raises(ValueError):
            cacher.get_cached_results(new_payload)

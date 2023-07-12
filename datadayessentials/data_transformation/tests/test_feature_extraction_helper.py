import pytest
from .._feature_extraction_helper import FeatureExtractor


class TestPreProcessingHelper:
    def test_get_available_categoricals(self):
        helper = FeatureExtractor()

        available_fields = ["one", "two", "three", "four", "five"]
        categorical_features = ["one", "three", "ten"]

        actual = helper._get_available_categoricals(
            categorical_features, available_fields
        )
        expected = ["one", "three"]

        assert set(actual) == set(expected)

        available_fields = ["one", "two", "three", "four", "five"]
        categorical_features = ["ten"]

        actual = helper._get_available_categoricals(
            categorical_features, available_fields
        )
        expected = []
        assert set(actual) == set(expected)

        available_fields = ["one", "two", "three", "four", "five"]
        categorical_features = []

        actual = helper._get_available_categoricals(
            categorical_features, available_fields
        )
        expected = []
        assert set(actual) == set(expected)

    def test_get_available_features(self):
        helper = FeatureExtractor()

        available_fields = ["one", "two", "three", "four", "five"]
        requested_features = ["one", "three", "ten"]

        actual = helper._get_available_categoricals(
            requested_features, available_fields
        )
        expected = ["one", "three"]

        assert set(actual) == set(expected)

        available_fields = ["one", "two", "three", "four", "five"]
        requested_features = []

        actual = helper._get_available_features(requested_features, available_fields)
        expected = ["one", "two", "three", "four", "five"]

        assert set(actual) == set(expected)

    def test_remove_unwanted_features(self):
        helper = FeatureExtractor()

        available_fields = ["one", "two", "three", "four", "five"]
        unwanted_features = ["one", "three", "ten"]

        actual = helper._remove_unwanted_features(unwanted_features, available_fields)
        expected = ["two", "four", "five"]

        assert set(actual) == set(expected)

    def test_validate_request(self):
        helper = FeatureExtractor()

        # required and unwanted have crossover
        available_fields = ["one", "two", "three", "four", "five"]
        target = "two"
        unwanted_features = ["one", "three", "ten"]
        required_features = ["one", "three", "ten", target]

        with pytest.raises(ValueError):
            helper._validate_request(
                unwanted_features=unwanted_features,
                dataset_features=available_fields,
                required_features=required_features,
                target=target,
            )

        # one of the required features is not in available features
        available_fields = ["one", "two", "three", "four", "five"]
        target = "two"
        unwanted_features = []
        required_features = ["ten", target]

        try:
            helper._validate_request(
                unwanted_features=unwanted_features,
                dataset_features=available_fields,
                required_features=required_features,
                target=target,
            )
        except Exception as e:
            assert False, f"'_validate_request' raised an exception {e}"

        # handles missing required fields as long as some fields are returned
        available_fields = ["one", "two", "three", "four", "five"]
        target = "two"
        unwanted_features = []
        required_features = ["one", "three", "ten", target]

        # target is in unwanted fields
        available_fields = ["one", "two", "three", "four", "five"]
        target = "two"
        unwanted_features = [target]
        required_features = []

        with pytest.raises(ValueError):
            helper._validate_request(
                unwanted_features=unwanted_features,
                dataset_features=available_fields,
                required_features=required_features,
                target=target,
            )

        # target is not in available_fields
        available_fields = ["one", "two", "three", "four", "five"]
        target = "ten"
        unwanted_features = []
        required_features = []

        with pytest.raises(ValueError):
            helper._validate_request(
                unwanted_features=unwanted_features,
                dataset_features=available_fields,
                required_features=required_features,
                target=target,
            )

        with pytest.raises(ValueError):
            helper._validate_request(
                unwanted_features=unwanted_features,
                dataset_features=available_fields,
                required_features=required_features,
                target=target,
            )

    def test_run(self):
        helper = FeatureExtractor()

        # correct setup works
        available_fields = ["one", "two", "three", "four", "five"]
        unwanted_features = ["five"]
        required_features = []
        categorical_features = ["one", "three"]
        target = "two"

        actual_features, actual_categoricals = helper.run(
            unwanted_features=unwanted_features,
            dataset_features=available_fields,
            categorical_features=categorical_features,
            required_features=required_features,
            target=target,
        )

        expected_features = ["one", "two", "three", "four"]
        expected_categoricals = ["one", "three"]

        assert set(expected_features) == set(actual_features)
        assert set(expected_categoricals) == set(actual_categoricals)

        # target is not in the available fields
        available_fields = ["one", "two", "three", "four", "five"]
        unwanted_features = ["five"]
        required_features = ["one"]
        categorical_features = ["one", "three"]
        target = "ten"

        with pytest.raises(ValueError):
            actual_features, actual_categoricals = helper.run(
                unwanted_features=unwanted_features,
                dataset_features=available_fields,
                categorical_features=categorical_features,
                required_features=required_features,
                target=target,
            )
        available_fields = ["one", "two", "three", "four", "five"]
        unwanted_features = ["five"]
        required_features = ["one"]
        categorical_features = ["one", "three"]
        target = "ten"

        with pytest.raises(ValueError):
            actual_features, actual_categoricals = helper.run(
                unwanted_features=unwanted_features,
                dataset_features=available_fields,
                categorical_features=categorical_features,
                required_features=required_features,
                target=target,
            )

        # required_features returns only required fields - no categoricals
        available_fields = ["one", "two", "three", "four", "five"]
        unwanted_features = []
        required_features = ["two"]
        categorical_features = ["one", "three"]
        target = "two"

        actual_features, actual_categoricals = helper.run(
            unwanted_features=unwanted_features,
            dataset_features=available_fields,
            categorical_features=categorical_features,
            required_features=required_features,
            target=target,
        )

        expected_features = ["one", "two", "three", "four"]
        expected_categoricals = []

        # required_features returns only required fields - with categoricals
        available_fields = ["one", "two", "three", "four", "five"]
        unwanted_features = []
        required_features = ["two"]
        categorical_features = ["one", "two"]
        target = "two"

        actual_features, actual_categoricals = helper.run(
            unwanted_features=unwanted_features,
            dataset_features=available_fields,
            categorical_features=categorical_features,
            required_features=required_features,
            target=target,
        )

        expected_features = ["one", "two", "three", "four"]
        expected_categoricals = []

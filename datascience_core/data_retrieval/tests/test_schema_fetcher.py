from .._schema_fetcher import SchemaFetcher, InvalidSchemaError
from io import StringIO
import pytest
from unittest.mock import patch
import json
import unittest
import os


class TestSchemaFetcher(unittest.TestCase):
    @patch("datascience_core.data_retrieval.ProjectDatasetManager.load_datasets")
    def test_get_schema(self, load_datasets_patch):
        test_schema = {
            "name_not_important": {
                "field_name1": {
                    "description": "",
                    "min_val": "0",
                    "max_val": "100",
                    "unique_categories": [],
                    "dtype": "int",
                }
            }
        }
        load_datasets_patch.return_value = test_schema

        schema_fetcher = SchemaFetcher()
        schema = schema_fetcher.get_schema("name_not_important")
        self.assertDictEqual(schema, test_schema["name_not_important"])

    def test_invalid_schema(self):
        # Missing min_val field
        test_schema = {
            "field_name1": {
                "description": "",
                "max_val": "100",
                "unique_categories": [],
                "dtype": "int",
            }
        }

        schema_fetcher = SchemaFetcher()
        with pytest.raises(InvalidSchemaError):
            schema = schema_fetcher._validate_schema(test_schema)

    def test_schema_doesnt_exist(self):
        schema_fetcher = SchemaFetcher()
        with pytest.raises(ValueError):
            schema_fetcher.get_schema("no_schema_is_called_this")

    def test_get_app_and_cra_schema(self):
        schema_fetcher = SchemaFetcher()
        combined_schema = schema_fetcher.get_schema("app_and_cra_payload")
        assert len(combined_schema.keys()) == 2122

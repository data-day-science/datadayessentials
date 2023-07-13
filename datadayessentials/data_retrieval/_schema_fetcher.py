from ._base import ISchemaFetcher
import os
import json
import logging
from typing import List
from datadayessentials.data_retrieval import ProjectDatasetManager

logger = logging.getLogger(__name__)


class InvalidSchemaError(Exception):
    ...


class SchemaFetcher(ISchemaFetcher):
    """
    For loading schemas from the project dataset manager. The _required_fields indicate the neccessary fields for each field in the schema. See below for an example schema, for data with a single field:

    {
        "QCB BAC435": {
            "description": "All Credit cards Utilisation (%) last 1 month (live)",
            "unique_categories": [],
            "is_date": false,
            "min_val": "0",
            "max_val": "999999999",
            "dtype": "str"
    }



    To load these schemas, you can pass the schema name (model) as an argument

    In the below example the source and target schemas are loaded using the SchemaFetcher, which looks for a schema of the format (name) stored in the ProjectDatasetManager project "datadayessentials_schemas" in the data_retrieval module. Please see the SchemaFetcher class for the format schemas should follow.
    ```python
    from datadayessentials.data_retrieval import DataFrameTap, TableLoader, SchemaFetcher
    from datadayessentials.data_retrieval import TableLoader

    authentication = DWHAuthentication()
    sql_statement = "SELECT * FROM table_1"
    loader = TableLoader(authentication, sql_statement)

    source_schema = SchemaFetcher('source_schema')
    target_schema = SchemaFetcher('target_schema')

    data_tap = DataFrameTap(authentication, source_schema, target_schema)
    casted_data = data_tap.run()
    ```
    """

    _required_fields = [
        "min_val",
        "max_val",
        "unique_categories",
        "description",
        "dtype",
    ]

    def __init__(self):
        self.schema_manager = ProjectDatasetManager("datadayessentials_schemas")

    def add_schema(self, schema_name: str, schema: dict) -> None:
        """Add a schema to the schemas folder

        Arguments:
            schema_name (str): Name of the schema to be saved
            schema (dict): Schema to be saved
        """
        self._validate_schema(schema)
        self.schema_manager.register_dataset(schema_name, schema)

    def _load_schema(self, schema_name):
        return self.schema_manager.load_datasets(get_these_datasets=[schema_name])[
            schema_name
        ]

    def get_schema(self, name: str) -> dict:
        """Retrieve a saved schema from the schemas folder in this package

        The input nameshould be either the name of the file or one of the following combined schemas:
        - 'app_and_cra_payload' (combines app + qcb schemas)

        Arguments:
            name (str): Name of the schema in the schemas folder

        Raises:
            FileNotFoundError: Raised if the schema requested cannot be found

        Returns:
            dict: Dictionary with the schema
        """
        try:
            schema = self._load_schema(name)
            self._validate_schema(schema)
        except:
            raise ValueError("no schema found with that name")
        return schema

    @classmethod
    def list_available_schemas(cls) -> List[str]:
        """Returns the available schemas in the schemas folder plus any extra combined schemas

        Returns:
            List[str]: List of schema filenames
        """
        schema_manager = ProjectDatasetManager("datadayessentials_schemas")
        return schema_manager.list_datasets() 

    def _validate_schema(self, schema: dict):
        """Validates the schema to ensure all the required fields are present
        Arguments:
            schema (dict): Dictionary of the schema to validate
        Raises:
            InvalidSchemaError: If the schema is not valid this error is raised
        """
        for field_name in schema.keys():
            for key in self._required_fields:
                if key not in schema[field_name].keys():
                    error = f"Schema not in expected format. Field {field_name} is missing {key}.\nMust follow the following example format, with minimum required fields min_val, max_val, unique_categories, dtype, description:\n"
                    error += """{
                        "QCB BAC435": {
                        "Description": "All Credit cards Utilisation (%) last 1 month (live)",
                        "unique_categories": [],
                        "is_date": false,
                        "min_val": "0",
                        "max_val": "999999999",
                        "dtype": "str"
                        }
                    }"""
                    raise InvalidSchemaError(error)

from ._base import ISchemaFetcher
import os
import json
import logging
from typing import List

logger = logging.getLogger(__name__)


class InvalidSchemaError(Exception):
    ...


class SchemaFetcher(ISchemaFetcher):
    """
    For loading schemas from the schemas folder. The _required_fields indicate the neccessary fields for each field in the schema. See below for an example schema, for data with a single field:

    {
        "QCB BAC435": {
            "description": "All Credit cards Utilisation (%) last 1 month (live)",
            "unique_categories": [],
            "is_date": false,
            "min_val": "0",
            "max_val": "999999999",
            "dtype": "str"
    }

    Different models versions can have different input schemas and these can be stored in the schemas file as:
    - 'model_v1.json'
    - 'model_v2.json'

    To load these schemas, you can pass the schema name (model) and the model_version (v1, v2) as arguments

    In the below example the source and target schemas are loaded using the SchemaFetcher, which looks for a schema of the format (name.json) stored in the schemas folder in the data_retrieval module. Please see the SchemaFetcher class for the format schemas should follow.
    ```python
    from datascience_core.data_retrieval import DataFrameTap, TableLoader, SchemaFetcher
    from datascience_core.data_retrieval import TableLoader

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

    def get_schema(self, name: str, model_version: str = "") -> dict:
        """Retrieve a saved schema from the schemas folder in this package

        The input nameshould be either the name of the file or one of the following combined schemas:
        - 'app_and_cra_payload' (combines app + bsb + qcb schemas)

        Arguments:
            name (str): Name of the schema in the schemas folder
            model_version (str, optional): Model version. Defaults to "".

        Raises:
            FileNotFoundError: Raised if the schema requested cannot be found

        Returns:
            dict: Dictionary with the schema
        """
        if name == "app_and_cra_payload":
            filenames = ["app_schema.json", "BSB_schema.json", "QCB_schema.json"]
            schema = dict()
            for filename in filenames:
                schema_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "schemas", filename
                )
                with open(schema_path, "r") as f:
                    schema = {**schema, **json.loads(f.read())}
            self._validate_schema(schema)
            return schema
        else:
            if model_version:
                filename = name + "_" + model_version + ".json"
            else:
                filename = name + ".json"

            schema_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "schemas", filename
            )
            if not os.path.exists(schema_path):
                raise FileNotFoundError(
                    f"No schema with name {filename}, are you sure it exists and is in the datascience_core/data_retrieval/schemas folder?"
                )

            with open(schema_path, "r") as f:
                schema = json.loads(f.read())
                logger.debug(f"Loaded in schema {schema}")
                self._validate_schema(schema)
            return schema

    @classmethod
    def list_available_schemas(cls) -> List[str]:
        """Returns the available schemas in the schemas folder plus any extra combined schemas

        Returns:
            List[str]: List of schema filenames
        """
        extra_schemas = ["app_and_cra_payload"]
        schemas = [
            filename.split(".")[0]
            for filename in os.listdir(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "schemas")
            )
            if filename[-4:] == "json"
        ]
        return schemas + extra_schemas

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
